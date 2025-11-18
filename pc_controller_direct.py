"""PC-side controller where SmolVLM outputs direct motion commands."""

from __future__ import annotations

import argparse
import logging
import time
from typing import List, Sequence, Tuple

import cv2
from PIL import Image

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)
ALLOWED_COMMANDS = ["FORWARD", "FORWARD_SLOW", "LEFT", "RIGHT", "STOP"]
OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT"}


class CommandGovernor:
    """Simple hysteresis logic to avoid rapid oscillation."""

    def __init__(self, *, stop_confirmation_loops: int = 2) -> None:
        self._stop_confirmation = max(1, stop_confirmation_loops)
        self._pending_stop = 0
        self._last_command: str = "STOP"

    def filter(self, candidate: str) -> str:
        command = candidate
        prev = self._last_command

        # 1) Prevent immediate LEFT⇄RIGHT反転
        if prev in OPPOSITE and command == OPPOSITE[prev]:
            LOG.debug("ヒステリシス: %s → %s を抑制", prev, command)
            command = prev

        # 2) Require multiple STOP votes before完全停止
        if command == "STOP":
            self._pending_stop += 1
            if self._pending_stop < self._stop_confirmation and prev not in (None, "STOP"):
                LOG.debug("ヒステリシス: STOP の投票 %d/%d → %s を維持", self._pending_stop,
                          self._stop_confirmation, prev)
                command = prev
        else:
            self._pending_stop = 0

        self._last_command = command
        return command

    @property
    def last_command(self) -> str:
        return self._last_command


class SmolVLMActionPlanner:
    """Wrap mlx-vlm so it outputs one of the allowed commands (JSON preferred)."""

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        from mlx_vlm.generate import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load as mlx_load

        LOG.info("smolVLM モデル読込中: %s", model_id)
        self._generate = mlx_generate
        self._apply_chat_template = apply_chat_template
        self.model, self.processor = mlx_load(model_id, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def decide(self, frame_bgr, instruction: str) -> Tuple[str, float, str]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": self.user_prompt_template.format(
                    instruction=instruction,
                    options=", ".join(ALLOWED_COMMANDS),
                ),
            }
        )

        prompt = self._apply_chat_template(
            self.processor,
            self.model.config,
            messages,
            num_images=1,
            add_generation_prompt=True,
        )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._generate(
            self.model,
            self.processor,
            prompt=prompt,
            image=[Image.fromarray(rgb)],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            verbose=False,
        )
        text = getattr(result, "text", "")
        LOG.debug("smolVLM 生応答: %s", text)
        payload = _parse_command_json(text)
        if payload:
            command = str(payload.get("command", "STOP")).upper()
            confidence = float(payload.get("confidence", 0.0))
        else:
            command = _extract_command(text.upper().strip(), ALLOWED_COMMANDS)
            confidence = 0.0
        return command, confidence, text


def _extract_command(text: str, candidates: Sequence[str]) -> str:
    tokens = text.replace("\n", " ").split()
    joined = "".join(tokens)
    for cmd in candidates:
        if cmd in tokens or cmd in joined:
            return cmd
    return "STOP"


def _parse_command_json(text: str) -> dict:
    import json
    import re

    snippets = re.findall(r"\{.*?\}", text, re.DOTALL)
    for snippet in snippets:
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if "command" in data:
            return data
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLM direct controller")
    parser.add_argument("--agent-url", required=True, help="raspi_agent のベースURL")
    parser.add_argument("--instruction", required=True, help="ロボットへの自然言語指示")
    parser.add_argument("--model-id", default="mlx-community/SmolVLM2-500M-Video-Instruct-mlx")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--loop-interval", type=float, default=0.5, help="ループ周期（秒）。0.3〜0.5秒を推奨")
    parser.add_argument(
        "--stop-confirmation-loops",
        type=int,
        default=2,
        help="STOP を実行するために必要な連続投票数（ヒステリシス用）",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="この値を下回る信頼度の結果は無視して直前コマンドを維持",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.loop_interval < 0.2:
        LOG.warning("loop-interval=%.2f 秒は短すぎます。0.3〜0.5 秒を推奨", args.loop_interval)

    client = RaspiAgentClient(args.agent_url)
    planner = SmolVLMActionPlanner(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        system_prompt=(
            "You control a differential-drive robot car. Respond ONLY with JSON like "
            '{{"command":"FORWARD","confidence":0.90}}. '
            "The command must be exactly one of: LEFT, RIGHT, FORWARD, FORWARD_SLOW, STOP."),
        user_prompt_template=(
            "Instruction: {instruction}. Analyse the camera image, then reply ONLY with JSON "
            'of the form {{"command":"<ONE_OF_OPTIONS>","confidence":0.xx}}. '
            "Options: {options}. No prose, no extra text."),
    )

    governor = CommandGovernor(stop_confirmation_loops=args.stop_confirmation_loops)

    try:
        while True:
            frame = client.fetch_frame()
            if frame is None:
                time.sleep(args.loop_interval)
                continue
            command, confidence, raw_text = planner.decide(frame, args.instruction)
            if command not in ALLOWED_COMMANDS:
                recovered = _extract_command(raw_text.upper().strip(), ALLOWED_COMMANDS)
                if recovered:
                    LOG.warning("未知コマンド '%s' を再解釈して '%s' を採用します", command, recovered)
                    command = recovered
                else:
                    LOG.error("未知コマンド '%s' を無視します（生応答: %s）", command, raw_text)
                    command = governor.last_command
            if confidence < args.min_confidence:
                filtered = governor.last_command
                LOG.info(
                    "信頼度 %.2f < %.2f: 直前のコマンド %s を維持",
                    confidence,
                    args.min_confidence,
                    filtered,
                )
            else:
                filtered = governor.filter(command)
                if filtered != command:
                    LOG.debug("フィルタ後コマンド: %s → %s", command, filtered)
                LOG.info("送信コマンド: %s (conf=%.2f)", filtered, confidence)
            client.send_command(filtered)
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。STOP を送信します。")
        client.send_command("STOP")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
