"""GPT版ハイブリッド制御コントローラ（認識→ルールでモーター制御）。

- raspi_agent からフレームを取得
- OpenAI gpt-5-mini-2025-08-07 VLM に画像+指示を投げ、position/distance/confidence を JSON で受け取る
- ヒステリシスで位置・距離を安定化し、単純ルールでコマンド決定

既存 smolVLM 版 (pc_controller_hybrid.py) は記録として残し、同等 CLI で
モデルだけ差し替えできるようにする。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
from dotenv import load_dotenv
from openai import OpenAI

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)

OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT"}


def _ensure_env():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=False)


class CommandGovernor:
    def __init__(self, *, stop_confirmation_loops: int = 2) -> None:
        self._stop_confirmation = max(1, stop_confirmation_loops)
        self._pending_stop = 0
        self._last_command: str = "STOP"

    def filter(self, candidate: str) -> str:
        command = candidate
        prev = self._last_command

        if prev in OPPOSITE and command == OPPOSITE[prev]:
            LOG.debug("ヒステリシス: %s → %s を抑制", prev, command)
            command = prev

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


class EstimateHysteresis:
    def __init__(self, *, position_hold: int, distance_hold: int) -> None:
        self._position_hold = max(1, position_hold)
        self._distance_hold = max(1, distance_hold)
        self._stable_position = "CENTER"
        self._stable_distance = "FAR"
        self._pos_counter = 0
        self._dist_counter = 0
        self._last_position = "CENTER"
        self._last_distance = "FAR"

    def stabilize(self, estimate: "SpatialEstimate") -> "SpatialEstimate":
        pos = self._apply_position(estimate.position)
        dist = self._apply_distance(estimate.distance)
        return SpatialEstimate(pos, dist, estimate.confidence)

    def _apply_position(self, position: str) -> str:
        if position == self._stable_position:
            self._pos_counter = 0
            self._last_position = position
            return self._stable_position
        if position == self._last_position:
            self._pos_counter += 1
        else:
            self._pos_counter = 1
            self._last_position = position
        if self._pos_counter >= self._position_hold:
            LOG.debug("位置ヒステリシス: %s → %s", self._stable_position, position)
            self._stable_position = position
            self._pos_counter = 0
        return self._stable_position

    def _apply_distance(self, distance: str) -> str:
        if distance == self._stable_distance:
            self._dist_counter = 0
            self._last_distance = distance
            return self._stable_distance
        if distance == self._last_distance:
            self._dist_counter += 1
        else:
            self._dist_counter = 1
            self._last_distance = distance
        if self._dist_counter >= self._distance_hold:
            LOG.debug("距離ヒステリシス: %s → %s", self._stable_distance, distance)
            self._stable_distance = distance
            self._dist_counter = 0
        return self._stable_distance


@dataclass
class SpatialEstimate:
    position: str
    distance: str
    confidence: float


POSITION_RULES: Dict[str, str] = {
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "CENTER": "FORWARD",
}

DISTANCE_RULES: Dict[str, str] = {
    "FAR": "FORWARD",
    "MID": "FORWARD_SLOW",
    "NEAR": "STOP",
}


def _encode_image(frame_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class GptLocator:
    def __init__(
        self,
        model_id: str,
        *,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float,
        max_new_tokens: int,
    ) -> None:
        _ensure_env()
        self.client = OpenAI()
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def analyse(self, frame_bgr, instruction: str) -> SpatialEstimate:
        image_url = _encode_image(frame_bgr)
        user_content = [
            {"type": "text", "text": self.user_prompt_template.format(instruction=instruction)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": user_content},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        content = resp.choices[0].message.content
        if isinstance(content, list):
            raw_text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
        else:
            raw_text = content or ""
        LOG.debug("GPT 生応答: %s", raw_text)
        payload = _parse_json(raw_text)
        if not payload:
            return SpatialEstimate("CENTER", "FAR", 0.0)
        return SpatialEstimate(
            position=str(payload.get("position", "CENTER")).upper(),
            distance=str(payload.get("distance", "FAR")).upper(),
            confidence=float(payload.get("confidence", 0.0)),
        )


def _parse_json(text: str) -> Dict[str, object]:
    import re

    snippets = re.findall(r"\{.*?\}", text, re.DOTALL)
    for snippet in snippets:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return {}


def _decide_command(estimate: SpatialEstimate) -> str:
    pos_cmd = POSITION_RULES.get(estimate.position, "FORWARD")
    if pos_cmd != "FORWARD":
        return pos_cmd
    return DISTANCE_RULES.get(estimate.distance, "FORWARD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT hybrid controller")
    parser.add_argument("--agent-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model-id", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--loop-interval", type=float, default=0.5, help="ループ周期（秒）。0.3〜0.5秒推奨")
    parser.add_argument(
        "--stop-confirmation-loops",
        type=int,
        default=2,
        help="STOP を実行するための連続投票数",
    )
    parser.add_argument("--position-hold", type=int, default=2, help="位置判定を切り替えるのに必要な連続回数")
    parser.add_argument("--distance-hold", type=int, default=2, help="距離判定を切り替えるのに必要な連続回数")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--min-confidence", type=float, default=0.4)
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
    locator = GptLocator(
        args.model_id,
        system_prompt=(
            "You are a perception module for a robot car. Respond ONLY with JSON containing "
            "position (LEFT/CENTER/RIGHT/UNVISIBLE), distance (FAR/MID/NEAR/NONE) and confidence 0..1."
        ),
        user_prompt_template=(
            "Instruction: {instruction}. Describe where the yellow TARGET box appears in the image. "
            "Return JSON like {\"position\": \"LEFT\", \"distance\": \"FAR\", \"confidence\": 0.82}."
        ),
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    stabilizer = EstimateHysteresis(
        position_hold=args.position_hold,
        distance_hold=args.distance_hold,
    )
    governor = CommandGovernor(stop_confirmation_loops=args.stop_confirmation_loops)

    try:
        while True:
            frame = client.fetch_frame()
            if frame is None:
                time.sleep(args.loop_interval)
                continue
            estimate = locator.analyse(frame, args.instruction)
            LOG.debug("raw-estimate=%s", estimate)
            if estimate.confidence < args.min_confidence:
                hold = governor.last_command
                LOG.info(
                    "信頼度 %.2f < %.2f: 直前のコマンド %s を維持",
                    estimate.confidence,
                    args.min_confidence,
                    hold,
                )
                filtered = governor.filter(hold)
            else:
                smoothed = stabilizer.stabilize(estimate)
                cmd = _decide_command(smoothed)
                LOG.info(
                    "決定コマンド: %s (pos=%s dist=%s conf=%.2f)",
                    cmd,
                    smoothed.position,
                    smoothed.distance,
                    smoothed.confidence,
                )
                filtered = governor.filter(cmd)
            client.send_command(filtered)
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。STOP を送信します。")
        client.send_command("STOP")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
