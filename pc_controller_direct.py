"""PC-side controller where SmolVLM outputs direct motion commands."""

from __future__ import annotations

import argparse
import logging
import time
from typing import List, Sequence

import cv2
from PIL import Image

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)
ALLOWED_COMMANDS = ["FORWARD", "FORWARD_SLOW", "LEFT", "RIGHT", "STOP"]


class SmolVLMActionPlanner:
    """Wrap mlx-vlm so it outputs one of the allowed commands."""

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

    def decide(self, frame_bgr, instruction: str) -> str:
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
        text = getattr(result, "text", "").upper().strip()
        LOG.debug("smolVLM 生応答: %s", text)
        return _extract_command(text, ALLOWED_COMMANDS)


def _extract_command(text: str, candidates: Sequence[str]) -> str:
    tokens = text.replace("\n", " ").split()
    joined = "".join(tokens)
    for cmd in candidates:
        if cmd in tokens or cmd in joined:
            return cmd
    return "STOP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLM direct controller")
    parser.add_argument("--agent-url", required=True, help="raspi_agent のベースURL")
    parser.add_argument("--instruction", required=True, help="ロボットへの自然言語指示")
    parser.add_argument("--model-id", default="mlx-community/SmolVLM2-500M-Video-Instruct-mlx")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--loop-interval", type=float, default=1.0, help="ループ周期（秒）")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = RaspiAgentClient(args.agent_url)
    planner = SmolVLMActionPlanner(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        system_prompt=(
            "You control a differential-drive robot car with commands LEFT, RIGHT, FORWARD, "
            "FORWARD_SLOW, STOP. Respond with exactly one of those tokens."),
        user_prompt_template=(
            "Instruction: {instruction}. Analyse the provided camera image and reply with "
            "ONLY one command token from this list: {options}. No punctuation."),
    )

    try:
        while True:
            frame = client.fetch_frame()
            if frame is None:
                time.sleep(args.loop_interval)
                continue
            command = planner.decide(frame, args.instruction)
            LOG.info("決定コマンド: %s", command)
            client.send_command(command)
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。STOP を送信します。")
        client.send_command("STOP")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
