"""PC-side controller where SmolVLM outputs perception, rules decide motion."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Dict

import cv2
from PIL import Image

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)


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


class SmolVLMLocator:
    def __init__(
        self,
        model_id: str,
        *,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float,
        max_new_tokens: int,
    ) -> None:
        from mlx_vlm.generate import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load as mlx_load

        LOG.info("smolVLM モデル読込中: %s", model_id)
        self._generate = mlx_generate
        self._apply_chat_template = apply_chat_template
        self.model, self.processor = mlx_load(model_id, trust_remote_code=True)
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def analyse(self, frame_bgr, instruction: str) -> SpatialEstimate:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": self.user_prompt_template.format(instruction=instruction),
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
        payload = _parse_json(text)
        if not payload:
            return SpatialEstimate("CENTER", "FAR", 0.0)
        return SpatialEstimate(
            position=str(payload.get("position", "CENTER")).upper(),
            distance=str(payload.get("distance", "FAR")).upper(),
            confidence=float(payload.get("confidence", 0.0)),
        )


def _parse_json(text: str) -> Dict[str, object]:
    import json
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
    parser = argparse.ArgumentParser(description="SmolVLM hybrid controller")
    parser.add_argument("--agent-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model-id", default="mlx-community/SmolVLM2-500M-Video-Instruct-mlx")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--loop-interval", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    client = RaspiAgentClient(args.agent_url)
    locator = SmolVLMLocator(
        args.model_id,
        system_prompt=(
            "You are a perception module for a robot car. Respond ONLY with JSON containing "
            "position (LEFT/CENTER/RIGHT), distance (FAR/MID/NEAR) and confidence in 0..1."),
        user_prompt_template=(
            "Instruction: {instruction}. Describe where the requested target appears in the image. "
            "Return JSON like {{\"position\": \"LEFT\", \"distance\": \"FAR\", \"confidence\": 0.82}}."),
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    try:
        while True:
            frame = client.fetch_frame()
            if frame is None:
                time.sleep(args.loop_interval)
                continue
            estimate = locator.analyse(frame, args.instruction)
            LOG.debug("estimate=%s", estimate)
            if estimate.confidence < args.min_confidence:
                LOG.info("信頼度 %.2f が閾値未満のため STOP", estimate.confidence)
                client.send_command("STOP")
            else:
                cmd = _decide_command(estimate)
                LOG.info("決定コマンド: %s (pos=%s dist=%s)", cmd, estimate.position, estimate.distance)
                client.send_command(cmd)
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。STOP を送信します。")
        client.send_command("STOP")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
