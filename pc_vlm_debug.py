"""Utility script to run SmolVLM inference only (no motor commands)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Dict, Optional

import cv2
from PIL import Image

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SmolVLM debug runner (fetch frame, run VLM, print result)."
    )
    parser.add_argument("--agent-url", required=True, help="raspi_agent のベースURL")
    parser.add_argument("--instruction", required=True, help="VLM へ渡す指示文")
    parser.add_argument("--model-id", default="mlx-community/SmolVLM2-500M-Video-Instruct-mlx")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "direct"],
        default="hybrid",
        help="hybrid=位置/距離JSON, direct=コマンドJSON",
    )
    parser.add_argument("--loop-interval", type=float, default=1.0, help="ループ周期（秒）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="DEBUG にすると VLM の生テキストや JSON が表示されます",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="1フレームだけ推論して終了（デフォルトは継続取得）",
    )
    return parser.parse_args()


class SmolVLMRunner:
    """Wrapper around mlx-vlm that returns raw text + parsed payload."""

    def __init__(
        self,
        model_id: str,
        *,
        mode: str,
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
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.mode = mode

        if mode == "direct":
            self.system_prompt = (
                "You control a differential-drive robot car. Respond ONLY with JSON shaped like "
                '{{"command":"FORWARD","confidence":0.9}}. '
                "command must be one of LEFT, RIGHT, FORWARD, FORWARD_SLOW, STOP."
            )
            self.user_prompt_template = (
                "Instruction: {instruction}. Analyse the camera image, then reply ONLY with JSON "
                'of the form {{"command":"TOKEN","confidence":0.xx}} where TOKEN is one of: '
                "LEFT, RIGHT, FORWARD, FORWARD_SLOW, STOP. No prose."
            )
        else:
            self.system_prompt = (
                "You are estimating a TARGET box position (LEFT/CENTER/RIGHT) and distance "
                "(FAR/MID/NEAR). Respond ONLY with JSON like "
                '{{"position":"CENTER","distance":"FAR","confidence":0.9}}.'
            )
            self.user_prompt_template = (
                "Instruction: {instruction}. Describe the yellow TARGET box location as JSON "
                'with fields position, distance, confidence. position must be LEFT/CENTER/RIGHT; '
                "distance must be FAR/MID/NEAR."
            )

    def run(self, frame_bgr, instruction: str) -> Dict[str, object]:
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
        return {"raw": text, "json": payload}


def _parse_json(text: str) -> Optional[Dict[str, object]]:
    import re

    snippets = re.findall(r"\{.*?\}", text, re.DOTALL)
    for snippet in snippets:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return None


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = RaspiAgentClient(args.agent_url)
    runner = SmolVLMRunner(
        args.model_id,
        mode=args.mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    iteration = 0
    try:
        while True:
            frame = client.fetch_frame()
            if frame is None:
                LOG.warning("フレーム取得に失敗。再試行します。")
                time.sleep(args.loop_interval)
                continue

            result = runner.run(frame, args.instruction)
            iteration += 1
            LOG.info("---- 推論 #%d ----", iteration)
            print(json.dumps(result["json"], ensure_ascii=False, indent=2))
            LOG.info("raw=%s", result["raw"])

            if args.once:
                break
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。終了します。")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
