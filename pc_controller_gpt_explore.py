"""GPT VLM を使った探索＋ダイレクト制御コントローラ。

ターゲット（黄色い TARGET 箱）が画面外にあっても、スキャンしながら前進・再配置を繰り返し、
見つけ次第接近して停止する。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)

ALLOWED_COMMANDS = ["LEFT", "RIGHT", "FORWARD", "FORWARD_SLOW", "STOP"]

# 探索に特化したデフォルトプロンプト
DEFAULT_INSTRUCTION = """You are the navigation brain of a small robot car.
Goal: find a yellow box labeled "TARGET" anywhere in the room, then stop in front of it.

Behavior policy (reply with only one token: LEFT, RIGHT, FORWARD, FORWARD_SLOW, STOP):
- If TARGET is clearly visible: move toward it (FORWARD or FORWARD_SLOW). When it is very near or fills most of the frame, issue STOP.
- If TARGET is NOT visible: keep exploring, do NOT stay stopped just because you can't see it.
  * Do two small scans (LEFT/RIGHT), then one FORWARD_SLOW to change position.
  * After 3–4 scan cycles with no TARGET, relocate: a wider RIGHT turn or 2× FORWARD, then resume scanning.
  * Never repeat the same turn more than 2 times in a row; alternate directions.
  * If you see mostly wall/floor nearby (dead end), back out with a small RIGHT turn then FORWARD_SLOW.
- If the view is too dark or completely uncertain: STOP.
"""


# --------------------------------------------------------------------------- #
# ユーティリティ
# --------------------------------------------------------------------------- #
def _ensure_env() -> None:
    # .env があれば読み込む（OPENAI_API_KEY 用）
    load_dotenv()


def _to_data_url(frame_bgr) -> str:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = cv2.imencode(".jpg", np.array(img))[1].tobytes()
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _parse_command_from_text(text: str) -> Tuple[str, float]:
    """JSON があれば使い、なければ単語抽出でフォールバック。"""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "command" in data:
            cmd = str(data["command"]).upper()
            conf = float(data.get("confidence", 0.0))
            return cmd, conf
    except Exception:
        pass
    tokens = text.upper().replace("\n", " ").split()
    joined = "".join(tokens)
    for cmd in ALLOWED_COMMANDS:
        if cmd in tokens or cmd in joined:
            return cmd, 0.0
    return "STOP", 0.0


# --------------------------------------------------------------------------- #
# VLM ラッパー
# --------------------------------------------------------------------------- #
@dataclass
class GptRunner:
    model_id: str
    temperature: Optional[float]
    max_new_tokens: int
    system_prompt: str

    def __post_init__(self) -> None:
        _ensure_env()
        self.client = OpenAI()

    def infer(self, frame_bgr, instruction: str) -> Tuple[str, float, str]:
        img_url = _to_data_url(frame_bgr)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            },
        ]
        resp = self.client.responses.create(
            model=self.model_id,
            input=messages,
            max_output_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        raw = resp.output_text
        cmd, conf = _parse_command_from_text(raw)
        return cmd, conf, raw


# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT VLM exploration direct controller")
    parser.add_argument("--agent-url", required=True, help="raspi_agent のベースURL")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="VLM へ渡す指示文（未指定なら探索プロンプト）",
    )
    parser.add_argument("--model-id", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--loop-interval", type=float, default=0.8, help="推論周期（秒）")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--save-raw", type=Path, help="VLM 生応答を逐次保存するファイル")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = RaspiAgentClient(args.agent_url)
    runner = GptRunner(
        model_id=args.model_id,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        system_prompt=(
            "You control a differential-drive robot car. "
            "Respond ONLY with JSON like {\"command\":\"LEFT\",\"confidence\":0.90} "
            "or with a single token (LEFT/RIGHT/FORWARD/FORWARD_SLOW/STOP). "
            "No explanations."
        ),
    )

    raw_file = None
    if args.save_raw:
        raw_file = open(args.save_raw, "a", encoding="utf-8")

    try:
        while True:
            start = time.perf_counter()
            frame = client.fetch_frame()
            if frame is None:
                LOG.warning("フレーム取得に失敗。再試行します。")
                time.sleep(args.loop_interval)
                continue

            cmd, conf, raw = runner.infer(frame, args.instruction)
            if raw_file:
                raw_file.write(raw + "\n")

            if cmd not in ALLOWED_COMMANDS:
                LOG.warning("未知コマンド '%s' → STOP に置換 (raw=%s)", cmd, raw)
                cmd = "STOP"

            if conf < args.min_confidence:
                LOG.info("信頼度 %.2f < %.2f: STOP に置換", conf, args.min_confidence)
                cmd = "STOP"

            LOG.info("送信: %s (conf=%.2f)", cmd, conf)
            client.send_command(cmd)

            elapsed = time.perf_counter() - start
            sleep = args.loop_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        LOG.info("停止シグナル。STOP を送信して終了。")
        client.send_command("STOP")
    finally:
        if raw_file:
            raw_file.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
