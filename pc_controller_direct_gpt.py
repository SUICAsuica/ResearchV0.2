"""GPT版ダイレクト制御コントローラ（モーター制御あり）。

- raspi_agent からフレームを取得
- OpenAI gpt-5-mini-2025-08-07 VLM に画像+指示を投げ、JSON の {"command": ...} を読み取る
- 受け取ったコマンドをそのまま /command API に送信

既存 smolVLM 版 (pc_controller_direct.py) は記録用に残しつつ、
同じ CLI IF でモデルだけ差し替えできるようにしている。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from pathlib import Path
from typing import Sequence, Tuple

import cv2
from dotenv import load_dotenv
from openai import OpenAI

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)
ALLOWED_COMMANDS = ["FORWARD", "FORWARD_SLOW", "LEFT", "RIGHT", "STOP"]
DEFAULT_INSTRUCTION = (
    "You are a robot car. Approach the yellow box labeled 'TARGET' and stop in front of it. "
    "Tell me concisely which way to move next. Respond only with JSON where the value is one of "
    "{forward, back, left, right}."
)
SYSTEM_PROMPT = 'Only return JSON {"command":"LEFT|RIGHT|FORWARD|FORWARD_SLOW|STOP"}.'


def _ensure_env():
    # .env があれば自動で読む（既に設定済みのキーは上書きしない）
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=False)


def _encode_image(frame_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class GptActionPlanner:
    """Call OpenAI VLM to get a motion command JSON."""

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        _ensure_env()
        self.client = OpenAI()
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def decide(self, frame_bgr, instruction: str) -> Tuple[str, str]:
        image_url = _encode_image(frame_bgr)
        user_content = [
            {"type": "text", "text": self.user_prompt_template.format(instruction=instruction)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        request_kwargs = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=self.max_new_tokens,
            response_format={"type": "json_object"},
        )
        resp = self.client.chat.completions.create(**request_kwargs)
        content = resp.choices[0].message.content
        if isinstance(content, list):
            raw_text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
        else:
            raw_text = content or ""
        LOG.debug("GPT 生応答: %s", raw_text)
        if not raw_text.strip():
            LOG.warning("GPT 応答が空です: %s", resp.choices[0].message)

        payload = _parse_command_json(raw_text)
        if payload:
            command = str(payload.get("command", "STOP")).upper()
        else:
            command = _extract_command(raw_text.upper().strip(), ALLOWED_COMMANDS)
        return command, raw_text


def _extract_command(text: str, candidates: Sequence[str]) -> str:
    tokens = text.replace("\n", " ").split()
    joined = "".join(tokens)
    for cmd in candidates:
        if cmd in tokens or cmd in joined:
            return cmd
    return "STOP"


def _parse_command_json(text: str) -> dict:
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
    parser = argparse.ArgumentParser(description="GPT direct controller")
    parser.add_argument("--agent-url", required=True, help="raspi_agent のベースURL")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="ロボットへの自然言語指示（未指定なら英語固定文）",
    )
    parser.add_argument("--model-id", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--loop-interval",
        type=float,
        default=10.0,
        help="ループ周期（秒）。gpt-5-mini の高レイテンシに合わせ 10 秒以上を推奨",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.loop_interval < 5.0:
        LOG.warning("loop-interval=%.2f 秒は短すぎます。推論完了まで 10 秒以上かかる前提で設定してください", args.loop_interval)

    client = RaspiAgentClient(args.agent_url)
    planner = GptActionPlanner(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template="{instruction}",
    )

    try:
        while True:
            frame = client.fetch_frame()
            if frame is None:
                time.sleep(args.loop_interval)
                continue
            command, raw_text = planner.decide(frame, args.instruction)
            if command not in ALLOWED_COMMANDS:
                recovered = _extract_command(raw_text.upper().strip(), ALLOWED_COMMANDS)
                if recovered:
                    LOG.warning("未知コマンド '%s' を再解釈して '%s' を採用します", command, recovered)
                    command = recovered
                else:
                    LOG.error("未知コマンド '%s' を STOP で置き換えます（生応答: %s）", command, raw_text)
                    command = "STOP"
            LOG.info("送信コマンド: %s", command)
            LOG.debug("GPT raw: %s", raw_text)
            client.send_command(command)
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。STOP を送信します。")
        client.send_command("STOP")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
