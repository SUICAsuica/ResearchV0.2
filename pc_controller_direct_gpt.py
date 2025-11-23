"""GPT版ダイレクト制御コントローラ（モーター制御あり）。

- raspi_agent からフレームを取得
- OpenAI gpt-5-mini-2025-08-07（もしくは指定モデル）に画像+短い指示を投げ、
  「LEFT/RIGHT/FORWARD/FORWARD_SLOW/STOP」の単語ひとつを応答してもらう
- 受け取ったコマンドをそのまま /command API に送信

備考:
- 以前の JSON 固定プロンプトは空返答を起こしやすかったため撤廃。
  現在はシンプルな一語出力プロンプトで安定動作を狙う。
- 旧 smolVLM 版 (pc_controller_direct.py) は記録用に残し、CLI 互換でモデルだけ差し替え可能。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple

import cv2
from dotenv import load_dotenv
from openai import OpenAI

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)
ALLOWED_COMMANDS = ["FORWARD", "FORWARD_SLOW", "LEFT", "RIGHT", "STOP"]
DEFAULT_INSTRUCTION = (
    "Look at the image. Choose ONE from LEFT, RIGHT, FORWARD, FORWARD_SLOW, STOP. "
    "If the yellow TARGET box is not clearly visible or you are unsure, output FORWARD_SLOW."
)


def _ensure_env():
    # .env があれば自動で読む（既に設定済みのキーは上書きしない）
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=True)
    key = os.getenv("OPENAI_API_KEY")
    if key:
        LOG.debug("OPENAI_API_KEY loaded (last4=%s)", key[-4:])


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
            response_format={"type": "text"},
            reasoning_effort="minimal",
        )
        raw_text = ""

        def _extract_from_choice(ch):
            msg = ch.message
            txt = ""
            content_local = msg.content
            if isinstance(content_local, list):
                txt = "\n".join(
                    part.get("text", "") for part in content_local if part.get("type") in {"text", "output_text"}
                )
            elif content_local:
                txt = str(content_local)
            if (not txt) and getattr(msg, "tool_calls", None):
                tc = msg.tool_calls[0]
                try:
                    txt = tc.function.arguments or ""
                except Exception:
                    txt = ""
            LOG.debug(
                "GPT 生応答: %s (finish_reason=%s, tool_calls=%s)",
                txt,
                ch.finish_reason,
                bool(getattr(msg, "tool_calls", None)),
            )
            return txt

        try:
            resp = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            LOG.error("OpenAI API error: %s", exc)
            raise
        try:
            LOG.debug("choice0 dump: %s", resp.choices[0].model_dump())
        except Exception:
            pass
        try:
            LOG.debug("usage: %s", resp.usage)
        except Exception:
            pass
        raw_text = _extract_from_choice(resp.choices[0])

        # 空ならプロンプトをより単純化して 1 回だけリトライ
        if not raw_text.strip():
            LOG.debug("1st response empty. Retrying with minimal text prompt (no image).")
            alt_kwargs = dict(request_kwargs)
            alt_kwargs["messages"] = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Return ONLY one of LEFT, RIGHT, FORWARD, FORWARD_SLOW, STOP."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "If unsure, FORWARD_SLOW."}]},
            ]
            alt_kwargs["max_completion_tokens"] = self.max_new_tokens
            alt_kwargs["reasoning_effort"] = "minimal"
            alt_kwargs["response_format"] = {"type": "text"}
            resp_alt = self.client.chat.completions.create(**alt_kwargs)
            raw_text = _extract_from_choice(resp_alt.choices[0])

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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--loop-interval",
        type=float,
        default=10.0,
        help="ループ周期（秒）。gpt-5-mini の高レイテンシに合わせ 10 秒以上を推奨",
    )
    parser.add_argument(
        "--fallback-command",
        choices=ALLOWED_COMMANDS,
        default="FORWARD_SLOW",
        help="VLM 応答が空/解釈不能だった場合に送る安全コマンド",
    )
    parser.add_argument(
        "--save-frames-dir",
        default="frames",
        help="取得フレームの保存先ディレクトリ。空文字なら保存しない",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="ログ保存ディレクトリ。空文字にするとファイル出力なし",
    )
    parser.add_argument("--log-file", default=None, help="ログファイルを明示指定する場合")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handlers = [logging.StreamHandler()]

    log_path = None
    if args.log_file:
        log_path = Path(args.log_file)
    elif args.log_dir:
        log_path = Path(args.log_dir) / f"gpt-direct-{datetime.now():%Y%m%d-%H%M%S}.log"

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    if log_path:
        LOG.info("ログをファイルにも保存します: %s", log_path)

    if args.loop_interval < 5.0:
        LOG.warning("loop-interval=%.2f 秒は短すぎます。推論完了まで 10 秒以上かかる前提で設定してください", args.loop_interval)

    client = RaspiAgentClient(args.agent_url)
    planner = GptActionPlanner(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        system_prompt=(
            "You output only the next driving command as a single uppercase token from "
            "{LEFT, RIGHT, FORWARD, FORWARD_SLOW, STOP}. If the target is not visible, "
            "output FORWARD_SLOW to explore safely."
        ),
        user_prompt_template="{instruction}",
    )

    frame_dir = None
    if args.save_frames_dir:
        run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        frame_dir = Path(args.save_frames_dir) / run_tag
        frame_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("フレーム保存先: %s", frame_dir)

    try:
        while True:
            frame = client.fetch_frame()
            if frame is None:
                time.sleep(args.loop_interval)
                continue
            if frame_dir:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
                fname = frame_dir / f"frame-{ts}.jpg"
                cv2.imwrite(str(fname), frame)
            try:
                command, raw_text = planner.decide(frame, args.instruction)
            except Exception as exc:  # APIエラー等
                LOG.error("VLM 呼び出しに失敗: %s", exc)
                command, raw_text = args.fallback_command, ""

            if not raw_text or not str(raw_text).strip():
                LOG.warning("VLM 応答が空。フォールバック %s を送信", args.fallback_command)
                command = args.fallback_command

            if command not in ALLOWED_COMMANDS:
                recovered = _extract_command(raw_text.upper().strip(), ALLOWED_COMMANDS)
                if recovered:
                    LOG.warning("未知コマンド '%s' を再解釈して '%s' を採用します", command, recovered)
                    command = recovered
                else:
                    LOG.error(
                        "未知コマンド '%s' をフォールバック %s で置き換えます（生応答: %s）",
                        command,
                        args.fallback_command,
                        raw_text,
                    )
                    command = args.fallback_command
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
