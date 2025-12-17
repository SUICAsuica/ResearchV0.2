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
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
from dotenv import load_dotenv
from openai import OpenAI

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)
DEFAULT_INSTRUCTION = (
    "You are a detector, not a driver. Look at the image and report only where the yellow box "
    "(with the word TARGET) appears and how far it is. Return ONLY JSON "
    "{\"position\":\"LEFT|CENTER|RIGHT|UNVISIBLE\",\"distance\":\"FAR|MID|NEAR|NONE\",\"confidence\":0-1}. "
    "If the box is not visible, use UNVISIBLE/NONE/0.0. Never output driving commands or prose."
)

OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
ALLOWED_POSITIONS = {"LEFT", "CENTER", "RIGHT", "UNVISIBLE"}
ALLOWED_DISTANCES = {"FAR", "MID", "NEAR", "NONE"}
ALLOWED_COMMANDS = {"FORWARD", "FORWARD_SLOW", "LEFT", "RIGHT", "STOP"}


def _ensure_env():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=True)
    key = os.getenv("OPENAI_API_KEY")
    if key:
        LOG.debug("OPENAI_API_KEY loaded (last4=%s)", key[-4:])


class CommandGovernor:
    def __init__(self, *, stop_confirmation_loops: int = 1) -> None:
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
    # ターゲットが見えないときは安全に停止
    "UNVISIBLE": "STOP",
}

DISTANCE_RULES: Dict[str, str] = {
    "FAR": "FORWARD",
    "MID": "FORWARD_SLOW",
    "NEAR": "STOP",
    # 距離不明時も暴走を避ける
    "NONE": "STOP",
}


def _encode_image(frame_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _extract_from_choice(ch) -> str:
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
        "GPT 応答抽出: %s (finish_reason=%s, tool_calls=%s)",
        txt,
        ch.finish_reason,
        bool(getattr(msg, "tool_calls", None)),
    )
    return txt


class GptLocator:
    def __init__(
        self,
        model_id: str,
        *,
        system_prompt: str,
        user_prompt_template: str,
        max_new_tokens: int,
    ) -> None:
        _ensure_env()
        self.client = OpenAI()
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.max_new_tokens = max_new_tokens

    def analyse(self, frame_bgr, instruction: str) -> tuple[SpatialEstimate, bool]:
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
        try:
            resp = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            LOG.error("OpenAI API error: %s", exc)
            raise

        try:
            LOG.debug("choice0 dump: %s", resp.choices[0].model_dump())
        except Exception:
            pass

        raw_text = _extract_from_choice(resp.choices[0])

        if not raw_text.strip():
            LOG.debug("1st response empty. Retrying with minimal text prompt (no image).")
            alt_kwargs = dict(request_kwargs)
            alt_kwargs["messages"] = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Return ONLY JSON {\"position\":\"LEFT|CENTER|RIGHT|UNVISIBLE\","
                                "\"distance\":\"FAR|MID|NEAR|NONE\",\"confidence\":0-1}."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "If unsure, use UNVISIBLE/NONE/0.0. No prose.",
                        }
                    ],
                },
            ]
            alt_kwargs["max_completion_tokens"] = self.max_new_tokens
            alt_kwargs["response_format"] = {"type": "text"}
            alt_kwargs["reasoning_effort"] = "minimal"
            resp_alt = self.client.chat.completions.create(**alt_kwargs)
            raw_text = _extract_from_choice(resp_alt.choices[0])

        LOG.debug("GPT 生応答: %s", raw_text)
        payload = _parse_json(raw_text)
        if not payload:
            return SpatialEstimate("UNVISIBLE", "NONE", 0.0), False
        return _sanitize_estimate(payload), True


def _parse_json(text: str) -> Dict[str, object]:
    import re

    snippets = re.findall(r"\{.*?\}", text, re.DOTALL)
    for snippet in snippets:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return {}


def _sanitize_estimate(payload: Dict[str, object]) -> SpatialEstimate:
    pos = str(payload.get("position", "CENTER")).upper()
    dist = str(payload.get("distance", "FAR")).upper()
    conf_raw = payload.get("confidence", 0.0)
    try:
        conf = float(conf_raw)
    except Exception:
        conf = 0.0

    if pos not in ALLOWED_POSITIONS:
        LOG.debug("位置値 %s を CENTER に補正", pos)
        pos = "CENTER"
    if dist not in ALLOWED_DISTANCES:
        LOG.debug("距離値 %s を FAR に補正", dist)
        dist = "FAR"
    conf = max(0.0, min(1.0, conf))
    return SpatialEstimate(pos, dist, conf)


def _decide_command(estimate: SpatialEstimate) -> str:
    pos_cmd = POSITION_RULES.get(estimate.position, "FORWARD")
    if pos_cmd != "FORWARD":
        return pos_cmd
    return DISTANCE_RULES.get(estimate.distance, "FORWARD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT hybrid controller")
    parser.add_argument("--agent-url", required=True)
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
        help="ループ周期（秒）。gpt-5-mini の応答遅延に合わせ 10 秒以上を推奨",
    )
    parser.add_argument(
        "--stop-confirmation-loops",
        type=int,
        default=1,
        help="STOP 実行に必要な連続投票数 (1 でヒステリシス無効化)",
    )
    parser.add_argument(
        "--position-hold",
        type=int,
        default=1,
        help="位置判定を切り替えるのに必要な連続回数 (1 でヒステリシス無効化)",
    )
    parser.add_argument(
        "--distance-hold",
        type=int,
        default=1,
        help="距離判定を切り替えるのに必要な連続回数 (1 でヒステリシス無効化)",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--fallback-command",
        choices=["FORWARD", "FORWARD_SLOW", "LEFT", "RIGHT", "STOP"],
        default="STOP",
        help="VLM 応答が空/解釈不能/低信頼のときに送る安全コマンド",
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
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--use-confidence-threshold",
        action="store_true",
        help="true のとき min-confidence 未満なら常にフォールバックコマンドを送る",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handlers = [logging.StreamHandler()]

    log_path = None
    if args.log_file:
        log_path = Path(args.log_file)
    elif args.log_dir:
        log_path = Path(args.log_dir) / f"gpt-hybrid-{datetime.now():%Y%m%d-%H%M%S}.log"

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
        LOG.warning("loop-interval=%.2f 秒は短すぎます。推論完了まで 10 秒程度かかる前提で設定してください", args.loop_interval)
    client = RaspiAgentClient(args.agent_url)
    locator = GptLocator(
        args.model_id,
        system_prompt=(
            "You are a vision classifier for a yellow box labeled TARGET. "
            "Do NOT give driving commands. Respond ONLY with JSON "
            '{"position":"LEFT|CENTER|RIGHT|UNVISIBLE","distance":"FAR|MID|NEAR|NONE","confidence":0-1}. '
            "If the box is absent, use UNVISIBLE/NONE/0.0. No prose, no extra fields."
        ),
        user_prompt_template="{instruction}",
        max_new_tokens=args.max_new_tokens,
    )
    stabilizer = EstimateHysteresis(
        position_hold=args.position_hold,
        distance_hold=args.distance_hold,
    )
    governor = CommandGovernor(stop_confirmation_loops=args.stop_confirmation_loops)

    frame_dir = None
    if args.save_frames_dir:
        run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        frame_dir = Path(args.save_frames_dir) / run_tag
        frame_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("フレーム保存先: %s", frame_dir)

    try:
        while True:
            loop_start = time.time()
            frame = client.fetch_frame()
            if frame is None:
                time.sleep(args.loop_interval)
                continue
            if frame_dir:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
                fname = frame_dir / f"frame-{ts}.jpg"
                cv2.imwrite(str(fname), frame)
            infer_start = time.time()
            try:
                estimate, parsed_ok = locator.analyse(frame, args.instruction)
            except Exception as exc:
                LOG.error("VLM 呼び出し失敗: %s", exc)
                estimate, parsed_ok = SpatialEstimate("UNVISIBLE", "NONE", 0.0), False

            infer_elapsed = (time.time() - infer_start) * 1000
            smoothed = stabilizer.stabilize(estimate)
            raw_cmd = _decide_command(smoothed)

            low_conf = args.use_confidence_threshold and (smoothed.confidence < args.min_confidence)
            parse_fail = (not parsed_ok) or (raw_cmd not in ALLOWED_COMMANDS)

            if low_conf or parse_fail:
                filtered = governor.filter(args.fallback_command)
                reason = "low_conf" if low_conf else "parse_fail"
            else:
                filtered = governor.filter(raw_cmd)
                reason = "rule"

            LOG.info(
                "before_filter pos=%s dist=%s conf=%.2f raw=%s infer_ms=%.0f reason=%s parsed=%s",
                smoothed.position,
                smoothed.distance,
                smoothed.confidence,
                raw_cmd,
                infer_elapsed,
                reason,
                parsed_ok,
            )
            client.send_command(filtered)
            LOG.info("after_filter=%s loop_ms=%.0f", filtered, (time.time() - loop_start) * 1000)
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。STOP を送信します。")
        client.send_command("STOP")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
