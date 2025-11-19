"""GPT VLM デバッグランナー（raspi_agent からフレーム取得、OpenAI VLM 推論のみ実施）。

smolVLM での純 VLM 評価が失敗したため、同じインタフェースで gpt-5-mini-2025-08-07
（ChatGPT VLM）を利用して比較する目的のスクリプト。モーター制御は行わない。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from raspycar.agent_client import RaspiAgentClient

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT VLM debug runner (fetch frame, call gpt-5.1-mini, print result).",
    )
    parser.add_argument("--agent-url", required=True, help="raspi_agent のベースURL")
    parser.add_argument("--instruction", required=True, help="VLM へ渡す指示文")
    parser.add_argument("--model-id", default="gpt-5-mini-2025-08-07")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "direct", "describe"],
        default="hybrid",
        help="hybrid=位置/距離JSON, direct=コマンドJSON, describe=写っている物体を列挙",
    )
    parser.add_argument(
        "--loop-interval",
        type=float,
        default=10.0,
        help="ループ周期（秒）。gpt-5-mini は 1 推論 ≈10 秒かかる前提で大きめに設定してください",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="OpenAI VLM の温度 (省略=モデルのデフォルト固定値)",
    )
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
    parser.add_argument(
        "--save-frame-dir",
        type=Path,
        help="指定すると各推論前のフレームをこのディレクトリに frame_XXXX.jpg として保存",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="推論を始める前に指定枚数のフレームを捨ててウォームアップする",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="OpenCV ウィンドウで推論結果を連続表示",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="ウィンドウ表示時の縮尺（例: 0.75）",
    )
    parser.add_argument(
        "--color-guard",
        action="store_true",
        help="黄色抽出ヒューリスティックで VLM の JSON を検証・補正",
    )
    parser.add_argument(
        "--min-yellow-area",
        type=float,
        default=0.01,
        help="色ヒューリスティックが有効とみなす最小面積比 (0-1)",
    )
    parser.add_argument(
        "--position-thresholds",
        type=float,
        nargs=2,
        metavar=("LEFT_MAX", "RIGHT_MIN"),
        default=(0.33, 0.66),
        help="中心比を LEFT/CENTER/RIGHT に分割する閾値",
    )
    parser.add_argument(
        "--distance-thresholds",
        type=float,
        nargs=2,
        metavar=("FAR_MAX", "MID_MAX"),
        default=(0.15, 0.35),
        help="高さ比を FAR/MID/NEAR に分割する閾値",
    )
    return parser.parse_args()


@dataclass
class ColorGuardResult:
    position: str
    distance: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    area_ratio: float
    center_ratio: float
    height_ratio: float


def _ensure_env() -> None:
    """.env が存在すれば自動で読み込む（既存環境変数は優先）。"""

    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=False)


def _parse_json(text: str) -> Optional[Dict[str, object]]:
    import re

    snippets = re.findall(r"\{.*?\}", text, re.DOTALL)
    for snippet in snippets:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return None


def _extract_response_text(choice) -> str:
    """Return assistant text content from ChatCompletion choice."""

    message = getattr(choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, Sequence):
        fragments = []
        for part in content:
            if isinstance(part, str):
                fragments.append(part)
                continue
            text_attr = getattr(part, "text", None)
            if isinstance(text_attr, str):
                fragments.append(text_attr)
                continue
            if isinstance(part, dict):
                text_val = part.get("text")
                if isinstance(text_val, str):
                    fragments.append(text_val)
        return "\n".join(fragments)

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return ""


def _coerce_hybrid(text: str, payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Hybrid 用: 生テキストに含まれる最初の有効ラベルを強制抽出・整形。"""

    import re

    upper = text.upper()
    pos_match = re.search(r"\b(LEFT|CENTER|RIGHT|UNVISIBLE)\b", upper)
    dist_match = re.search(r"\b(FAR|MID|NEAR|NONE)\b", upper)
    conf_match = re.search(r"CONFIDENCE[^0-9]*([0-9]+(?:\.[0-9]+)?)", upper)

    pos = pos_match.group(1) if pos_match else None
    dist = dist_match.group(1) if dist_match else None
    if conf_match:
        try:
            conf = float(conf_match.group(1))
        except ValueError:
            conf = 0.0
    else:
        conf = None

    if pos and not dist:
        dist = "FAR" if pos != "UNVISIBLE" else "NONE"
    elif dist and not pos:
        pos = "UNVISIBLE"
    if not pos or not dist:
        pos, dist, conf = "UNVISIBLE", "NONE", 0.0
    if conf is None:
        conf = float(payload.get("confidence", 0.0)) if payload else 0.0

    return {"position": pos, "distance": dist, "confidence": conf}


def _estimate_by_color(
    frame_bgr,
    *,
    min_area_ratio: float,
    position_thresholds: Tuple[float, float],
    distance_thresholds: Tuple[float, float],
) -> Optional[ColorGuardResult]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 90, 80], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    h, w = frame_bgr.shape[:2]
    area_ratio = area / float(w * h)
    if area_ratio < min_area_ratio:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    center_ratio = (x + bw / 2) / w
    height_ratio = bh / h
    left_max, right_min = position_thresholds
    if center_ratio < left_max:
        position = "LEFT"
    elif center_ratio > right_min:
        position = "RIGHT"
    else:
        position = "CENTER"
    far_max, mid_max = distance_thresholds
    if height_ratio < far_max:
        distance = "FAR"
    elif height_ratio < mid_max:
        distance = "MID"
    else:
        distance = "NEAR"
    confidence = float(min(0.99, max(0.3, area_ratio * 4)))
    return ColorGuardResult(
        position=position,
        distance=distance,
        confidence=confidence,
        bbox=(x, y, bw, bh),
        area_ratio=area_ratio,
        center_ratio=center_ratio,
        height_ratio=height_ratio,
    )


def _draw_overlay(frame_bgr, vlm_json: Optional[Dict[str, object]], guard: Optional[ColorGuardResult]):
    canvas = frame_bgr.copy()
    if guard:
        x, y, w, h = guard.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"guard {guard.position}/{guard.distance} h={guard.height_ratio:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    if vlm_json:
        cv2.putText(
            canvas,
            f"VLM {vlm_json.get('position')} / {vlm_json.get('distance')} ({vlm_json.get('confidence', 0.0):.2f})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 0),
            2,
        )
    return canvas


class GptVLMRunner:
    """Wrapper around OpenAI Chat Completions (GPT-5.1-mini)."""

    def __init__(
        self,
        model_id: str,
        *,
        mode: str,
        temperature: Optional[float],
        max_new_tokens: int,
    ) -> None:
        LOG.info("GPT VLM モデルを使用: %s", model_id)
        self.client = OpenAI()
        self.model_id = model_id
        # gpt-5.* 系は API 側で温度が固定（1.0）のため、明示指定は無視する
        if model_id.startswith("gpt-5") and temperature is not None:
            if abs(temperature - 1.0) > 1e-6:
                LOG.warning(
                    "モデル %s は temperature を変更できません (指定 %.2f)。デフォルト値を使用します。",
                    model_id,
                    temperature,
                )
            self.temperature = None
        else:
            self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.mode = mode

        if mode == "direct":
            self.system_prompt = (
                'Respond only with JSON {"action":"LEFT|RIGHT|FORWARD|STOP"} and nothing else.'
            )
        elif mode == "describe":
            self.system_prompt = "Only reply with a JSON array of short nouns you see in the frame."
        else:  # hybrid (position + distance)
            self.system_prompt = (
                "Only reply with JSON {\"position\":\"LEFT|CENTER|RIGHT|UNVISIBLE\","
                "\"distance\":\"FAR|MID|NEAR|NONE\",\"confidence\":0-1} describing the yellow TARGET box. "
                "Use UNVISIBLE/NONE/0.0 if the box is missing."
            )

    def _prepare_image(self, frame_bgr) -> str:
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame to JPEG")
        b64 = base64.b64encode(buf).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def run(self, frame_bgr, instruction: str) -> Dict[str, object]:
        image_url = self._prepare_image(frame_bgr)

        user_content = [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        request_kwargs = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=self.max_new_tokens,
        )
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        resp = self.client.chat.completions.create(**request_kwargs)

        choice = resp.choices[0]
        raw_text = _extract_response_text(choice)

        payload = _parse_json(raw_text)
        if self.mode == "describe":
            parsed = payload if isinstance(payload, list) else None
            return {"raw": raw_text, "json": parsed}
        elif self.mode == "direct":
            parsed = payload if isinstance(payload, dict) else None
            return {"raw": raw_text, "json": parsed}
        else:
            coerced = _coerce_hybrid(raw_text, payload)
            return {"raw": raw_text, "json": coerced}


def main() -> int:
    _ensure_env()
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.loop_interval < 5.0:
        LOG.warning("loop-interval=%.2f 秒は短すぎます。推論 1 回あたり ≈10 秒を想定してください", args.loop_interval)

    client = RaspiAgentClient(args.agent_url)
    runner = GptVLMRunner(
        args.model_id,
        mode=args.mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    save_dir: Optional[Path] = None
    session_tag = time.strftime("%Y%m%d-%H%M%S")
    if args.save_frame_dir:
        save_dir = args.save_frame_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0
    if args.warmup_frames > 0:
        LOG.info("ウォームアップとして先頭 %d フレームを破棄します", args.warmup_frames)
        for i in range(args.warmup_frames):
            frame = client.fetch_frame()
            if frame is None:
                LOG.warning("ウォームアップでフレーム取得に失敗 (%d/%d)", i + 1, args.warmup_frames)
                time.sleep(args.loop_interval)
            else:
                LOG.debug("warmup frame %d 取得", i + 1)
                time.sleep(args.loop_interval)

    try:
        while True:
            loop_start = time.perf_counter()
            frame = client.fetch_frame()
            if frame is None:
                LOG.warning("フレーム取得に失敗。再試行します。")
                time.sleep(args.loop_interval)
                continue

            guard_result: Optional[ColorGuardResult] = None
            if args.color_guard or args.display:
                guard_result = _estimate_by_color(
                    frame,
                    min_area_ratio=args.min_yellow_area,
                    position_thresholds=tuple(args.position_thresholds),
                    distance_thresholds=tuple(args.distance_thresholds),
                )
                if guard_result:
                    LOG.debug(
                        "color-guard pos=%s dist=%s height=%.3f area=%.3f center=%.3f",
                        guard_result.position,
                        guard_result.distance,
                        guard_result.height_ratio,
                        guard_result.area_ratio,
                        guard_result.center_ratio,
                    )
                else:
                    LOG.debug("color-guard: TARGET 未検出")

            if save_dir:
                iteration_preview = iteration + 1
                save_path = save_dir / f"{session_tag}_frame_{iteration_preview:04d}.jpg"
                cv2.imwrite(str(save_path), frame)
                LOG.debug("フレームを保存しました: %s", save_path)

            result = runner.run(frame, args.instruction)
            if args.color_guard:
                if guard_result:
                    override = False
                    if not result["json"]:
                        override = True
                    else:
                        pos = str(result["json"].get("position", "")).upper()
                        dist = str(result["json"].get("distance", "")).upper()
                        valid_pos = pos in {"LEFT", "CENTER", "RIGHT"}
                        valid_dist = dist in {"FAR", "MID", "NEAR"}
                        if not (valid_pos and valid_dist):
                            override = True
                        elif (pos, dist) != (guard_result.position, guard_result.distance):
                            override = True
                            LOG.warning(
                                "VLM estimate %s/%s とヒューリスティック %s/%s が矛盾",
                                pos,
                                dist,
                                guard_result.position,
                                guard_result.distance,
                            )
                    if override:
                        result["json"] = {
                            "position": guard_result.position,
                            "distance": guard_result.distance,
                            "confidence": guard_result.confidence,
                        }
                        LOG.info("color-guard により結果を補正しました (conf=%.2f)", guard_result.confidence)
                else:
                    result["json"] = {
                        "position": "UNVISIBLE",
                        "distance": "NONE",
                        "confidence": 0.0,
                    }
                    LOG.info("color-guard 未検出のため UNVISIBLE/NONE にフォールバックしました")

            iteration += 1
            LOG.info("---- 推論 #%d ----", iteration)
            print(json.dumps(result["json"], ensure_ascii=False, indent=2))
            LOG.info("raw=%s", result["raw"])

            if args.display:
                annotated = _draw_overlay(frame, result["json"], guard_result)
                if args.display_scale != 1.0:
                    annotated = cv2.resize(
                        annotated,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow("GPT VLM Debug", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    LOG.info("'q' 入力で終了します")
                    break

            if args.once:
                break
            if args.loop_interval > 0:
                elapsed = time.perf_counter() - loop_start
                remain = args.loop_interval - elapsed
                if remain > 0:
                    time.sleep(remain)
                else:
                    LOG.debug("loop interval exceeded by %.3fs", -remain)
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。終了します。")
    finally:
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
