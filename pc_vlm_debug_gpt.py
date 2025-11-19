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
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
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
    parser.add_argument(
        "--vlm-scale",
        type=float,
        default=1.0,
        help="VLM 入力前にフレームをこの倍率でリサイズ (例: 1.5 や 2.0)",
    )
    parser.add_argument(
        "--crop-center-ratio",
        type=float,
        default=1.0,
        help="VLM 入力前に中央をこの比率で正方形クロップ (1.0で無効)",
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


def _parse_json(text: str) -> Optional[Dict[str, object]]:
    import re

    snippets = re.findall(r"\{.*?\}", text, re.DOTALL)
    for snippet in snippets:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return None


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
        temperature: float,
        max_new_tokens: int,
        vlm_scale: float = 1.0,
        crop_center_ratio: float = 1.0,
    ) -> None:
        LOG.info("GPT VLM モデルを使用: %s", model_id)
        self.client = OpenAI()
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.mode = mode
        self.scale = vlm_scale
        self.crop = crop_center_ratio

        if mode == "direct":
            self.system_prompt = (
                "You are a vision model that controls a small robot car. "
                "Look at the current camera frame and output EXACTLY ONE JSON with the next action.\n"
                "Allowed actions: LEFT, RIGHT, FORWARD, STOP.\n"
                "Keys: action (string), confidence (0.0-1.0).\n"
                "Output format: {\"action\": \"LEFT\", \"confidence\": 0.74}"
            )
        elif mode == "describe":
            self.system_prompt = (
                "You are a vision model. List visible objects (short nouns) in the frame as a JSON array."
            )
        else:  # hybrid (position + distance)
            self.system_prompt = (
                "You are a vision model that controls a small robot car.\n"
                "Task:\n"
                "- Look at the current camera frame.\n"
                "- Find a yellow box with the word \"TARGET\" written on it.\n"
                "- Based on this single frame, classify:\n"
                "  - position of the TARGET box: LEFT, CENTER, RIGHT.\n"
                "  - distance to the TARGET box: FAR, MID, NEAR.\n"
                "If you do NOT clearly see the yellow TARGET box:\n"
                "- position = UNVISIBLE\n"
                "- distance = NONE\n"
                "- confidence = 0.0\n"
                "Output format (very important):\n"
                "- Return EXACTLY ONE JSON object. No code block, no extra text.\n"
                "- Keys: position, distance, confidence\n"
                "- position: one of [LEFT, CENTER, RIGHT, UNVISIBLE]\n"
                "- distance: one of [FAR, MID, NEAR, NONE]\n"
                "- confidence: number 0.0-1.0\n"
                "Valid examples:\n"
                "{\"position\": \"LEFT\", \"distance\": \"FAR\", \"confidence\": 0.82}\n"
                "{\"position\": \"CENTER\", \"distance\": \"NEAR\", \"confidence\": 0.91}\n"
                "{\"position\": \"UNVISIBLE\", \"distance\": \"NONE\", \"confidence\": 0.0}\n"
                "Rules:\n"
                "- Choose EXACTLY ONE value for position and EXACTLY ONE for distance.\n"
                "- NEVER use any words other than the allowed values above.\n"
                "- If unsure, output {\"position\":\"UNVISIBLE\",\"distance\":\"NONE\",\"confidence\":0.0}"
            )

    def _prepare_image(self, frame_bgr) -> str:
        frame = frame_bgr
        h, w = frame.shape[:2]
        if self.crop < 1.0:
            side = int(min(h, w) * self.crop)
            x0 = (w - side) // 2
            y0 = (h - side) // 2
            frame = frame[y0 : y0 + side, x0 : x0 + side]
        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(".jpg", frame)
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
        # content can be list[dict] or str depending on SDK
        if isinstance(content, list):
            raw_text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
        else:
            raw_text = content or ""

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
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = RaspiAgentClient(args.agent_url)
    runner = GptVLMRunner(
        args.model_id,
        mode=args.mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        vlm_scale=args.vlm_scale,
        crop_center_ratio=args.crop_center_ratio,
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
