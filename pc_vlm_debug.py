"""Utility script to run SmolVLM inference only (no motor commands)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
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
    parser.add_argument(
        "--save-frame-dir",
        type=Path,
        help="指定すると各推論前のフレームをこのディレクトリに frame_XXXX.jpg として保存",
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

    save_dir: Optional[Path] = None
    session_tag = time.strftime("%Y%m%d-%H%M%S")
    if args.save_frame_dir:
        save_dir = args.save_frame_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0
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
            if args.color_guard and guard_result:
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
                cv2.imshow("SmolVLM Debug", annotated)
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
