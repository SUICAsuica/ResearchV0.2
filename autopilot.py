"""
SmolVLM を用いて MJPEG カメラ映像からターゲット矩形を検出し、
OSOYOO Raspberry Pi Robot Car (Lesson 6) を HTTP 制御する自律走行ループ。
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from raspycar.server import CarClient, default_path_command_map

LOG = logging.getLogger(__name__)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class BoxDetection:
    center: Tuple[int, int]
    width_ratio: float
    height_ratio: float
    confidence: float

    @property
    def area_ratio(self) -> float:
        return clamp01(self.width_ratio) * clamp01(self.height_ratio)


class SmolVLMDetector:
    """
    mlx-vlm の SmolVLM2 モデルを使ってターゲット矩形を推定する。
    """

    def __init__(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        try:
            from mlx_vlm.generate import generate as mlx_generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load as mlx_load
        except ImportError as exc:  # pragma: no cover - 実行環境依存
            raise RuntimeError(
                "mlx および mlx-vlm が未インストールです。'pip install mlx mlx-vlm' を実行してください。"
            ) from exc

        LOG.info("smolVLM モデル読込中: %s", model_id)
        self._generate = mlx_generate
        self._apply_chat_template = apply_chat_template
        self.model, self.processor = mlx_load(model_id, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        self.prompt = self._apply_chat_template(
            self.processor,
            self.model.config,
            messages,
            num_images=1,
            add_generation_prompt=True,
        )

    def detect(self, frame: np.ndarray) -> Optional[BoxDetection]:
        """
        BGR フレームを受け取り、ターゲット矩形を推定する。
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        try:
            result = self._generate(
                self.model,
                self.processor,
                prompt=self.prompt,
                image=[pil_image],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - 実行時保護
            LOG.exception("smolVLM 推論に失敗: %s", exc)
            return None

        text = getattr(result, "text", "") or ""
        if not text:
            LOG.debug("smolVLM から空応答.")
            return None

        payload = _extract_json(text)
        if not payload:
            LOG.debug("smolVLM 応答を JSON として解釈できず: %s", text.strip())
            return None
        if not payload.get("present"):
            LOG.debug("smolVLM がターゲット無しと判断.")
            return None

        try:
            confidence = clamp01(float(payload.get("confidence", 0.0)))
            width_ratio = clamp01(float(payload.get("box_width", 0.0)))
            height_ratio = clamp01(float(payload.get("box_height", 0.0)))
            center_x = clamp01(float(payload.get("center_x", 0.5)))
            center_y = clamp01(float(payload.get("center_y", 0.5)))
        except (TypeError, ValueError):
            LOG.debug("smolVLM 応答に非数値項目: %s", payload)
            return None

        if confidence <= 0.0 or width_ratio <= 0.0 or height_ratio <= 0.0:
            LOG.debug("信頼度または矩形サイズがゼロ: %s", payload)
            return None

        h, w = frame.shape[:2]
        cx_px = int(center_x * w)
        cy_px = int(center_y * h)

        return BoxDetection(
            center=(cx_px, cy_px),
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            confidence=confidence,
        )


def _extract_json(text: str) -> Optional[dict]:
    import json
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


class TargetFollower:
    """
    カメラ映像からターゲットを追跡し、CarClient 経由で移動コマンドを送る。
    """

    def __init__(
        self,
        client: CarClient,
        detector: SmolVLMDetector,
        camera_url: str,
        *,
        detection_interval: int,
        detection_timeout: float,
        min_confidence: float,
        turn_deadzone: float,
        stop_area_ratio: float,
        forward_speed: int,
        turn_speed: int,
        poll_sleep: float,
        preview: bool,
    ) -> None:
        self._client = client
        self._detector = detector
        self._camera_url = camera_url
        self._detection_interval = max(1, detection_interval)
        self._detection_timeout = max(0.1, detection_timeout)
        self._min_confidence = clamp01(min_confidence)
        self._turn_deadzone = max(0.0, turn_deadzone)
        self._stop_area_ratio = max(0.0, min(1.0, stop_area_ratio))
        self._forward_speed = int(np.clip(forward_speed, 0, 100))
        self._turn_speed = int(np.clip(turn_speed, 0, 100))
        self._poll_sleep = max(0.0, poll_sleep)
        self._preview = preview

    def run(self) -> None:
        cap = cv2.VideoCapture(self._camera_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"カメラストリームを開けませんでした: {self._camera_url}")

        if self._preview:
            cv2.namedWindow("raspycar-preview", cv2.WINDOW_NORMAL)

        last_detection: Optional[BoxDetection] = None
        last_detection_ts: float = 0.0
        frame_idx = 0
        last_command = ("", None)

        LOG.info("自律制御ループを開始します。Ctrl+C で停止。")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    LOG.warning("カメラフレームの取得に失敗。再試行します。")
                    time.sleep(0.2)
                    continue
                frame_idx += 1
                now = time.time()

                detection: Optional[BoxDetection] = None
                if frame_idx % self._detection_interval == 0:
                    candidate = self._detector.detect(frame)
                    if candidate:
                        if candidate.confidence >= self._min_confidence:
                            last_detection = candidate
                            last_detection_ts = now
                        else:
                            LOG.debug(
                                "smolVLM 応答の信頼度 %.2f が閾値 %.2f 未満のため破棄",
                                candidate.confidence,
                                self._min_confidence,
                            )
                            candidate = None
                    if candidate is None and now - last_detection_ts > self._detection_timeout:
                        last_detection = None

                if last_detection and now - last_detection_ts <= self._detection_timeout:
                    detection = last_detection

                command, speed = self._decide_command(frame, detection)
                if (command, speed) != last_command:
                    self._send_command(command, speed)
                    last_command = (command, speed)

                if self._preview:
                    self._update_preview(frame, detection)

                if self._poll_sleep > 0.0:
                    time.sleep(self._poll_sleep)
        except KeyboardInterrupt:
            LOG.info("停止シグナルを受信。車体を停止します。")
            self._client.stop()
        finally:
            cap.release()
            if self._preview:
                cv2.destroyAllWindows()

    def _decide_command(
        self,
        frame: np.ndarray,
        detection: Optional[BoxDetection],
    ) -> Tuple[str, Optional[int]]:
        if detection is None:
            return "stop", None

        if detection.area_ratio >= self._stop_area_ratio:
            LOG.debug("ターゲットが十分近いと判断: area=%.3f", detection.area_ratio)
            return "stop", None

        h, w = frame.shape[:2]
        cx_norm = detection.center[0] / max(w, 1)
        offset = cx_norm - 0.5

        if abs(offset) <= self._turn_deadzone:
            return "forward", self._forward_speed
        if offset < 0:
            return "left", self._turn_speed
        return "right", self._turn_speed

    def _send_command(self, command: str, speed: Optional[int]) -> None:
        LOG.info("送信コマンド: %s speed=%s", command, speed)
        if command == "stop":
            self._client.stop()
            return
        self._client.move(command, speed=speed)

    @staticmethod
    def _update_preview(frame: np.ndarray, detection: Optional[BoxDetection]) -> None:
        if detection:
            cx, cy = detection.center
            h, w = frame.shape[:2]
            box_w = int(clamp01(detection.width_ratio) * w / 2)
            box_h = int(clamp01(detection.height_ratio) * h / 2)
            top_left = (max(0, cx - box_w), max(0, cy - box_h))
            bottom_right = (min(w - 1, cx + box_w), min(h - 1, cy + box_h))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"conf={detection.confidence:.2f}",
                (top_left[0], max(20, top_left[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("raspycar-preview", frame)
        if cv2.waitKey(1) == 27:  # ESC で終了
            raise KeyboardInterrupt


def _parse_mapping_items(items: Optional[list[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not items:
        return mapping
    for item in items:
        if "=" not in item:
            raise ValueError(f"direction=name 形式で指定してください: {item}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="smolVLM を用いた Raspberry Pi Car 自律制御",
    )
    parser.add_argument("--camera-url", required=True, help="MJPEG ストリーム URL 例: http://192.168.11.4:8899/stream.mjpg")
    parser.add_argument("--base-url", required=True, help="Flask 制御エンドポイント 例: http://192.168.11.4:5000")
    parser.add_argument("--smol-model-id", default=None, help="mlx-vlm に渡すモデル ID またはパス（省略時は環境変数 SMOL_MODEL_ID を参照）")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a vision detector for a mobile robot. Respond ONLY with a JSON dict containing keys "
            "present, confidence, center_x, center_y, box_width, box_height. Detect a bright white card "
            "with a bold black rectangular border; if uncertain, output present=false."
        ),
    )
    parser.add_argument(
        "--user-prompt",
        default=(
            'Look at the image from the robot. If you clearly see the white card with a thick black border, '
            'output only JSON like {"present": true, "confidence": 0.85, "center_x": 0.51, "center_y": 0.48, '
            '"box_width": 0.22, "box_height": 0.18}. '
            "If it is missing, occluded, or ambiguous, output present=false and zeros for all numeric values."
        ),
    )
    parser.add_argument("--detection-interval", type=int, default=10, help="何フレームごとに smolVLM を呼ぶか")
    parser.add_argument("--detection-timeout", type=float, default=3.0, help="推論結果を保持する最大秒数")
    parser.add_argument("--turn-deadzone", type=float, default=0.05, help="正規化中心からのずれがこの範囲なら前進を継続")
    parser.add_argument("--stop-area-ratio", type=float, default=0.12, help="ターゲット面積がこの割合を超えたら停止")
    parser.add_argument("--min-confidence", type=float, default=0.85, help="smolVLM 応答を採用する最小信頼度（0-1）")
    parser.add_argument("--forward-speed", type=int, default=60, help="前進時の速度（0-100）")
    parser.add_argument("--turn-speed", type=int, default=40, help="旋回時の速度（0-100）")
    parser.add_argument("--poll-sleep", type=float, default=0.05, help="各ループ後のスリープ秒数")
    parser.add_argument("--no-preview", action="store_true", help="OpenCV プレビューウィンドウを無効化")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--car-api-style",
        choices=("query", "path"),
        default="query",
        help="Lesson 6 互換のクエリ式 or /move/<cmd> のパス式 API を選択",
    )
    parser.add_argument("--car-move-path", default="/move", help="パス式 API のベースパス（既定: /move）")
    parser.add_argument(
        "--car-path-map",
        nargs="*",
        metavar="direction=name",
        help="パス式 API で direction→コマンド名を上書き (例: stop=stopcar left=turnleft)",
    )
    args = parser.parse_args(argv)
    try:
        args.car_path_map = _parse_mapping_items(args.car_path_map)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model_id = args.smol_model_id or _resolve_default_model_id()
    detector = SmolVLMDetector(
        model_id=model_id,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    path_map = default_path_command_map()
    if args.car_path_map:
        path_map.update(args.car_path_map)
    client = CarClient(
        base_url=args.base_url,
        api_style=args.car_api_style,
        move_path=args.car_move_path,
        path_command_map=path_map,
    )
    follower = TargetFollower(
        client,
        detector,
        args.camera_url,
        detection_interval=args.detection_interval,
        detection_timeout=args.detection_timeout,
        turn_deadzone=args.turn_deadzone,
        min_confidence=args.min_confidence,
        stop_area_ratio=args.stop_area_ratio,
        forward_speed=args.forward_speed,
        turn_speed=args.turn_speed,
        poll_sleep=args.poll_sleep,
        preview=not args.no_preview,
    )

    def handle_sigint(signum, frame):  # pragma: no cover - シグナル処理
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)
    follower.run()
    return 0


def _resolve_default_model_id() -> str:
    import os

    env_value = os.getenv("SMOL_MODEL_ID")
    if env_value:
        return env_value
    local_path = os.path.join(os.path.dirname(__file__), "models", "smolvlm2-mlx")
    expanded = os.path.abspath(local_path)
    if os.path.isdir(expanded):
        LOG.info("ローカルの smolVLM 重みを検出: %s", expanded)
        return expanded
    LOG.info("SMOL_MODEL_ID が未設定のため Hugging Face からダウンロードします。")
    return "mlx-community/SmolVLM2-500M-Video-Instruct-mlx"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
