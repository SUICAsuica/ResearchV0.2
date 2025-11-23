"""Raspberry Pi side agent that exposes camera frames and motor commands."""

from __future__ import annotations

import argparse
import io
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple

import cv2
import requests

from raspycar.hardware import MotorController

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# コマンドベクトル生成（ステアリングオフセット適用）
# --------------------------------------------------------------------------- #
DEFAULT_STEER_OFFSET = -0.025  # 負値で「右寄せ」、正値で「左寄せ」補正


def _apply_offset(vector: Tuple[float, float], steer_offset: float) -> Tuple[float, float]:
    """左右のモータ出力にオフセットを適用するヘルパ。"""
    left, right = vector
    return (left + steer_offset, right - steer_offset)


def build_command_vectors(
    steer_offset: float,
    *,
    slow_ratio: float = 0.4,
) -> Dict[str, Tuple[float, float]]:
    """
    ステアリングオフセット込みのコマンド辞書を生成する。

    :param steer_offset: 前進系コマンドに足す左右差。負なら右寄せ。
    :param slow_ratio: FORWARD_SLOW に対してオフセットを何倍に縮めるか。
    """
    forward_base = -0.675  # 片側の基準速度（前進）
    slow_base = -0.29      # 前進スローの基準速度
    slow_offset = steer_offset * slow_ratio

    return {
        "STOP": (0.0, 0.0),
        # 現状の配線では教材と正転方向が逆なので、符号を反転させて整合させる。
        "FORWARD": _apply_offset((forward_base, forward_base), steer_offset),
        # 細かい前進調整用に速度をさらに落としたもの。
        "FORWARD_SLOW": _apply_offset((slow_base, slow_base), slow_offset),
        # 後退は左右差を付けず素直に動かす（必要なら別途調整）。
        "BACKWARD": (0.6, 0.6),
        # 左右コマンドはその場回頭ではなく「少し左/少し右」の補正にする。
        "LEFT": (-0.30, -0.60),
        "RIGHT": (-0.60, -0.30),
    }


class CameraWorker:
    """Continuously captures frames and keeps the latest JPEG in memory."""

    def __init__(
        self,
        source: int | str,
        *,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._interval = 1.0 / max(fps, 0.1)
        self._lock = threading.Lock()
        self._latest_jpeg: bytes = b""
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg if self._latest_jpeg else None

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        # HTTP MJPEG ソース（例: http://<ip>:8899/stream.mjpg）を直接読む場合の簡易パーサ
        if isinstance(self._source, str) and self._source.startswith(("http://", "https://")):
            while not self._stop_evt.is_set():
                last_ts = time.monotonic()
                try:
                    resp = requests.get(self._source, stream=True, timeout=5)
                except Exception as exc:  # pragma: no cover - ネットワーク系例外
                    LOG.error("HTTP カメラへの接続に失敗しました: %s", exc)
                    time.sleep(1.0)
                    continue
                if resp.status_code != 200:
                    LOG.error("HTTP カメラが %s を返しました", resp.status_code)
                    resp.close()
                    time.sleep(1.0)
                    continue
                LOG.info("HTTP カメラストリームを %s から受信開始", self._source)
                buffer = b""
                try:
                    for chunk in resp.iter_content(chunk_size=4096):
                        if self._stop_evt.is_set():
                            break
                        if not chunk:
                            # しばらくフレーム更新が無い場合は再接続
                            if time.monotonic() - last_ts > 3.0:
                                LOG.warning("HTTP カメラのフレーム更新が途切れたため再接続します")
                                break
                            continue

                        buffer += chunk
                        # バッファが膨らみすぎたら古いデータを捨てて最新付近だけ残す。
                        # MJPEG 受信が遅延して TCP バッファに溜まった場合でも、ここで強制的に追いつきやすくする。
                        if len(buffer) > 2_000_000:  # 約2MBを閾値にする
                            buffer = buffer[-200_000:]  # 末尾20万バイトだけ残す（複数フレーム分相当）
                            LOG.warning("HTTP カメラのバッファが膨らんだため古いフレームを破棄しました")

                        while True:
                            start = buffer.find(b"\xff\xd8")  # JPEG SOI
                            end = buffer.find(b"\xff\xd9", start + 2) if start != -1 else -1  # EOI
                            if start != -1 and end != -1:
                                jpeg = buffer[start : end + 2]
                                buffer = buffer[end + 2 :]
                                with self._lock:
                                    self._latest_jpeg = jpeg
                                last_ts = time.monotonic()
                                break
                            else:
                                # keep accumulating
                                break
                    # ループが break で抜けた場合は resp を閉じて再接続
                finally:
                    resp.close()
                time.sleep(0.2)  # 短い間隔で再接続
            return

        # それ以外は従来どおり OpenCV でデバイスを開く
        cap = cv2.VideoCapture(self._source)
        if self._width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        while not self._stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                LOG.warning("カメラフレームの取得に失敗しました。0.5 秒後に再試行します")
                time.sleep(0.5)
                continue
            success, encoded = cv2.imencode(".jpg", frame)
            if success:
                with self._lock:
                    self._latest_jpeg = encoded.tobytes()
            time.sleep(self._interval)
        cap.release()


class RobotAgent:
    def __init__(
        self,
        motor: MotorController,
        *,
        watchdog_timeout: float,
        command_vectors: Dict[str, Tuple[float, float]],
    ) -> None:
        self._motor = motor
        self._command_vectors = command_vectors
        self._watchdog_timeout = max(0.5, watchdog_timeout)
        self._last_command = "STOP"
        self._last_timestamp = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._watchdog_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=1.0)
        self._motor.stop()

    def apply_command(self, command: str) -> Dict[str, str]:
        command = command.strip().upper()
        if command == "PING":
            return {"status": "PONG"}
        vector = self._command_vectors.get(command)
        if vector is None:
            raise ValueError(f"Unknown command: {command}")
        self._motor.set_speed(*vector)
        with self._lock:
            self._last_command = command
            self._last_timestamp = time.time()
        return {"status": "ok", "command": command}

    def status(self) -> Dict[str, str | float | bool]:
        with self._lock:
            return {
                "last_command": self._last_command,
                "last_ts": self._last_timestamp,
                "dry_run": self._motor.dry_run,
            }

    # ------------------------------------------------------------------ #
    def _watchdog_loop(self) -> None:
        while self._running:
            now = time.time()
            with self._lock:
                elapsed = now - self._last_timestamp
            if elapsed > self._watchdog_timeout:
                self._motor.stop()
            time.sleep(0.2)


def build_http_handler(agent: RobotAgent, camera: CameraWorker):
    class AgentHandler(BaseHTTPRequestHandler):
        server_version = "RaspiAgent/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:  # pragma: no cover
            LOG.info("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health"):
                self._write_json(agent.status())
                return
            if self.path.startswith("/frame.jpg"):
                jpeg = camera.latest_jpeg()
                if not jpeg:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no frame yet")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path.startswith("/stream.mjpg"):
                self._stream_mjpeg()
                return
            if self.path.startswith("/command"):
                command = self._extract_query_command()
                if not command:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing command param")
                    return
                try:
                    payload = agent.apply_command(command)
                except ValueError as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._write_json(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/command"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            command = None
            if body:
                try:
                    data = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    command = body.decode("utf-8", errors="ignore").strip()
                else:
                    command = str(data.get("command", ""))
            if not command:
                self.send_error(HTTPStatus.BAD_REQUEST, "Command not provided")
                return
            try:
                payload = agent.apply_command(command)
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._write_json(payload)

        # -------------------------------------------------------------- #
        def _write_json(self, data: Dict[str, object]) -> None:
            encoded = json.dumps(data).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _extract_query_command(self) -> Optional[str]:
            if "?" not in self.path:
                return None
            query = self.path.split("?", 1)[1]
            for pair in query.split("&"):
                if pair.startswith("command="):
                    return pair.split("=", 1)[1]
            return None

        def _stream_mjpeg(self) -> None:
            boundary = "frame"
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.end_headers()
            try:
                while True:
                    jpeg = camera.latest_jpeg()
                    if not jpeg:
                        time.sleep(0.1)
                        continue
                    self.wfile.write(b"--" + boundary.encode("ascii") + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg + b"\r\n")
                    time.sleep(0.1)
            except ConnectionError:
                return

    return AgentHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi robot agent")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--camera-source",
        default=0,
        help="OpenCV VideoCapture source index/path or HTTP MJPEG URL",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=2.0)
    parser.add_argument("--watchdog-timeout", type=float, default=2.0)
    parser.add_argument(
        "--steer-offset",
        type=float,
        default=DEFAULT_STEER_OFFSET,
        help="前進時の左右差を補正する値（負で右寄せ、正で左寄せ）",
    )
    parser.add_argument(
        "--servo-center",
        type=float,
        default=0.0,
        help="起動直後にカメラサーボへ書き込む角度（度）。サーボが左に寝る場合はここで調整。",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true", help="強制的にモータ制御を無効化")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    motor = MotorController(dry_run=args.dry_run)
    camera = CameraWorker(
        args.camera_source,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    # 起動時にサーボを既定角へ合わせておく（パルス初期化で左に振れる個体対策）。
    try:
        motor.set_servo_angle(args.servo_center)
        LOG.info("サーボ初期角を %.1f° に設定しました", args.servo_center)
    except Exception:
        LOG.warning("サーボ初期角の設定に失敗しました（dry-run もしくはサーボ未接続の可能性）")

    camera.start()
    command_vectors = build_command_vectors(args.steer_offset)
    LOG.info(
        "ステアリング補正値 steer_offset=%.3f を使用してコマンドベクトルを生成しました",
        args.steer_offset,
    )
    agent = RobotAgent(motor, watchdog_timeout=args.watchdog_timeout, command_vectors=command_vectors)
    agent.start()

    handler_cls = build_http_handler(agent, camera)
    server = ThreadingHTTPServer((args.bind, args.port), handler_cls)
    LOG.info("raspi_agent が %s:%d で待ち受け開始", args.bind, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("停止シグナルを受信。シャットダウンします")
    finally:
        server.shutdown()
        agent.stop()
        camera.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
