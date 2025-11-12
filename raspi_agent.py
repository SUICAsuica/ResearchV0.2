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

from raspycar.hardware import MotorController

LOG = logging.getLogger(__name__)


COMMAND_VECTORS: Dict[str, Tuple[float, float]] = {
    "STOP": (0.0, 0.0),
    # 現状の配線では教材と正転方向が逆なので、符号を反転させて整合させる。
    "FORWARD": (-0.8, -0.8),
    "FORWARD_SLOW": (-0.4, -0.4),
    "BACKWARD": (0.6, 0.6),
    "LEFT": (-0.4, 0.4),
    "RIGHT": (0.4, -0.4),
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
    ) -> None:
        self._motor = motor
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
        vector = COMMAND_VECTORS.get(command)
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
    parser.add_argument("--camera-source", default=0, help="OpenCV VideoCapture source index/path")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=2.0)
    parser.add_argument("--watchdog-timeout", type=float, default=2.0)
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
    camera.start()
    agent = RobotAgent(motor, watchdog_timeout=args.watchdog_timeout)
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
