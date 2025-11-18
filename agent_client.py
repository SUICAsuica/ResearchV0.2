"""Client helper for communicating with ``raspi_agent.py``."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests

LOG = logging.getLogger(__name__)


class RaspiAgentClient:
    """Minimal HTTP client for the Raspberry Pi agent service."""

    def __init__(self, base_url: str, *, timeout: float = 2.5) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------ #
    def send_command(self, command: str) -> Dict[str, Any]:
        payload = {"command": command}
        response = self.session.post(
            f"{self.base_url}/command",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def ping(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def fetch_frame(self) -> Optional[np.ndarray]:
        response = self.session.get(
            f"{self.base_url}/frame.jpg",
            params={"ts": time.time()},  # キャッシュ防止のためクエリを付与
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            LOG.warning("エージェントから取得した JPEG フレームのデコードに失敗しました")
        return frame
