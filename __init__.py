"""
OSOYOO Raspberry Pi Robot Car (2023) と Mac 上の VLM 推論をつなぐ補助モジュール。

- `server` : Flask 制御エンドポイントへ HTTP リクエストを送るクライアント。
- `hardware` : （任意）ラズパイ上で直接 PCA9685 を操作するためのラッパ。
"""

__all__ = ["hardware", "server"]
