"""
Flask ベースの OSOYOO Raspberry Pi Robot Car に対して HTTP で制御コマンドを送るクライアント。

2023 年版 Lesson 6 の `webcar.py` が提供するエンドポイント
（例: `http://<pi-ip>:5000/?action=command&move=forward`）を想定し、
ブラウザのボタン操作を Mac 上のスクリプトから呼び出す用途に利用する。
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urljoin

import requests

LOG = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 2.0


def default_path_command_map() -> Dict[str, str]:
    """
    Lesson 6 (OSOYOO) の Flask 実装が提供するデフォルトのパス式エンドポイント。
    """
    return {
        "forward": "forward",
        "backward": "backward",
        "left": "turnleft",
        "right": "turnright",
        "stop": "stopcar",
    }


def _normalise_base_url(base_url: str) -> str:
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base_url must include scheme (e.g. http://192.168.0.13:5000)")
    return base_url.rstrip("/")


@dataclass
class CarClient:
    """
    Lesson 6 の Flask (`webcar.py`) へクエリを送るシンプルなクライアント。

    デフォルトでは `/?action=command&move=forward` のようなクエリを発行するが、
    パラメータ名やエンドポイントパスはオプションで上書きできる。
    """

    base_url: str
    endpoint: str = "/"
    action_param: str = "action"
    action_value: str = "command"
    move_param: str = "move"
    speed_param: str = "speed"
    servo_param: str = "servo"
    timeout: float = DEFAULT_TIMEOUT
    api_style: str = "query"  # "query" or "path"
    move_path: str = "/move"
    path_command_map: Dict[str, str] = field(default_factory=default_path_command_map)
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.base_url = _normalise_base_url(self.base_url)
        self.endpoint = "/" + self.endpoint.lstrip("/")
        if self.api_style not in {"query", "path"}:
            raise ValueError("api_style must be either 'query' or 'path'")
        if not self.move_path.startswith("/"):
            self.move_path = "/" + self.move_path

    # -------------------------------------------------------------- #
    def move(self, direction: str, speed: Optional[int] = None) -> requests.Response:
        """
        direction に指定した動作を実行する。
        例: forward / backward / left / right / stop
        """
        if self.api_style == "path":
            command_name = self.path_command_map.get(direction, direction)
            path = f"{self.move_path.rstrip('/')}/{command_name.lstrip('/')}"
            if speed is not None:
                LOG.debug("Path API では speed パラメータを無視します: %s", speed)
            return self._request({}, path=path)

        params: Dict[str, Any] = {self.move_param: direction}
        if speed is not None:
            params[self.speed_param] = str(_clamp_int(speed, 0, 100))
        return self._request(params)

    def stop(self) -> requests.Response:
        """車体停止用のショートカット。"""
        return self.move("stop")

    def set_speed(self, speed: int) -> requests.Response:
        """
        教材 UI のスライダーに相当する速度設定。
        サーバ側が move パラメータと組み合わせる場合は move() の speed 引数を使う。
        """
        params = {self.speed_param: str(_clamp_int(speed, 0, 100))}
        return self._request(params)

    def set_servo_angle(self, angle_deg: float) -> requests.Response:
        """
        カメラパン用サーボ角度（Degrees）。
        Lesson 6 ではおおむね -90〜90° を想定しているため、その範囲でクリップする。
        """
        clamped = max(-90.0, min(90.0, angle_deg))
        params = {self.servo_param: f"{clamped:.1f}"}
        return self._request(params)

    def raw_command(self, params: Dict[str, Any]) -> requests.Response:
        """任意のクエリパラメータをそのまま送信する。"""
        return self._request(params)

    # -------------------------------------------------------------- #
    def _request(self, params: Dict[str, Any], path: Optional[str] = None) -> requests.Response:
        if path:
            url = urljoin(self.base_url + "/", path.lstrip("/"))
            payload = params or None
        else:
            url = urljoin(self.base_url + "/", self.endpoint.lstrip("/"))
            payload = {self.action_param: self.action_value}
            payload.update(params)

        LOG.debug("HTTP GET %s params=%s", url, payload)
        response = self.session.get(url, params=payload, timeout=self.timeout)
        response.raise_for_status()
        return response


# ------------------------------------------------------------------ #
# CLI ヘルパ
# ------------------------------------------------------------------ #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OSOYOO Raspberry Pi Robot Car (2023) の Flask 制御エンドポイントを呼び出します。",
    )
    parser.add_argument("--base-url", required=True, help="例: http://192.168.0.13:5000")
    parser.add_argument("--endpoint", default="/", help="相対パス（既定: /）")
    parser.add_argument("--action-param", default="action", help="アクション名のクエリキー（既定: action）")
    parser.add_argument("--action-value", default="command", help="アクション値（既定: command）")
    parser.add_argument("--move-param", default="move", help="移動コマンドのクエリキー（既定: move）")
    parser.add_argument("--speed-param", default="speed", help="速度指定のクエリキー（既定: speed）")
    parser.add_argument("--servo-param", default="servo", help="サーボ角度のクエリキー（既定: servo）")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP タイムアウト秒数")
    parser.add_argument("--api-style", choices=("query", "path"), default="query", help="Lesson 6 互換のクエリ式か /move/<cmd> のパス式か")
    parser.add_argument("--move-path", default="/move", help="パス式 API で使用するベースパス（既定: /move）")
    parser.add_argument(
        "--path-command-map",
        nargs="*",
        metavar="direction=name",
        help="パス式 API 用の direction→コマンド名マッピング (例: stop=stopcar left=turnleft)",
    )
    parser.add_argument("--log-level", default="INFO", help="ログレベル（DEBUG/INFO/WARNING/...）")

    subparsers = parser.add_subparsers(dest="command", required=True)

    move = subparsers.add_parser("move", help="forward/backward/left/right/stop などを送信")
    move.add_argument("direction", help="教材ボタン名に合わせた動作名")
    move.add_argument("--speed", type=int, help="0〜100 の整数。指定しない場合はサーバ既定値")

    subparsers.add_parser("stop", help="停止コマンドを送信（move stop の糖衣）")

    speed = subparsers.add_parser("speed", help="速度スライダーのみ更新")
    speed.add_argument("value", type=int, help="0〜100 の整数")

    servo = subparsers.add_parser("servo", help="カメラサーボ角を設定")
    servo.add_argument("angle", type=float, help="-90〜90 程度の角度（度）")

    raw = subparsers.add_parser(
        "raw",
        help="任意のクエリを key=value 形式で指定して送信（例: raw action=command move=forward）",
    )
    raw.add_argument("params", nargs="+", help="key=value 形式で指定してください")

    return parser


def _client_from_args(args: argparse.Namespace) -> CarClient:
    path_map = default_path_command_map()
    if args.path_command_map:
        path_map.update(_parse_key_value_pairs(args.path_command_map))

    return CarClient(
        base_url=args.base_url,
        endpoint=args.endpoint,
        action_param=args.action_param,
        action_value=args.action_value,
        move_param=args.move_param,
        speed_param=args.speed_param,
        servo_param=args.servo_param,
        timeout=args.timeout,
        api_style=args.api_style,
        move_path=args.move_path,
        path_command_map=path_map,
    )


def _handle_move(client: CarClient, args: argparse.Namespace) -> requests.Response:
    return client.move(args.direction, speed=args.speed)


def _handle_stop(client: CarClient, args: argparse.Namespace) -> requests.Response:
    return client.stop()


def _handle_speed(client: CarClient, args: argparse.Namespace) -> requests.Response:
    return client.set_speed(args.value)


def _handle_servo(client: CarClient, args: argparse.Namespace) -> requests.Response:
    return client.set_servo_angle(args.angle)


def _handle_raw(client: CarClient, args: argparse.Namespace) -> requests.Response:
    params = _parse_key_value_pairs(args.params)
    return client.raw_command(params)


def _parse_key_value_pairs(pairs: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid key=value pair: {pair}")
        key, value = pair.split("=", maxsplit=1)
        parsed[key] = value
    return parsed


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = _client_from_args(args)
    handlers = {
        "move": _handle_move,
        "stop": _handle_stop,
        "speed": _handle_speed,
        "servo": _handle_servo,
        "raw": _handle_raw,
    }

    handler = handlers[args.command]
    try:
        response = handler(client, args)
    except requests.RequestException as exc:
        LOG.error("HTTP リクエストに失敗しました: %s", exc)
        return 1
    except ValueError as exc:
        LOG.error("引数エラー: %s", exc)
        return 2

    text = response.text.strip()
    if text:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
