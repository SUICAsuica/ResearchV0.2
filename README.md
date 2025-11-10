# Osoyoo Raspberry Pi Car Bridge (2023)

本リポジトリは、OSOYOO Raspberry Pi Robot Car（2023年版 Lesson 1 / Lesson 6）に合わせて、  
**「ラズパイ側で映像配信とFlask制御を動かし、Mac側でVLM推論・意思決定を行う」** 構成を支援するためのメモとツール群です。  
目的は、教材が想定するブラウザ操作を Mac 上のプログラム（VLM 連携）から自動化しやすくすることにあります。

---

## アーキテクチャ整理

- **映像**: ラズパイ上で `camera.sh` → `startcam.py`（教材 Lesson 6）を起動し、MJPEG ストリームを `http://<PiのIP>:8899/stream.mjpg` で配信する。
- **操作**: 同じくラズパイ上で `picar4.sh` → `webcar.py`（Flask アプリ）を起動し、HTTP リクエストで前進／後退／左右／停止／カメラ角度などを受け付ける。
- **Mac**: 映像ストリームを Pull して VLM 推論 → Flask エンドポイントへ HTTP リクエストで指示を返す。ブラウザで押していたボタン操作を Mac のスクリプトから送るイメージ。

---

## Raspberry Pi 側の準備（Lesson 1 を踏襲）

1. **OS 書き込み**  
   - Mac に Raspberry Pi Imager をインストールし、教材推奨の Raspberry Pi OS（2023年版レッスンが案内する64bit版）を書き込む。  
   - 書き込み時に SSH / Wi-Fi / ロケール設定を済ませておくと後工程が楽。
2. **起動後の基本セットアップ**  
   - Lesson 1 に従い `sudo raspi-config` で I²C とカメラを有効化（Pi 5 の場合は CSI 22ピン→15ピン変換ケーブルを忘れずに）。  
   - PCA9685 HAT や L298N、サーボの配線を教材図面どおりに確認。
3. **教材スクリプトの配置**  
   - Lesson 6 の指示で `camera.sh` / `startcam.py` / `picar4.sh` / `webcar.py` を配置。最新版では `~/osoyoo-robot/` 配下にスクリプトが展開される想定。

---

## ラズパイで映像とFlask制御を起動

```bash
# 映像ストリーム
cd ~/osoyoo-robot
sudo bash camera.sh          # 初回：依存パッケージ導入
python3 startcam.py          # 毎回：カメラ配信（8899番ポート）

# Flask 制御
cd ~/osoyoo-robot
sudo bash picar4.sh          # 初回：ライブラリ導入
python3 webcar.py            # 毎回：Flask サーバ（デフォルト 5000 番）
```

ブラウザから `http://<PiのIP>:8899/stream.mjpg` を開いて動画が表示されること、  
`http://<PiのIP>:5000` にアクセスして教材の操作画面が動くことを Mac から確認しておく。

---

## Mac 側（本リポジトリ）の役割

- `raspycar.server` は **Flask のボタン操作を HTTP 経由で送る軽量クライアント** です。  
  既存の VLM 推論ループから呼び出すことで、ブラウザを経由せずに Lesson 6 のエンドポイントへ指示できます。
- MJPEG の取得や VLM モデル自体はこのリポジトリには含めていません。既存コードから `requests` 等でストリームを Pull してください。

### インストール

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SmolVLM を用いた自律制御ループを使う場合は、追加で以下のパッケージが必要です。

```bash
pip install mlx mlx-vlm opencv-python pillow numpy

### SmolVLM2 モデルの配置

SmolVLM を使う場合は、Hugging Face の MLX 変換済みモデル
`mlx-community/SmolVLM2-500M-Video-Instruct-mlx` をローカルに配置してください。

```bash
huggingface-cli download mlx-community/SmolVLM2-500M-Video-Instruct-mlx \
  --local-dir ./models/smolvlm2-mlx --local-dir-use-symlinks False
```

`model.safetensors` を既にダウンロード済みの場合は、上記コマンドで残りの設定ファイルのみ取得し、
`models/smolvlm2-mlx/` にまとめて配置します。環境変数 `SMOL_MODEL_ID` を設定するか、
`--smol-model-id ./models/smolvlm2-mlx` を指定すると `raspycar.autopilot` がこのモデルを読み込みます。
```

### コマンドライン例

```bash
# 前進（必要に応じて speed パラメータを付与）
python -m raspycar.server --base-url http://192.168.11.4:5000 move forward --speed 40

# 左旋回
python -m raspycar.server --base-url http://192.168.11.4:5000 move left

# 停止
python -m raspycar.server --base-url http://192.168.11.4:5000 stop

# カメラパン（角度を絶対指定）
python -m raspycar.server --base-url http://192.168.11.4:5000 servo 10
```

VLM 連携時は、推論結果に応じて上記コマンド相当の関数を呼び出す想定です。  
エンドポイントやクエリはカスタマイズ可能で、教材側の `webcar.py` を変更した場合でも `--move-param` 等で合わせられます。

---

## SmolVLM を用いた簡易自律走行

`raspycar.autopilot` は MJPEG ストリームを取り込み、SmolVLM モデルで白枠ターゲットを検出しながら Lesson 6 の Flask エンドポイントへ前進／左右／停止コマンドを送るサンプルループです。

```bash
python -m raspycar.autopilot \
  --camera-url http://192.168.11.4:8899/stream.mjpg \
  --base-url http://192.168.11.4:5000 \
  --smol-model-id /Users/shori/laboratory/Researchv2.0/raspycar/models/smolvlm2-mlx
```

または環境変数 `CAM_URL`, `BASE_URL`, `MODEL_DIR`（任意）を設定しておけば、リポジトリ直下で `make autopilot` だけで同じコマンドが実行できます。`\` のエスケープ漏れや改行を気にする必要がありません。`DEBUG` ログを見たい場合は `make autopilot-debug` を使用してください（`AUTOPILOT_LOG_LEVEL` で上書きも可能）。追加オプションを渡したい場合は `EXTRA_ARGS='--min-confidence 0.9 --detection-timeout 1.0' make autopilot` のように環境変数で指定します。

- `--smol-model-id` にはローカルの SmolVLM 重みディレクトリ、または Hugging Face 上のモデル ID（例: `mlx-community/SmolVLM2-500M-Video-Instruct-mlx`）を指定します。環境変数 `SMOL_MODEL_ID` が設定されていれば省略可能です。
- デフォルトでは OpenCV プレビューウィンドウを開きます。不要な場合は `--no-preview` を付与してください。
- ターゲット面積や左右旋回の判定しきい値は `--stop-area-ratio`、`--turn-deadzone` などのオプションで調整できます。
- Pi 側 Flask が `/move/<command>` 形式（例: `/move/forward`, `/move/stopcar`）を提供している場合は、`--car-api-style path --base-url http://192.168.11.4:80` を指定してください。必要に応じて `--car-path-map stop=stopcar left=turnleft` のように direction→エンドポイント名を上書きできます（既定マッピングは Osoyoo Lesson 6 に合わせています）。パス式 API では速度パラメータが受け付けられないため、`forward_speed` / `turn_speed` は無視されます。
- `--min-confidence` で SmolVLM が返す `confidence` の最低値を指定できます（デフォルト 0.85）。閾値未満の検出は破棄されるため、誤検出を抑えたい場合に利用してください。

SmolVLM の応答 JSON は「ターゲットが見つかったか」とその位置・サイズを返す構成を想定しており、デフォルトの `--system-prompt` / `--user-prompt` は Hugging Face MLX コミュニティ README で推奨されている JSON 形式（`present`, `confidence`, `center_x`, `center_y`, `box_width`, `box_height` を必ず含む辞書）をそのまま使用します。別の目標を扱う場合は以下のようにプロンプトを上書きしてください。

```bash
python -m raspycar.autopilot \
  --camera-url http://192.168.11.4:8899/stream.mjpg \
  --base-url http://192.168.11.4:5000 \
  --smol-model-id ./models/smolvlm2-mlx \
  --system-prompt "You are a vision model that must always respond with a JSON dict containing keys: present, confidence, center_x, center_y, box_width, box_height." \
  --user-prompt "Given the image, output only JSON like {\"present\": true, \"confidence\": 0.8, \"center_x\": 0.5, \"center_y\": 0.5, \"box_width\": 0.2, \"box_height\": 0.2}."
```

---

## 主要関数（VLM から直接呼び出す場合）

```python
from raspycar.server import CarClient

client = CarClient(base_url="http://192.168.11.4:5000")
client.move("forward", speed=40)
client.move("left")
client.stop()
client.set_servo_angle(10)
```

- `CarClient.move(direction, speed=None)`  
  - `direction`: `forward`, `backward`, `left`, `right`, `stop` など教材のボタン名に合わせる。  
  - `speed`: 0〜100 相当（教材の UI スライダーに合わせて整数化）。
- `CarClient.stop()` は `move("stop")` の糖衣構文。
- `CarClient.set_servo_angle(angle_deg)` は `/?action=command&servo=<角度>` を呼び出し、教材で定義された範囲に収める。
- `CarClient.raw_command(params)` で任意のクエリパラメータを送ることも可能。

---

## トラブルシュート

- **映像が取得できない**  
  - `startcam.py` を再起動し、Mac からブラウザで `http://<PiのIP>:8899/stream.mjpg` にアクセスして切り分ける。
- **HTTP で制御できない**  
  - Flask (`webcar.py`) が稼働しているかを `ps -ef | grep webcar.py` などで確認。  
  - `CarClient` の `base_url` が `http://<PiのIP>:5000` になっているか、LAN 内から疎通できるかをチェック。
- **Pi 5 のカメラが認識しない**  
  - 22ピン→15ピン変換ケーブルを用意し、差し込み向きとロックレバーの締め具合を確認。

---

## 備考

- `hardware.py` は教材外で直接 PCA9685 を操作したい場合の補助モジュールとして残しています（ラズパイ上でのみ利用可能）。  
  通常の Lesson 6 ワークフローでは使用しません。
- 旧 ESP32-CAM／WebSocket 互換のコードは削除しました。既存プロジェクトから移行する際は `CarClient` を直接呼び出す形に書き換えてください。

---

## ライセンス

このリポジトリ内のコードおよびドキュメントは、特記がない限り MIT ライセンスです。
