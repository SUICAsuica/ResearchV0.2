# Osoyoo Raspberry Pi Car Bridge (2023)

本リポジトリは、OSOYOO Raspberry Pi Robot Car（2023年版 Lesson 1 / Lesson 6）に合わせて、  
**「ラズパイ側で映像配信と制御エージェント（`raspi_agent.py`）を走らせ、Mac 側で VLM 推論・意思決定を行う」** 構成を支援するためのメモとツール群です。  
研究タスクでは 8080 番ポートの `raspi_agent` + `pc_controller_direct` / `pc_controller_hybrid` をメイン経路として使用します。Lesson 6 の `webcar.py` 互換手順は参考情報として末尾にまとめています。

---

## アーキテクチャ整理

- **映像＋操作（研究フロー）**: ラズパイ上で `raspi_agent.py` を常駐させ、`http://<PiのIP>:8080/frame.jpg` / `/stream.mjpg` を配信しつつ、`/command` で `FORWARD/LEFT/...` を受け付ける。
- **映像＋操作（Lesson 6 互換）**: 教材どおり `startcam.py`（8899 番）と `webcar.py`（5000 番）を起動するパターンも維持しており、必要な場合は「Lesson 6 (webcar.py) 互換運用」セクションを参照する。
- **Mac**: `pc_controller_direct.py` / `pc_controller_hybrid.py` などのクライアントが `AGENT_URL`（8080 番）へリクエストを送り、ブラウザ操作を人手で行っていた部分を自動化する。

研究タスクを進める場合は、以下の 3 コンポーネントをセットで使います。

1. **`raspi_agent.py`**: ラズパイ上でカメラフレームを JPEG 化しつつ、PC からのコマンド (`FORWARD/LEFT/...`) を受け付ける簡易 HTTP サーバ。教材の `webcar.py` とは独立しており、研究用の通信インターフェースを提供します。
2. **`pc_controller_direct.py`**: SmolVLM2-mlx に「次の行動コマンドそのもの」を決めさせるダイレクト制御実験用クライアント。
3. **`pc_controller_hybrid.py`**: SmolVLM2-mlx には「ターゲットの位置・距離」だけ答えさせ、実際の移動はルールベースで決めるハイブリッド制御クライアント。

これらを同じターゲット到達タスクで比較することで、「VLM をどこまで信用してロボットを動かすべきか？」という卒研テーマに沿った実験が行えます。

---

## Raspberry Pi 側の準備（Lesson 1 を踏襲）

1. **OS 書き込み**  
   - Mac に Raspberry Pi Imager をインストールし、教材推奨の Raspberry Pi OS（2023年版レッスンが案内する64bit版）を書き込む。  
   - 書き込み時に SSH / Wi-Fi / ロケール設定を済ませておくと後工程が楽。
2. **起動後の基本セットアップ**  
   - Lesson 1 に従い `sudo raspi-config` で I²C とカメラを有効化（Pi 5 の場合は CSI 22ピン→15ピン変換ケーブルを忘れずに）。  
   - PCA9685 HAT や L298N、サーボの配線を教材図面どおりに確認。
3. **研究用エージェントの配置**  
   - 本リポジトリ（`raspycar/`）をラズパイにも配置し、`raspi_agent.py` を起動できるようにしておく。Lesson 6 の `camera.sh` / `webcar.py` は互換運用が必要なときだけ導入すればよい（後述）。

---

## ラズパイで `raspi_agent` (8080) を起動

研究フローでは `raspi_agent.py` が映像配信とモータ制御を同時に担当します。Pi に SSH したら次のように起動してください（`make raspi-agent EXTRA_ARGS='...'` で同じコマンドをラップできます）。

```bash
cd ~/laboratory/Researchv2.0/raspycar
python3 raspi_agent.py \
  --bind 0.0.0.0 \
  --port 8080 \
  --camera-source 0 \
  --camera-width 640 --camera-height 480 --camera-fps 2 \
  --watchdog-timeout 2.0
```

- `GET /frame.jpg` / `/stream.mjpg` で最新フレームを取得し、`POST /command {"command": "FORWARD"}` などで操作します。
- 2 秒間コマンドが来ないと自動で `STOP` を送る WATCHDOG が有効です。実験中も手元で停止コマンドを即送れるようにしておいてください。
- 旧 `webcar.py` を同時に立ち上げる必要はありません（ポートも競合しません）。Lesson 6 互換フローが必要な場合のみ本 README の末尾セクションを参照してください。

---

## Mac 側（本リポジトリ）の役割

- `pc_controller_direct.py` / `pc_controller_hybrid.py` が `raspi_agent` (8080) と通信し、VLM 応答をそのままコマンド化する、または VLM 認識＋ルール制御で動かします。
- `raspycar.server` や `CarClient` など、Lesson 6 の `webcar.py` を直接操作するツールも残してありますが、通常の研究フローでは不要です（利用方法は末尾の互換セクション参照）。
- MJPEG の取得や VLM モデル本体はこのリポジトリには含めていないため、必要なパッケージは後述のインストール手順に従ってセットアップしてください。

---

## 拠点別 IP と環境切り替え

- 2025-11-12 時点の静的 IP は **自宅 = `192.168.0.12` / 研究室 = `192.168.11.11`** です。
- それぞれの環境で `AGENT_URL` / `CAM_URL` / `BASE_URL` を取り違えないよう、`env.home.sh` と `env.lab.sh` を用意しました。作業前に `source` してください。

```bash
# 自宅（192.168.0.12）で direct 制御を試す例
source env.home.sh
make pc-direct

# 研究室（192.168.11.11）で hybrid 制御を試す例
source env.lab.sh
make pc-hybrid
```

- スクリプトは `cam`/`base`/`agent` それぞれに現在の IP を埋め込むだけなので、値を変えたら README と `Agents.md` の更新履歴に日付付きで記録してください。

---

## 研究用 3 スクリプトの要点

| 役割 | ファイル | 主な機能 |
| --- | --- | --- |
| ラズパイ側エージェント | `raspi_agent.py` | カメラを `frame.jpg`/`stream.mjpg` として配信、`/command` で `FORWARD/LEFT/...` を受信してモータ制御。2 秒コマンドが来なければ自動 STOP。 |
| PC: VLM ダイレクト制御 | `pc_controller_direct.py` | SmolVLM に「次の行動コマンドを 1 語で答えよ」と指示し、そのまま `raspi_agent` に送信。VLM の揺れをそのまま観測できる。 |
| PC: VLM + ルール制御 | `pc_controller_hybrid.py` | SmolVLM には JSON で `position/distance/confidence` だけ答えさせ、左右調整と減速／停止は手書きルールで決める。 |

### 1. Raspberry Pi: `raspi_agent.py`

```bash
python3 raspi_agent.py \
  --bind 0.0.0.0 \
  --port 8080 \
  --camera-source 0 \
  --camera-width 640 --camera-height 480 --camera-fps 2 \
  --watchdog-timeout 2.0
```

- `GET /frame.jpg` … 最新の JPEG 1 枚を返す。
- `GET /stream.mjpg` … シンプルな MJPEG ストリーム。
- `POST /command {"command": "FORWARD"}` … 走行コマンドを適用（`FORWARD/LEFT/RIGHT/FORWARD_SLOW/BACKWARD/STOP`）。
- `GET /health` … `last_command` や `dry_run` を JSON で返す。

### 2. PC: `pc_controller_direct.py`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt mlx mlx-vlm opencv-python pillow numpy

python pc_controller_direct.py \
  --agent-url http://192.168.0.12:8080 \
  --instruction "Please stop in front of the black UCC bottle" \
  --model-id ./models/smolvlm2-mlx \
  --loop-interval 1.0
```

- ループ毎に `frame.jpg` を取得 → SmolVLM で `LEFT/RIGHT/FORWARD/FORWARD_SLOW/STOP` のどれか 1 つを選択 → `raspi_agent` に送信。
- ログには VLM の生出力を DEBUG で残すので、コマンド切り替え回数や揺れを統計化しやすい。

### 3. PC: `pc_controller_hybrid.py`

```bash
python pc_controller_hybrid.py \
  --agent-url http://192.168.0.12:8080 \
  --instruction "Move to the black bottle and stop" \
  --model-id ./models/smolvlm2-mlx \
  --min-confidence 0.4
```

- SmolVLM には JSON (`{"position": "LEFT", "distance": "FAR", "confidence": 0.82}`) で答えさせ、
  - 位置が LEFT/RIGHT なら旋回、CENTER なら距離（FAR/MID/NEAR）に応じて前進/減速/停止。
- 同じタスクを `pc_controller_direct.py` と比較することで、成功率・到達時間・コマンド反転回数などを評価可能。

Makefile には以下のターゲットを用意しています。

```bash
# 環境変数 CAM_URL / BASE_URL / MODEL_DIR を従来どおり設定してから:
make raspi-agent       # Pi 上の agent を python で起動（ssh越しに利用）
make pc-direct        # Mac 側で direct 制御を起動（AGENT_URL, INSTRUCTION で上書き可）
make pc-hybrid        # Mac 側で hybrid 制御を起動
```

必要に応じて `EXTRA_ARGS` を渡すと CLI オプションを追加できます。

### インストール

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SmolVLM を用いた自律制御ループを使う場合は、追加で以下のパッケージが必要です。

```bash
pip install mlx mlx-vlm opencv-python pillow numpy
```

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

## Lesson 6 (`webcar.py`) 互換運用（オプション）

8080 番の `raspi_agent` だけで実験が完結する場合、このセクションの手順は不要です。ブラウザ UI や教材の Flask API（5000 番）を維持したいときだけ参照してください。

### ラズパイで Lesson 6 を起動する

```bash
# 映像ストリーム（8899 番）
cd ~/osoyoo-robot
sudo bash camera.sh          # 初回：依存パッケージ導入
python3 startcam.py          # 毎回：カメラ配信

# Flask 制御（5000 番）
cd ~/osoyoo-robot
sudo bash picar4.sh          # 初回：ライブラリ導入
python3 webcar.py            # 毎回：Flask サーバ
```

`http://<PiのIP>:8899/stream.mjpg` と `http://<PiのIP>:5000` にアクセスできれば準備完了です。`raspi_agent` と併用してもポートは競合しません。

### コマンドライン例（`raspycar.server`）

```bash
# 前進（必要に応じて speed パラメータを付与）
python -m raspycar.server --base-url http://192.168.0.12:5000 move forward --speed 40

# 左旋回
python -m raspycar.server --base-url http://192.168.0.12:5000 move left

# 停止
python -m raspycar.server --base-url http://192.168.0.12:5000 stop

# カメラパン（角度を絶対指定）
python -m raspycar.server --base-url http://192.168.0.12:5000 servo 10
```

VLM 連携時は、推論結果に応じて上記コマンド相当の関数を呼び出す想定です。  
エンドポイントやクエリはカスタマイズ可能で、教材側の `webcar.py` を変更した場合でも `--move-param` 等で合わせられます。

---

### SmolVLM を用いた簡易自律走行（`webcar.py` 向け）

`raspycar.autopilot` は MJPEG ストリームを取り込み、SmolVLM モデルで黒いコーヒーボトル（UCC BLACK）の位置を検出しながら Lesson 6 の Flask エンドポイントへ前進／左右／停止コマンドを送るサンプルループです。

```bash
python -m raspycar.autopilot \
  --camera-url http://192.168.0.12:8899/stream.mjpg \
  --base-url http://192.168.0.12:5000 \
  --smol-model-id /Users/shori/laboratory/Researchv2.0/raspycar/models/smolvlm2-mlx
```

または環境変数 `CAM_URL`, `BASE_URL`, `MODEL_DIR`（任意）を設定しておけば、リポジトリ直下で `make autopilot` だけで同じコマンドが実行できます。`\` のエスケープ漏れや改行を気にする必要がありません。`DEBUG` ログを見たい場合は `make autopilot-debug` を使用してください（`AUTOPILOT_LOG_LEVEL` で上書きも可能）。追加オプションを渡したい場合は `EXTRA_ARGS='--min-confidence 0.9 --detection-timeout 1.0' make autopilot` のように環境変数で指定します。

- `--smol-model-id` にはローカルの SmolVLM 重みディレクトリ、または Hugging Face 上のモデル ID（例: `mlx-community/SmolVLM2-500M-Video-Instruct-mlx`）を指定します。環境変数 `SMOL_MODEL_ID` が設定されていれば省略可能です。
- デフォルトでは OpenCV プレビューウィンドウを開きます。不要な場合は `--no-preview` を付与してください。
- ターゲット面積や左右旋回の判定しきい値は `--stop-area-ratio`、`--turn-deadzone` などのオプションで調整できます。
- Pi 側 Flask が `/move/<command>` 形式（例: `/move/forward`, `/move/stopcar`）を提供している場合は、`--car-api-style path --base-url http://192.168.0.12:80` を指定してください。必要に応じて `--car-path-map stop=stopcar left=turnleft` のように direction→エンドポイント名を上書きできます（既定マッピングは Osoyoo Lesson 6 に合わせています）。パス式 API では速度パラメータが受け付けられないため、`forward_speed` / `turn_speed` は無視されます。
- `--detection-interval`（デフォルト 6）で推論を挟むフレーム間隔、`--stop-area-ratio`（デフォルト 0.08）で停止判定面積が調整できます。
- `--bottom-focus-ratio`（デフォルト 0.6）で SmolVLM に渡す画像の下部割合を指定できます。床付近にあるターゲットへ注意を集中させたい場合に有効です。
- `--min-confidence` で SmolVLM が返す `confidence` の最低値を指定できます（デフォルト 0.5）。閾値未満の検出は破棄されるため、誤検出を抑えたい場合に利用してください。
- どうしても VLM が応答しない場合に備え、`--enable-dark-bottle-heuristic` を付けると簡易的な暗色ボトル推定を追加できます（デフォルトは無効）。
- `--low-latency` を付けると OpenCV プレビューを強制的に無効化し、`VideoCapture` のバッファを 1 フレームに制限してフレーム遅延を抑えます（`poll_sleep` も最大全 0.01 秒に切り詰め）。ブラウザでストリームを確認しつつ Mac 側は制御専用にしたい場合に便利です。

SmolVLM の応答 JSON は「ターゲットが見つかったか」とその位置・サイズを返す構成を想定しており、デフォルトの `--system-prompt` / `--user-prompt` は UCC BLACK ボトルの検出に特化した文面になっています（`present`, `confidence`, `center_x`, `center_y`, `box_width`, `box_height` を必ず含む辞書のみ許可）。別の目標を扱う場合は以下のようにプロンプトを上書きしてください。

```bash
python -m raspycar.autopilot \
  --camera-url http://192.168.0.12:8899/stream.mjpg \
  --base-url http://192.168.0.12:5000 \
  --smol-model-id ./models/smolvlm2-mlx \
  --system-prompt "You are a vision model that must always respond with a JSON dict containing keys: present, confidence, center_x, center_y, box_width, box_height." \
  --user-prompt "Given the image, output only JSON like {\"present\": true, \"confidence\": 0.8, \"center_x\": 0.5, \"center_y\": 0.5, \"box_width\": 0.2, \"box_height\": 0.2}."
```

---

### 主要関数（`CarClient` / `webcar.py`）

```python
from raspycar.server import CarClient

client = CarClient(base_url="http://192.168.0.12:5000")
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

- **映像が取得できない / raspi_agent が応答しない**  
  - `curl http://<PiのIP>:8080/health` で状態を確認し、ログにカメラ初期化エラーが出ていないかチェックする。  
  - `--camera-source`（USB カメラ番号）や解像度が環境に合っているかを見直し、必要なら `EXTRA_ARGS='--camera-source 1' make raspi-agent` のように上書きする。
- **8080 へのコマンドが届かない**  
  - `env.home.sh` / `env.lab.sh` を読みなおして `AGENT_URL` が現在のネットワークになっているか確認する。  
  - `raspi_agent` の WATCHDOG が STOP を出し続けていないか（`/health` の `last_command` を参照）。
- **Pi 5 のカメラが認識しない**  
  - 22ピン→15ピン変換ケーブルを用意し、差し込み向きとロックレバーの締め具合を確認。
- **Lesson 6 (`webcar.py`) を使う場合の追加切り分け**  
  - `startcam.py` / `webcar.py` の稼働状況、`CarClient` の `base_url`（5000 番）を確認する。詳しくは「Lesson 6 互換運用」セクションを参照。

---

## 備考

- `hardware.py` は教材外で直接 PCA9685 を操作したい場合の補助モジュールとして残しています（ラズパイ上でのみ利用可能）。  
  通常の Lesson 6 ワークフローでは使用しません。
- 旧 ESP32-CAM／WebSocket 互換のコードは削除しました。既存プロジェクトから移行する際は `CarClient` を直接呼び出す形に書き換えてください。

---

## ライセンス

このリポジトリ内のコードおよびドキュメントは、特記がない限り MIT ライセンスです。
