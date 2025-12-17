# raspycar 実験ガイド（draft_2025-12-10_v1 対応）

卒研論文 `/Users/shori/laboratory/paper_agent/draft_2025-12-10_v1.pdf` の実験を再現するための最小手順だけをまとめました。よく使う手順は上側、今回使わなかったフローは「参考」に退避しています。

---

## すぐ使う手順（ここだけ見れば走る）
1. **Raspberry Pi でエージェント起動（8080 番）**
   ```bash
   cd ~/laboratory/Researchv2.0/raspycar
   sudo -E PYTHONPATH=/home/shori/laboratory:/home/shori/laboratory/Researchv2.0 \
     /home/shori/laboratory/Researchv2.0/raspycar/.venv/bin/python \
     -m raspycar.raspi_agent \
       --bind 0.0.0.0 --port 8080 \
       --camera-source 0 \
       --camera-width 640 --camera-height 480 --camera-fps 5 \
       --watchdog-timeout 2.0 \
       --servo-center -50
   ```
   - 広角 USB カメラを直接読む構成。`/frame.jpg` / `/stream.mjpg` を配信し、2 秒コマンドが来ないと自動 `STOP`。
   - サーボ初期位置が左に寝る場合は `--servo-center` を微調整。

2. **Mac/PC から制御（GPT 版ダイレクト）**
   ```bash
   cd ~/laboratory/Researchv2.0/raspycar
   source .venv/bin/activate
   source env.home.sh        # 自宅=192.168.0.13 の例（環境に応じて変更）
   make pc-direct-gpt        # VLM が一語コマンドを出し続ける構成
   ```
   - `AGENT_URL` は `env.home.sh` / `env.home.26.sh` / `env.lab.sh` で切替。Pi 側を変えたらこの README と `Agents.md` も更新。
   - デフォルトプロンプトは論文記載の「黄色い箱が見えれば接近、近距離で STOP。見えなければ STOP」を実装済み。

3. **ログと安全運用**
   - `--log-level DEBUG` を付けると VLM 生出力と送信コマンドが時系列で残る（研究ノート用）。
   - WATCHDOG に頼らず、必ず手元で STOP を即送れる状態を維持。

---

## この構成で使うコンポーネント
- `raspycar/raspi_agent.py` … Pi 側 HTTP サーバ。映像配信 + `/command` でモータ制御。WATCHDOG 2 秒。
- `pc_controller_direct_gpt.py` … GPT-5-mini-2025-08-07 に一語コマンドを出させ、そのまま `raspi_agent` に送信。
- `Makefile` … `make pc-direct-gpt` / `make pc-hybrid-gpt` / `make raspi-agent` など。`EXTRA_ARGS` でオプション追加可。
- 環境切替 … `env.home.sh`（192.168.0.13）、`env.home.26.sh`（192.168.0.26）、`env.lab.sh`（192.168.11.3）。

---

## 実験メモ（再現時のチェックポイント）
- ループ間隔や遅延は PC 側ログで確認し、走行距離と合わせて記録する。
- カメラは 640×480 / 5 fps を前提に論文を書いている。解像度や fps を変えた場合は別途記載。
- 推論が途切れたら `GET /health` で `dry_run` / `last_command` を確認。

---

## 参考：今回使わなかったフロー
### SmolVLM 系（旧評価用）
- `pc_controller_direct.py` / `pc_controller_hybrid.py` と `make pc-direct` / `make pc-hybrid` は SmolVLM2-mlx を使った A/B 比較用。モデルは `models/smolvlm2-mlx` を配置し、`mlx` 系パッケージを追加インストールする。
- `raspycar.autopilot` は Lesson 6 の Flask API（5000 番）向け自律走行サンプル。`make autopilot` で起動。

### Lesson 6 (`webcar.py`) 互換手順
- カメラ配信: `cd ~/osoyoo-robot && python3 startcam.py`（8899 番）。
- Flask 制御: `cd ~/osoyoo-robot && python3 webcar.py`（5000 番）。
- コマンド送信例: `python -m raspycar.server --base-url http://<Pi>:5000 move forward --speed 40`。

### CarClient API 簡易リファレンス
- `from raspycar.server import CarClient`
- `CarClient(base_url).move("forward", speed=40)`, `move("left")`, `stop()`, `set_servo_angle(10)` など。

（詳細なチュートリアルやオプションの説明は旧 README を参照したい場合に備えて `git show` 等で履歴から取得してください。）

---

## 付録：フルセットアップ手順（SD書き込みから実験開始まで）
最短フローだけで不安な場合は、ここを上から実施すれば Pi も Mac も揃います。

1. **Raspberry Pi OS を microSD に書き込む（Mac）**
   - Raspberry Pi Imager → OS: Raspberry Pi OS 64-bit（Liteでも可）。
   - 「設定」で SSH 有効化・Wi‑Fi SSID/パスワード・ロケール(JP)を事前入力して書き込み。

2. **Pi を起動し IP を確認**
   - 起動後、Mac から `arp -a | grep raspberrypi` またはルーター画面で IP を確認。
   - 既定の静的 IP（2025-11-25 時点）は自宅 `192.168.0.13` / 研究室 `192.168.11.3`。DHCP で自宅が `192.168.0.26` になる場合あり。
   - SSH でログイン: `ssh pi@<PiのIP>`（初期パスは raspberry。初回ログインで必ず変更）。
   - OS セットアップ時にユーザ名を `shori` にした場合の例: `ssh shori@192.168.11.34`（研究室ネットワークで IP が 11.34 になっているケース）。

3. **Pi 側にコードと依存を入れる**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-venv python3-dev git ffmpeg libjpeg-dev
   mkdir -p ~/laboratory && cd ~/laboratory
   git clone https://github.com/SUICAsuica/ResearchV0.2.git Researchv2.0
   cd Researchv2.0/raspycar
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   - カメラ確認（任意）: `libcamera-still -o test.jpg`

4. **Pi でエージェント起動（研究本番構成）**
   ```bash
   cd ~/laboratory/Researchv2.0/raspycar
   sudo -E PYTHONPATH=/home/pi/laboratory:/home/pi/laboratory/Researchv2.0 \
     /home/pi/laboratory/Researchv2.0/raspycar/.venv/bin/python \
     -m raspycar.raspi_agent \
       --bind 0.0.0.0 --port 8080 \
       --camera-source 0 \
       --camera-width 640 --camera-height 480 --camera-fps 5 \
       --watchdog-timeout 2.0 \
       --servo-center -50
   ```
   - `http://<PiのIP>:8080/frame.jpg` で映像、`/command` に `FORWARD/LEFT/...` を送る。

5. **Mac 側を準備**
   ```bash
   cd ~/laboratory/Researchv2.0/raspycar
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   source env.home.sh   # または env.home.26.sh / env.lab.sh
   ```

6. **実験を開始（GPT ダイレクト）**
   ```bash
   make pc-direct-gpt
   # 必要に応じ: EXTRA_ARGS="--log-level DEBUG --loop-interval 0.5" make pc-direct-gpt
   ```
   - ログは `logs/gpt-direct-YYYYMMDD-HHMMSS.log`。STOP を即送れるように準備して運用。

7. **コードを Pi に送る場合**
   - 軽微変更: `scp -r ~/laboratory/Researchv2.0/raspycar pi@<PiのIP>:/home/pi/laboratory/Researchv2.0/`
   - もしくは Pi 上で `git pull`（秘密情報を入れた `.env` は追跡しないこと）。

8. **Wi‑Fi 再接続時の IP 再確認**
   - Pi で `hostname -I`
   - Mac で `ping <hostname>.local`（mDNS 有効時）

9. **トラブルシュート簡易メモ**
   - GPIO 権限エラー → 上記のとおり root で `raspi_agent` を起動。
   - カメラ未認識 → `--camera-source` を 0/1 で切り替え、解像度は 640×480 に戻す。
   - コマンドが届かない → `curl http://<Pi>:8080/health` で `dry_run` / `last_command` を確認。

---

## 付録：フルセットアップ手順（SD書き込みから実験開始まで）
最短フローだけで不安な場合は、ここを上から実施すれば Pi も Mac も揃います。

1. **Raspberry Pi OS を microSD に書き込む（Mac）**
   - Raspberry Pi Imager → OS: Raspberry Pi OS 64-bit（Liteでも可）。
   - 「設定」で SSH 有効化・Wi‑Fi SSID/パスワード・ロケール(JP)を事前入力して書き込み。

2. **Pi を起動し IP を確認**
   - 起動後、Mac から `arp -a | grep raspberrypi` またはルーター画面で IP を確認。
   - 既定の静的 IP（2025-11-25 時点）は自宅 `192.168.0.13` / 研究室 `192.168.11.3`。DHCP で自宅が `192.168.0.26` になる場合あり。
   - SSH でログイン: `ssh pi@<PiのIP>`（初期パスは raspberry。初回ログインで必ず変更）。

3. **Pi 側にコードと依存を入れる**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-venv python3-dev git ffmpeg libjpeg-dev
   mkdir -p ~/laboratory && cd ~/laboratory
   git clone https://github.com/SUICAsuica/ResearchV0.2.git Researchv2.0
   cd Researchv2.0/raspycar
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   - カメラ確認（任意）: `libcamera-still -o test.jpg`

4. **Pi でエージェント起動（研究本番構成）**
   ```bash
   cd ~/laboratory/Researchv2.0/raspycar
   sudo -E PYTHONPATH=/home/pi/laboratory:/home/pi/laboratory/Researchv2.0 \
     /home/pi/laboratory/Researchv2.0/raspycar/.venv/bin/python \
     -m raspycar.raspi_agent \
       --bind 0.0.0.0 --port 8080 \
       --camera-source 0 \
       --camera-width 640 --camera-height 480 --camera-fps 5 \
       --watchdog-timeout 2.0 \
       --servo-center -50
   ```
   - `http://<PiのIP>:8080/frame.jpg` で映像、`/command` に `FORWARD/LEFT/...` を送る。

5. **Mac 側を準備**
   ```bash
   cd ~/laboratory/Researchv2.0/raspycar
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   source env.home.sh   # または env.home.26.sh / env.lab.sh
   ```

6. **実験を開始（GPT ダイレクト）**
   ```bash
   make pc-direct-gpt
   # 必要に応じ: EXTRA_ARGS="--log-level DEBUG --loop-interval 0.5" make pc-direct-gpt
   ```
   - ログは `logs/gpt-direct-YYYYMMDD-HHMMSS.log`。STOP を即送れるように準備して運用。

7. **コードを Pi に送る場合**
   - 軽微変更: `scp -r ~/laboratory/Researchv2.0/raspycar pi@<PiのIP>:/home/pi/laboratory/Researchv2.0/`
   - もしくは Pi 上で `git pull`（秘密情報を入れた `.env` は追跡しないこと）。

8. **Wi‑Fi 再接続時の IP 再確認**
   - Pi で `hostname -I`
   - Mac で `ping <hostname>.local`（mDNS 有効時）

9. **トラブルシュート簡易メモ**
   - GPIO 権限エラー → 上記のとおり root で `raspi_agent` を起動。
   - カメラ未認識 → `--camera-source` を 0/1 で切り替え、解像度は 640×480 に戻す。
   - コマンドが届かない → `curl http://<Pi>:8080/health` で `dry_run` / `last_command` を確認。
