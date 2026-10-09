#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
echo 'セットアップが完了しました。start_mac.command を実行してください。'
