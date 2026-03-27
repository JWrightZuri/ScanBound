#!/bin/bash
echo "[1/3] Creating Virtual Environment..."
python3 -m venv venv
echo "[2/3] Activating Environment..."
source venv/bin/activate
echo "[3/3] Installing Dependencies..."
pip install -r requirements.txt
echo "Setup Complete!"