# 🌸 AR Pookalam (Continuous Gemma 3 Live Judge)

Two-player split-screen Augmented Reality Pookalam mandala design game with live Onam judging powered by Google's Gemma 3 on local Ollama.

---

## 📦 Files to Transfer to Raspberry Pi

You only need **2 files**:

1. `ar_pookalam.py` - The complete AR game engine.
2. `requirements.txt` - Dependencies list.

*(Do NOT copy the `venv/` folder from Mac, as it contains Mac ARM binaries. You will create a fresh venv on the Pi in 10 seconds!)*

---

## 🚀 Raspberry Pi 5 Setup (Step-by-Step)

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1-mesa-glx libglib2.0-0
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install & Start Ollama with Gemma 3

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model (Recommended for 4GB Pi):
ollama run gemma3:4b-it-qat
# Or lightweight 1B model:
# ollama run gemma3:1b
```

### 4. Run the Game

```bash
# For USB Webcam:
python3 ar_pookalam.py

# For Official Raspberry Pi Camera Module:
libcamerify python3 ar_pookalam.py
```
