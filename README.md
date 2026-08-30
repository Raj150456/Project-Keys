# 🌸 AR Pookalam: Gemma 3 Live Judge

![AR Pookalam Banner](https://img.shields.io/badge/Status-Hackathon_Ready-success?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python) ![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-orange?style=for-the-badge) ![Ollama](https://img.shields.io/badge/AI-Ollama_Gemma3-white?style=for-the-badge)

**AR Pookalam** is a highly interactive, two-player split-screen Augmented Reality game where players craft beautiful traditional Onam mandalas (Pookalams) in mid-air using their bare hands.

But there's a twist: A live, local AI judge powered by **Google's Gemma 3** (via Ollama) watches the game in real-time, calculating symmetry, color variety, and completeness to provide live commentary and crown the ultimate Pookalam artisan!

---

## 🚀 The Hackathon Edge

We built this to combine **Kerala’s rich cultural heritage** with **cutting-edge Spatial AI**.

1. **Zero Latency AR:** Uses Google MediaPipe to track index fingers with millimeter precision—no gloves, no controllers.
2. **Local AI (No Internet Required):** The entire AI judging pipeline runs locally on an Ollama daemon. It evaluates the aesthetics of the mandala and streams live commentary to the screen in milliseconds.
3. **Immersive Glassmorphism UI:** Features an incredibly sleek, highly-readable UI with translucent floating palettes and smooth 60FPS anti-aliased floral rendering.

---

## 🧠 Architecture (Sense → Think → Act)

* **SENSE:** OpenCV captures the webcam feed while MediaPipe extracts 3D hand landmarks to detect the player's "Magic Wand" index finger.
* **THINK:** A continuous background thread calculates symmetry algorithms and sends real-time game state JSONs to **Gemma 3**.
* **ACT:** The UI dynamically renders the virtual flowers, tracks scores, updates leaderboards, and flashes live AI commentary on a beautiful glassmorphic HUD.

---

## 🎮 Features

* **Multiplayer Split-Screen:** Player 1 (Golden Mandala) vs Player 2 (Royal Mandala).
* **Virtual Flower Palette:** 7 traditional Kerala floral colors (Marigold Gold, Saffron Orange, Lotus Pink, etc.).
* **Generative Art Evaluation:** The AI rule engine calculates completeness, petal symmetry, and color variance.
* **Fallback Engine:** If Ollama isn't running, the game seamlessly falls back to a lightning-fast offline rule engine.

---

## 🛠️ Tech Stack

* **Computer Vision:** OpenCV, Google MediaPipe
* **Generative AI:** Ollama, Gemma 3 (1B/2B/4B depending on hardware constraints)
* **Language:** Python 3
* **Concurrency:** Native Python Threading for non-blocking UI

---

## 💻 Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Requires `opencv-python` and `mediapipe`)*

### 2. Start the Local AI Judge (Optional but Recommended)

Install [Ollama](https://ollama.com/) and download the Gemma 3 model:

```bash
ollama run gemma:2b
```

### 3. Launch the Game

```bash
python ar_pookalam.py
```

---

## 🕹️ How to Play

1. Stand in front of the webcam.
2. Use your **index finger** to hover over the floating palette on the left or right side of the screen to equip a flower color.
3. Touch the circular sockets on your mandala grid to plant the flower.
4. Build a beautiful, symmetrical design to score points!
5. Watch the Live Judge HUD at the bottom to see Gemma's live commentary and the current score.

---
*Built with ❤️ for the Onam 2026 Physical AI Hackathon.*
