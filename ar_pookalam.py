import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import math
import random
import time
import json
import os
import threading
import urllib.request
import urllib.error


HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)


def ensure_hand_landmarker_model():
    """Download the MediaPipe hand landmarker model if not already present."""
    if os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        return
    print("Downloading MediaPipe hand landmarker model (~2 MB)...")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download hand landmarker model: {e}\n"
            f"Please manually download it from:\n  {HAND_LANDMARKER_MODEL_URL}\n"
            f"and place it at: {HAND_LANDMARKER_MODEL_PATH}"
        ) from e

# --- Traditional Kerala & Modern Floral Colors (BGR) ---
FLOWER_PALETTE = [
    {"name": "Marigold Gold",   "bgr": (0, 215, 255)},   # Bright Yellow / Gold
    {"name": "Saffron Orange",  "bgr": (0, 135, 255)},   # Vivid Saffron Orange
    {"name": "Hibiscus Red",    "bgr": (35, 35, 225)},   # Rich Crimson Red
    {"name": "Lotus Pink",      "bgr": (175, 95, 245)},  # Soft Rose Pink
    {"name": "Krishna Violet",  "bgr": (205, 75, 150)},  # Royal Lilac / Violet
    {"name": "Cyan Blossom",    "bgr": (245, 170, 30)},  # Vibrant Cyan / Blue
    {"name": "Mint Emerald",    "bgr": (130, 225, 90)},  # Fresh Mint Leaf Green
]

# Global Live Evaluation State (Thread-safe shared dictionary)
live_judge_state = {
    "p1_score": 0,
    "p2_score": 0,
    "leader": "TIE",
    "commentary": "Touch flower colors & fill the mandala circles to begin!",
    "eval_mode": "FALLBACK",  # "GEMMA_AI" or "FALLBACK"
    "active_model": "None (Offline)",
    "last_eval_time": 0,
    "is_evaluating": False,
    "eval_count": 0
}


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    """Draws clean anti-aliased rounded rectangles."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    radius = max(2, min(radius, w // 2, h // 2))
    
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_glass_card(frame, pt1, pt2, bg_color=(10, 14, 24), border_color=(65, 80, 105), alpha=0.82, radius=14):
    """Renders sleek glassmorphism panels."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = frame.copy()
    draw_rounded_rect(overlay, (x1, y1), (x2, y2), bg_color, radius=radius, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    draw_rounded_rect(frame, (x1, y1), (x2, y2), border_color, radius=radius, thickness=1)


def draw_styled_flower(frame, cx, cy, radius, primary_bgr, center_bgr=(0, 225, 255), is_selected=False):
    """
    Renders the beautiful 4-petal flower:
    - 4 overlapping smooth circular petals (Top, Left, Right, Bottom).
    - Multi-tonal shaded layers with clean anti-aliasing.
    - Central bright gold pistil with a white rim.
    """
    cx, cy, radius = int(cx), int(cy), int(radius)
    
    b, g, r = primary_bgr
    darker_bgr = (int(b * 0.72), int(g * 0.72), int(r * 0.72))
    lighter_bgr = (min(255, int(b * 1.18)), min(255, int(g * 1.18)), min(255, int(r * 1.18)))
    
    petal_r = int(radius * 0.58)
    offset = int(radius * 0.42)
    
    # 4 Petal Layers:
    # 1. Top Petal (Deeper base tone)
    cv2.circle(frame, (cx, cy - offset), petal_r, darker_bgr, -1, cv2.LINE_AA)
    # 2. Left & Right Petals (Primary tone)
    cv2.circle(frame, (cx - offset, cy), petal_r, primary_bgr, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + offset, cy), petal_r, primary_bgr, -1, cv2.LINE_AA)
    # 3. Bottom Petal (Lighter top tone)
    cv2.circle(frame, (cx, cy + offset), petal_r, lighter_bgr, -1, cv2.LINE_AA)
    
    # Crisp white outlines for definition
    cv2.circle(frame, (cx, cy - offset), petal_r, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx - offset, cy), petal_r, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx + offset, cy), petal_r, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy + offset), petal_r, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Center Pistil (Bright Gold bead with White Rim)
    center_r = max(4, int(radius * 0.28))
    cv2.circle(frame, (cx, cy), center_r + 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), center_r, center_bgr, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), center_r + 2, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Active Selection Beacon
    if is_selected:
        cv2.circle(frame, (cx, cy), radius + 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius + 10, primary_bgr, 1, cv2.LINE_AA)


class PookalamSocket:
    def __init__(self, player_id, x, y, radius, ring_idx, ring_pos_idx):
        self.player_id = player_id  # 1 (Left) or 2 (Right)
        self.x = int(x)
        self.y = int(y)
        self.radius = int(radius)
        self.ring_idx = ring_idx
        self.ring_pos_idx = ring_pos_idx
        self.color = None  # (B, G, R) when filled
        self.flower_name = None
        self.highlight_anim = 0

    def draw(self, frame):
        pos = (self.x, self.y)
        
        # Touch highlight pulse
        if self.highlight_anim > 0:
            burst_r = self.radius + int(self.highlight_anim * 3)
            cv2.circle(frame, pos, burst_r, (255, 255, 255), 1, cv2.LINE_AA)
            self.highlight_anim -= 1

        if self.color is not None:
            # Filled socket: Renders the 4-petal flower blossom!
            draw_styled_flower(frame, self.x, self.y, self.radius, self.color)
        else:
            # Empty socket: Translucent glass guide circle
            cv2.circle(frame, pos, self.radius, (25, 32, 48), -1, cv2.LINE_AA)
            cv2.circle(frame, pos, self.radius, (110, 130, 165), 1, cv2.LINE_AA)
            cv2.circle(frame, pos, 2, (170, 190, 220), -1, cv2.LINE_AA)


class PaletteFlower:
    def __init__(self, slot_id, player_id, x, y, radius, color, name=""):
        self.slot_id = slot_id
        self.player_id = player_id
        self.x = int(x)
        self.y = int(y)
        self.radius = int(radius)
        self.color = color
        self.name = name
        self.is_selected = False
        self.touch_anim = 0

    def draw(self, frame):
        # Touch ripple
        if self.touch_anim > 0:
            cv2.circle(frame, (self.x, self.y), self.radius + self.touch_anim * 3, (255, 255, 255), 1, cv2.LINE_AA)
            self.touch_anim -= 1

        # Render styled 4-petal flower
        draw_styled_flower(frame, self.x, self.y, self.radius, self.color, is_selected=self.is_selected)


def build_sockets(center_x, center_y, player_id, scale=1.0):
    """
    Builds spacious snap sockets scaled to display resolution.
    1 Center + 6 Ring1 + 10 Ring2 + 14 Ring3 = 31 total sockets.
    """
    sockets = []
    
    # 0. Center Core Socket
    sockets.append(PookalamSocket(player_id, center_x, center_y, radius=int(30 * scale), ring_idx=0, ring_pos_idx=0))
    
    # Ring configuration: (base_distance, count, base_socket_radius)
    rings = [
        (72, 6, 25),   # Inner Petal Ring
        (138, 10, 25), # Middle Petal Ring
        (204, 14, 25)  # Outer Petal Ring
    ]
    
    for ring_idx, (base_dist, count, base_rad) in enumerate(rings, start=1):
        dist = int(base_dist * scale)
        s_rad = int(base_rad * scale)
        for i in range(count):
            angle = i * (2 * math.pi / count)
            sx = center_x + dist * math.cos(angle)
            sy = center_y + dist * math.sin(angle)
            sockets.append(PookalamSocket(player_id, sx, sy, radius=s_rad, ring_idx=ring_idx, ring_pos_idx=i))
            
    return sockets


def create_ui_overlay(width, height, p1_center, p2_center, scale=1.0):
    """Pre-renders clean, elegant geometric guide lines scaled to resolution."""
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    center_x = width // 2
    
    def draw_mandala_guides(cx, cy, color):
        for base_dist in [72, 138, 204]:
            cv2.circle(overlay, (cx, cy), int(base_dist * scale), color, 1, cv2.LINE_AA)
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            x1 = int(cx + 35 * scale * math.cos(angle))
            y1 = int(cy + 35 * scale * math.sin(angle))
            x2 = int(cx + 225 * scale * math.cos(angle))
            y2 = int(cy + 225 * scale * math.sin(angle))
            cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Subtle center divider line (avoiding top bar and bottom HUD)
    top_margin = int(60 * scale)
    bottom_margin = int(80 * scale)
    cv2.line(overlay, (center_x, top_margin), (center_x, height - bottom_margin), (55, 70, 95), 1, cv2.LINE_AA)
    
    # Left & Right Mandala Guides
    draw_mandala_guides(p1_center[0], p1_center[1], (70, 95, 130))
    draw_mandala_guides(p2_center[0], p2_center[1], (95, 80, 135))

    return overlay


def create_side_palette(width, height, scale=1.0, content_top=100, content_bottom=None):
    """Generates vertical flower palette on left and right edges, scaled to resolution."""
    dots = []
    num_colors = len(FLOWER_PALETTE)
    if content_bottom is None:
        content_bottom = height - int(100 * scale)
    
    # Leave room for "FLOWERS" header above and clear button below
    flower_top = content_top + int(28 * scale)
    flower_bottom = content_bottom - int(8 * scale)
    spacing = max(1, (flower_bottom - flower_top) // max(1, num_colors - 1))
    
    radius = int(20 * scale)
    dock_cx_left = int(55 * scale)
    dock_cx_right = width - int(55 * scale)
    
    for i, f in enumerate(FLOWER_PALETTE):
        y = flower_top + i * spacing
        dots.append(PaletteFlower(f"P1_{i}", 1, dock_cx_left, y, radius, f["bgr"], f["name"]))
        dots.append(PaletteFlower(f"P2_{i}", 2, dock_cx_right, y, radius, f["bgr"], f["name"]))

    return dots


def calculate_metrics(sockets, pid):
    """Computes stats for a single player's Pookalam."""
    p_sockets = [s for s in sockets if s.player_id == pid]
    total = len(p_sockets)
    filled = [s for s in p_sockets if s.color is not None]
    filled_count = len(filled)
    completeness = int((filled_count / total) * 100)

    colors_used = set(s.flower_name for s in filled if s.flower_name)
    color_variety_score = min(100, int((len(colors_used) / 4) * 100))

    symmetry_points = 0
    for r_idx in range(4):
        ring_s = [s for s in p_sockets if s.ring_idx == r_idx]
        ring_colors = [s.flower_name for s in ring_s if s.flower_name]
        if ring_colors and len(ring_colors) == len(ring_s):
            most_common_count = max(ring_colors.count(c) for c in set(ring_colors))
            symmetry_points += (most_common_count / len(ring_s)) * 25

    symmetry_score = int(symmetry_points)
    final_score = int(completeness * 0.4 + symmetry_score * 0.35 + color_variety_score * 0.25)
    
    return {
        "filled": filled_count,
        "total": total,
        "completeness": completeness,
        "symmetry": symmetry_score,
        "variety": len(colors_used),
        "colors": list(colors_used),
        "score": final_score
    }


def get_available_ollama_models():
    """Checks Ollama for currently installed models."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = [m.get("name", "") for m in data.get("models", [])]
            return models
    except Exception:
        return []


def sanitize_text(text):
    """Strips quotes, question marks, and non-ASCII characters for clean OpenCV rendering."""
    if not text:
        return ""
    # Strip quotes, backticks, question marks from start/end
    cleaned = text.strip(' "\'`“”‘’')
    # Filter only printable ASCII characters
    cleaned = "".join(c for c in cleaned if 32 <= ord(c) <= 126)
    return " ".join(cleaned.split())


def query_ollama_judge(p1_stats, p2_stats):
    """
    Asynchronously queries local Ollama instance with structured reasoning prompt.
    Returns (commentary_string, eval_mode, active_model).
    """
    installed_models = get_available_ollama_models()
    preferred_models = ["gemma3:4b-it-qat", "gemma3:4b", "gemma3:1b", "gemma3", "gemma2:2b", "gemma:2b", "gemma"]
    
    selected_model = None
    if installed_models:
        for pm in preferred_models:
            for im in installed_models:
                if pm in im:
                    selected_model = im
                    break
            if selected_model:
                break
        if not selected_model:
            selected_model = installed_models[0]

    if selected_model:
        p1_color_str = ", ".join(p1_stats['colors']) if p1_stats['colors'] else "None"
        p2_color_str = ", ".join(p2_stats['colors']) if p2_stats['colors'] else "None"
        
        prompt = (
            f"You are the master grand judge of the Kerala Onam Pookalam flower contest.\n"
            f"Evaluate and compare both artisan creations:\n"
            f"- Player 1: {p1_stats['filled']}/31 flowers placed, {p1_stats['symmetry']}% symmetry, "
            f"Palette: {p1_color_str}, Total Score: {p1_stats['score']}/100.\n"
            f"- Player 2: {p2_stats['filled']}/31 flowers placed, {p2_stats['symmetry']}% symmetry, "
            f"Palette: {p2_color_str}, Total Score: {p2_stats['score']}/100.\n"
            f"In exactly 1 concise sentence (under 16 words, plain text without quotes), declare who is winning, give their key artistic strength, and give 1 quick advice to the other."
        )
        try:
            req_data = json.dumps({
                "model": selected_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 1024,
                    "num_predict": 25,
                    "temperature": 0.5,
                    "num_thread": 3
                }
            }).encode('utf-8')

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=12.0) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                raw_text = result.get("response", "").strip()
                clean_text = sanitize_text(raw_text)
                if len(clean_text) > 0:
                    return clean_text[:110], "GEMMA_AI", selected_model
        except Exception:
            pass

    # --- Fallback Engine if Ollama is Offline ---
    if p1_stats['score'] == 0 and p2_stats['score'] == 0:
        commentary = "Touch flower colors and fill circles to begin crafting your Pookalams!"
    elif p1_stats['score'] > p2_stats['score']:
        lead_diff = p1_stats['score'] - p2_stats['score']
        commentary = f"Player 1 leads (+{lead_diff} pts) with {p1_stats['symmetry']}% symmetry! Player 2, add contrasting petals."
    elif p2_stats['score'] > p1_stats['score']:
        lead_diff = p2_stats['score'] - p1_stats['score']
        commentary = f"Player 2 leads (+{lead_diff} pts) with {p2_stats['symmetry']}% symmetry! Player 1, complete the outer ring."
    else:
        commentary = f"Both tied at {p1_stats['score']} pts! Both artisans showing balanced Onam geometry."

    return sanitize_text(commentary), "FALLBACK", "Offline (Rule Engine)"


def live_judge_worker(get_sockets_callback):
    """Background daemon thread that runs continuous judging every 5 seconds."""
    time.sleep(0.5)
    
    while True:
        try:
            sockets_snapshot = get_sockets_callback()
            if sockets_snapshot:
                p1_stats = calculate_metrics(sockets_snapshot, 1)
                p2_stats = calculate_metrics(sockets_snapshot, 2)

                live_judge_state["is_evaluating"] = True
                commentary, eval_mode, model_tag = query_ollama_judge(p1_stats, p2_stats)

                # Determine Leader
                if p1_stats['score'] > p2_stats['score']:
                    leader = "PLAYER 1"
                elif p2_stats['score'] > p1_stats['score']:
                    leader = "PLAYER 2"
                else:
                    leader = "TIE"

                # Update shared state
                live_judge_state["p1_score"] = p1_stats['score']
                live_judge_state["p2_score"] = p2_stats['score']
                live_judge_state["leader"] = leader
                live_judge_state["commentary"] = commentary
                live_judge_state["eval_mode"] = eval_mode
                live_judge_state["active_model"] = model_tag
                live_judge_state["last_eval_time"] = time.time()
                live_judge_state["eval_count"] += 1
                live_judge_state["is_evaluating"] = False

        except Exception:
            live_judge_state["is_evaluating"] = False

        time.sleep(5.0)


def main():
    ensure_hand_landmarker_model()
    base_options = mp_tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
    landmarker_options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    hands = mp_vision.HandLandmarker.create_from_options(landmarker_options)
    video_timestamp_ms = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Request highest available camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Camera Warmup Retry Loop
    frame = None
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        print("Error: Could not capture initial frame after camera warmup.")
        return

    cam_h, cam_w = frame.shape[:2]
    print(f"Camera native resolution: {cam_w}x{cam_h}")

    # Ensure minimum display size while maintaining camera aspect ratio
    if cam_w < 1280 or cam_h < 720:
        upscale = max(1280.0 / cam_w, 720.0 / cam_h)
        display_w = int(cam_w * upscale)
        display_h = int(cam_h * upscale)
    else:
        display_w, display_h = cam_w, cam_h

    width, height = display_w, display_h
    print(f"Display resolution: {width}x{height}")

    # Scale factor normalised to 1280x720 baseline
    scale = min(width / 1280.0, height / 720.0)

    # --- ADAPTIVE WINDOW SETUP ---
    cv2.namedWindow('AR Pookalam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AR Pookalam', width, height)

    # --- LAYOUT ZONES (scale-relative, non-overlapping) ---
    center_x = width // 2
    top_bar_h = int(55 * scale)
    bottom_hud_h = int(75 * scale)
    dock_w = int(115 * scale)

    # Play area boundaries
    play_left = dock_w
    play_right = width - dock_w
    play_top = top_bar_h
    play_bottom = height - bottom_hud_h
    play_w = play_right - play_left
    play_h = play_bottom - play_top

    # Pookalam scale: ensures mandala fits within available play area
    max_pookalam_r = 229  # outer_ring_dist(204) + socket_radius(25) at base scale
    pookalam_scale = min(scale,
                         play_w / (4.0 * max_pookalam_r) * 0.88,
                         play_h / (2.0 * max_pookalam_r) * 0.88)

    # Pookalam centers (centered in each half of play area)
    p1_center = (play_left + play_w // 4, (play_top + play_bottom) // 2)
    p2_center = (play_left + 3 * play_w // 4, (play_top + play_bottom) // 2)

    # Side dock rects
    dock_pad = int(8 * scale)
    dock_content_top = top_bar_h + dock_pad
    dock_content_bottom = height - bottom_hud_h - dock_pad
    left_dock_rect = (dock_pad, dock_content_top, dock_w - dock_pad, dock_content_bottom)
    right_dock_rect = (width - dock_w + dock_pad, dock_content_top, width - dock_pad, dock_content_bottom)

    # Clear buttons at dock bottoms
    clear_btn_h = int(28 * scale)
    p1_clear_rect = (left_dock_rect[0] + 4, left_dock_rect[3] - clear_btn_h - 4,
                     left_dock_rect[2] - 4, left_dock_rect[3] - 4)
    p2_clear_rect = (right_dock_rect[0] + 4, right_dock_rect[3] - clear_btn_h - 4,
                     right_dock_rect[2] - 4, right_dock_rect[3] - 4)

    # Bottom Judge HUD (commentary only, no scores)
    hud_w = min(int(850 * scale), width - int(20 * scale))
    judge_hud_rect = (center_x - hud_w // 2, height - bottom_hud_h,
                      center_x + hud_w // 2, height - int(6 * scale))

    # 1. Pre-render Geometric Guide Overlay
    ui_overlay = create_ui_overlay(width, height, p1_center, p2_center, pookalam_scale)

    # 2. Build Scaled Sockets
    sockets = build_sockets(p1_center[0], p1_center[1], player_id=1, scale=pookalam_scale) + \
              build_sockets(p2_center[0], p2_center[1], player_id=2, scale=pookalam_scale)

    # 3. Create Scaled Palette Flowers
    palette_dots = create_side_palette(width, height, scale,
                                       content_top=dock_content_top,
                                       content_bottom=p1_clear_rect[1] - int(8 * scale))

    # Equipped Color per hand index
    hand_equipped_color = {
        0: {"color": FLOWER_PALETTE[0]["bgr"], "name": FLOWER_PALETTE[0]["name"]},
        1: {"color": FLOWER_PALETTE[1]["bgr"], "name": FLOWER_PALETTE[1]["name"]}
    }

    # Start Asynchronous Live Gemma Judge Background Thread
    judge_thread = threading.Thread(
        target=live_judge_worker,
        args=(lambda: sockets,),
        daemon=True
    )
    judge_thread.start()

    print("AR Pookalam High-Readability Flower Edition Running!")
    print("👉 Touch flowers to equip colors, touch circles to paint!")
    print("Press 'q' in the game window to quit.")

    prev_time = 0
    fps = 30
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Horizontal Mirror + Scale to display resolution
        frame = cv2.flip(frame, 1)
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # MediaPipe Hands Detection (Tasks API)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        video_timestamp_ms += int(1000 / 30)
        results = hands.detect_for_video(mp_image, video_timestamp_ms)

        # Reset selection outlines
        for dot in palette_dots:
            dot.is_selected = False

        if results.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                if hand_idx not in hand_equipped_color:
                    hand_equipped_color[hand_idx] = {
                        "color": FLOWER_PALETTE[0]["bgr"],
                        "name": FLOWER_PALETTE[0]["name"]
                    }

                curr_equipped = hand_equipped_color[hand_idx]

                # Index Fingertip is the Touch Pointer (Landmark 8)
                index_tip = hand_landmarks[8]
                ix = int(index_tip.x * width)
                iy = int(index_tip.y * height)

                index_mcp = hand_landmarks[5]
                mx = int(index_mcp.x * width)
                my = int(index_mcp.y * height)

                # --- 1. TOUCH DOCK: Equip / Select Color ---
                touch_r = int(40 * scale)
                for dot in palette_dots:
                    dist_to_dock = math.hypot(ix - dot.x, iy - dot.y)
                    if dist_to_dock <= touch_r:
                        hand_equipped_color[hand_idx] = {"color": dot.color, "name": dot.name}
                        curr_equipped = hand_equipped_color[hand_idx]
                        dot.touch_anim = 4

                    if dot.color == curr_equipped["color"]:
                        dot.is_selected = True

                # --- 2. TOUCH SOCKET: Fill / Paint Socket ---
                for sock in sockets:
                    dist_to_socket = math.hypot(ix - sock.x, iy - sock.y)
                    if dist_to_socket <= sock.radius + int(12 * scale):
                        if sock.color != curr_equipped["color"]:
                            sock.color = curr_equipped["color"]
                            sock.flower_name = curr_equipped["name"]
                            sock.highlight_anim = 3

                # --- 3. TOUCH CLEAR BUTTONS ---
                if p1_clear_rect[0] <= ix <= p1_clear_rect[2] and p1_clear_rect[1] <= iy <= p1_clear_rect[3]:
                    for sock in sockets:
                        if sock.player_id == 1:
                            sock.color = None
                            sock.flower_name = None
                elif p2_clear_rect[0] <= ix <= p2_clear_rect[2] and p2_clear_rect[1] <= iy <= p2_clear_rect[3]:
                    for sock in sockets:
                        if sock.player_id == 2:
                            sock.color = None
                            sock.flower_name = None

                # --- 4. RENDER MAGIC WAND POINTER (Flower Cursor) ---
                brush_color = curr_equipped["color"]
                cv2.line(frame, (mx, my), (ix, iy), (220, 235, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (mx, my), int(4 * scale), (180, 200, 230), -1, cv2.LINE_AA)

                # Render active miniature flower cursor at fingertip
                draw_styled_flower(frame, ix, iy, int(15 * scale), brush_color)

                # Floating Brush Pill Tag
                pill_w = int(105 * scale)
                pill_h = int(19 * scale)
                badge_x = max(int(10 * scale), ix - pill_w // 2)
                badge_y = max(int(25 * scale), iy - int(26 * scale))
                draw_rounded_rect(frame, (badge_x, badge_y - pill_h + 4), (badge_x + pill_w, badge_y + 4),
                                  (12, 16, 26), radius=int(6 * scale), thickness=-1)
                draw_rounded_rect(frame, (badge_x, badge_y - pill_h + 4), (badge_x + pill_w, badge_y + 4),
                                  brush_color, radius=int(6 * scale), thickness=1)
                brush_font = max(0.28, 0.38 * scale)
                cv2.putText(frame, f"{curr_equipped['name'][:8]}", (badge_x + int(8 * scale), badge_y - int(3 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, brush_font, (245, 245, 255), 1, cv2.LINE_AA)

        # --- COMPOSITING & MODERNISED UI RENDERING ---
        
        # 1. Blend Subtle Geometric Mandala Guides
        frame = cv2.addWeighted(frame, 1.0, ui_overlay, 0.65, 0)

        # 2. Draw Sockets (Renders 4-Petal Flowers when filled)
        for sock in sockets:
            sock.draw(frame)

        # 3. Translucent Left & Right Palette Docks
        draw_glass_card(frame, (left_dock_rect[0], left_dock_rect[1]), (left_dock_rect[2], left_dock_rect[3]),
                        bg_color=(10, 14, 24), border_color=(75, 95, 125), alpha=0.22, radius=int(16 * scale))
        draw_glass_card(frame, (right_dock_rect[0], right_dock_rect[1]), (right_dock_rect[2], right_dock_rect[3]),
                        bg_color=(10, 14, 24), border_color=(75, 95, 125), alpha=0.22, radius=int(16 * scale))

        # Dock Header Labels
        dock_hdr_font = max(0.3, 0.4 * scale)
        cv2.putText(frame, "FLOWERS", (left_dock_rect[0] + int(6 * scale), left_dock_rect[1] + int(18 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, dock_hdr_font, (200, 215, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "FLOWERS", (right_dock_rect[0] + int(6 * scale), right_dock_rect[1] + int(18 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, dock_hdr_font, (200, 215, 240), 1, cv2.LINE_AA)

        # Draw Palette Flowers
        for dot in palette_dots:
            dot.draw(frame)

        # Clear Buttons
        draw_glass_card(frame, (p1_clear_rect[0], p1_clear_rect[1]), (p1_clear_rect[2], p1_clear_rect[3]),
                        bg_color=(45, 18, 24), border_color=(190, 60, 75), alpha=0.45, radius=int(8 * scale))
        draw_glass_card(frame, (p2_clear_rect[0], p2_clear_rect[1]), (p2_clear_rect[2], p2_clear_rect[3]),
                        bg_color=(45, 18, 24), border_color=(190, 60, 75), alpha=0.45, radius=int(8 * scale))
        clear_font = max(0.28, 0.38 * scale)
        for cr in [p1_clear_rect, p2_clear_rect]:
            (tw, th), _ = cv2.getTextSize("CLEAR", cv2.FONT_HERSHEY_SIMPLEX, clear_font, 1)
            ctx = cr[0] + ((cr[2] - cr[0]) - tw) // 2
            cty = cr[1] + ((cr[3] - cr[1]) + th) // 2
            cv2.putText(frame, "CLEAR", (ctx, cty),
                        cv2.FONT_HERSHEY_SIMPLEX, clear_font, (255, 195, 205), 1, cv2.LINE_AA)

        # 4. TOP SCOREBOARD (Scores prominently on top for each player)
        p1_s = live_judge_state["p1_score"]
        p2_s = live_judge_state["p2_score"]
        leader_text = live_judge_state["leader"]

        card_h = int(42 * scale)
        card_y1 = int(7 * scale)
        card_y2 = card_y1 + card_h
        card_w = int(250 * scale)

        # Player 1 Scoreboard (cyan accent)
        p1_card_x1 = p1_center[0] - card_w // 2
        p1_card_x2 = p1_center[0] + card_w // 2
        p1_leading = leader_text == "PLAYER 1"
        p1_border = (0, 230, 255) if p1_leading else (0, 120, 165)
        draw_glass_card(frame, (p1_card_x1, card_y1), (p1_card_x2, card_y2),
                        bg_color=(8, 14, 28), border_color=p1_border, alpha=0.88, radius=int(10 * scale))
        p_label_font = max(0.28, 0.34 * scale)
        p_score_font = max(0.38, 0.56 * scale)
        cv2.putText(frame, "PLAYER 1", (p1_card_x1 + int(14 * scale), card_y1 + int(17 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, p_label_font, (0, 200, 240), 1, cv2.LINE_AA)
        p1_sc_color = (0, 255, 255) if p1_leading else (180, 200, 225)
        cv2.putText(frame, f"{p1_s} PTS", (p1_card_x1 + int(14 * scale), card_y2 - int(8 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, p_score_font, p1_sc_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "GOLDEN", (p1_card_x2 - int(80 * scale), card_y2 - int(8 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.25, 0.32 * scale), (110, 140, 180), 1, cv2.LINE_AA)

        # Player 2 Scoreboard (magenta accent)
        p2_card_x1 = p2_center[0] - card_w // 2
        p2_card_x2 = p2_center[0] + card_w // 2
        p2_leading = leader_text == "PLAYER 2"
        p2_border = (205, 90, 255) if p2_leading else (130, 55, 165)
        draw_glass_card(frame, (p2_card_x1, card_y1), (p2_card_x2, card_y2),
                        bg_color=(8, 14, 28), border_color=p2_border, alpha=0.88, radius=int(10 * scale))
        cv2.putText(frame, "PLAYER 2", (p2_card_x1 + int(14 * scale), card_y1 + int(17 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, p_label_font, (210, 100, 245), 1, cv2.LINE_AA)
        p2_sc_color = (255, 140, 255) if p2_leading else (180, 200, 225)
        cv2.putText(frame, f"{p2_s} PTS", (p2_card_x1 + int(14 * scale), card_y2 - int(8 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, p_score_font, p2_sc_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "ROYAL", (p2_card_x2 - int(70 * scale), card_y2 - int(8 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.25, 0.32 * scale), (110, 140, 180), 1, cv2.LINE_AA)

        # Center Leader Pill
        leader_pill_w = int(120 * scale)
        leader_pill_h = int(26 * scale)
        lp_x1 = center_x - leader_pill_w // 2
        lp_x2 = center_x + leader_pill_w // 2
        lp_y1 = card_y1 + (card_h - leader_pill_h) // 2
        lp_y2 = lp_y1 + leader_pill_h
        leader_str = leader_text if leader_text != "TIE" else "TIED"
        draw_glass_card(frame, (lp_x1, lp_y1), (lp_x2, lp_y2),
                        bg_color=(18, 32, 52), border_color=(90, 150, 215), alpha=0.9, radius=int(8 * scale))
        leader_font = max(0.28, 0.4 * scale)
        (tw, th), _ = cv2.getTextSize(leader_str, cv2.FONT_HERSHEY_DUPLEX, leader_font, 1)
        cv2.putText(frame, leader_str, (center_x - tw // 2, lp_y1 + (leader_pill_h + th) // 2),
                    cv2.FONT_HERSHEY_DUPLEX, leader_font, (120, 240, 255), 1, cv2.LINE_AA)

        # FPS (top-right corner, subtle)
        curr_time = time.time()
        actual_fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        fps_font = max(0.28, 0.36 * scale)
        cv2.putText(frame, f"FPS:{int(actual_fps)}", (width - int(85 * scale), int(18 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fps_font, (100, 180, 120), 1, cv2.LINE_AA)

        # 5. BOTTOM JUDGE COMMENTARY HUD (commentary only)
        jx1, jy1, jx2, jy2 = judge_hud_rect
        draw_glass_card(frame, (jx1, jy1), (jx2, jy2),
                        bg_color=(6, 8, 16), border_color=(85, 115, 155), alpha=0.94, radius=int(14 * scale))

        eval_mode = live_judge_state["eval_mode"]
        model_name = live_judge_state["active_model"]
        commentary = sanitize_text(live_judge_state["commentary"])

        hud_mid_y = (jy1 + jy2) // 2

        # AI status pill
        if eval_mode == "GEMMA_AI":
            status_tag = f"AI: {model_name}"
            tag_color = (0, 255, 120)
            tag_bg = (10, 48, 24)
        else:
            status_tag = "AI: OFFLINE"
            tag_color = (0, 200, 255)
            tag_bg = (48, 30, 10)

        tag_font = max(0.25, 0.34 * scale)
        (stw, sth), _ = cv2.getTextSize(status_tag[:22], cv2.FONT_HERSHEY_DUPLEX, tag_font, 1)
        ai_pill_w = stw + int(14 * scale)
        ai_pill_h = sth + int(12 * scale)
        pill_pad = int(10 * scale)
        pill_x1 = jx1 + pill_pad
        pill_y1 = hud_mid_y - ai_pill_h // 2
        pill_y2 = pill_y1 + ai_pill_h
        draw_rounded_rect(frame, (pill_x1, pill_y1), (pill_x1 + ai_pill_w, pill_y2),
                          tag_bg, radius=int(6 * scale), thickness=-1)
        draw_rounded_rect(frame, (pill_x1, pill_y1), (pill_x1 + ai_pill_w, pill_y2),
                          tag_color, radius=int(6 * scale), thickness=1)
        cv2.putText(frame, status_tag[:22], (pill_x1 + int(7 * scale), hud_mid_y + sth // 2),
                    cv2.FONT_HERSHEY_DUPLEX, tag_font, tag_color, 1, cv2.LINE_AA)

        # Judge commentary (right of AI pill)
        judge_font = max(0.3, 0.42 * scale)
        comment_font = max(0.28, 0.4 * scale)
        comment_x = pill_x1 + ai_pill_w + int(14 * scale)
        cv2.putText(frame, "JUDGE:", (comment_x, hud_mid_y + sth // 2),
                    cv2.FONT_HERSHEY_DUPLEX, judge_font, (120, 240, 255), 1, cv2.LINE_AA)
        (jw, _), _ = cv2.getTextSize("JUDGE:", cv2.FONT_HERSHEY_DUPLEX, judge_font, 1)
        max_chars = max(30, int(70 * scale))
        cv2.putText(frame, commentary[:max_chars], (comment_x + jw + int(8 * scale), hud_mid_y + sth // 2),
                    cv2.FONT_HERSHEY_DUPLEX, comment_font, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('AR Pookalam', frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()  # HandLandmarker cleanup

if __name__ == "__main__":
    main()