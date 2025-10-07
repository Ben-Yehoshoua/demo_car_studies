# app.py — mode MANUEL : capture sur bouton, puis OCR → set_speed_limit
from flask import Flask, render_template, request, jsonify, Response
import cv2
import atexit
import time
from threading import Lock, RLock
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import numpy as np
import base64
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union
from datetime import datetime

# ========= Configuration générale ============================================
DEBUG = False

# Port série pour Arduino (désactivé si DEBUG=True)
if DEBUG:
    ser = None
else:
    import serial
    ser = serial.Serial(
        port='/dev/serial0',   # /dev/ttyAMA0 ou /dev/ttyS0 selon modèle
        baudrate=9600,
        timeout=1
    )

def envoyer_message(message: str):
    """Envoie une commande simple à l’Arduino via série."""
    if DEBUG:
        print(f"[SERIAL:DEBUG] {message}")
    else:
        ser.write((message + '\n').encode('utf-8'))
        print(f"Envoyé à l'Arduino : {message}")

# --- Détection panneau (seulement utilisée à la demande) ---------------------
# Plages HSV « rouge » + heuristique « cœur blanc » (panneau vitesse)
_LOWER_RED_1 = np.array([0,   40, 40], np.uint8)
_UPPER_RED_1 = np.array([12, 255,255], np.uint8)
_LOWER_RED_2 = np.array([168, 40, 40], np.uint8)
_UPPER_RED_2 = np.array([180,255,255], np.uint8)
_SAT_MAX_WHITE = 80
_VAL_MIN_WHITE = 140
_MIN_AREA = 500
_MIN_PERIM = 80
_MIN_CIRCULARITY = 0.35

def _circularity(area: float, perim: float) -> float:
    if perim <= 1e-6:
        return 0.0
    return 4.0 * np.pi * (area / (perim * perim))

def detect_speed_limit_candidates(frame_bgr: np.ndarray):
    """
    Détecte des candidats de panneaux de limitation de vitesse (anneau rouge + cœur blanc).
    Retourne une liste de bounding boxes [(x, y, w, h), ...].
    """
    try:
        img = cv2.bilateralFilter(frame_bgr, d=7, sigmaColor=60, sigmaSpace=60)
        # Unsharp mask léger
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        img = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Masque rouge
        mask1 = cv2.inRange(hsv, _LOWER_RED_1, _UPPER_RED_1)
        mask2 = cv2.inRange(hsv, _LOWER_RED_2, _UPPER_RED_2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((7, 7), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < _MIN_AREA:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < _MIN_PERIM:
                continue
            if _circularity(area, perim) < _MIN_CIRCULARITY:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < 14:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Vérifie un « cœur blanc »
            h_img, w_img = hsv.shape[:2]
            inner_radius = max(int(radius * 0.6), 8)
            x0 = max(int(cx - inner_radius), 0)
            y0 = max(int(cy - inner_radius), 0)
            x1 = min(int(cx + inner_radius), w_img - 1)
            y1 = min(int(cy + inner_radius), h_img - 1)
            if x1 <= x0 or y1 <= y0:
                continue

            roi_hsv = hsv[y0:y1, x0:x1]
            mask_inner = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.circle(mask_inner, (int(cx) - x0, int(cy) - y0), inner_radius, 255, -1)

            sat = roi_hsv[:, :, 1]
            val = roi_hsv[:, :, 2]
            inner_pixels = mask_inner > 0
            if inner_pixels.sum() == 0:
                continue
            mean_sat = float(sat[inner_pixels].mean())
            mean_val = float(val[inner_pixels].mean())
            if not (mean_sat <= _SAT_MAX_WHITE and mean_val >= _VAL_MIN_WHITE):
                continue

            boxes.append((x, y, w, h))

        # Fallback Hough si vide (souvent utile en faible lumière)
        if not boxes:
            v = hsv[:, :, 2]
            v = cv2.GaussianBlur(v, (9,9), 2)
            circles = cv2.HoughCircles(v, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                                       param1=80, param2=25, minRadius=14, maxRadius=180)
            if circles is not None:
                for (cx, cy, r) in np.round(circles[0, :]).astype(int):
                    y0, y1 = max(0, cy-r), min(hsv.shape[0], cy+r)
                    x0, x1 = max(0, cx-r), min(hsv.shape[1], cx+r)
                    roi = frame_bgr[y0:y1, x0:x1]
                    if roi.size == 0:
                        continue
                    b, g, rch = cv2.split(roi)
                    if float(rch.mean()) > 1.15*float(g.mean()) and float(rch.mean()) > 1.15*float(b.mean()):
                        boxes.append((x0, y0, x1-x0, y1-y0))
        return boxes
    except Exception:
        return []

# ========= OCR ===============================================================
# Dossier snapshots
SNAP_DIR = os.path.join(os.getcwd(), "snapshots_speed_sign")
os.makedirs(SNAP_DIR, exist_ok=True)

# Exécuteur OCR
_OCR_EXEC = ThreadPoolExecutor(max_workers=1)

# Charge variables d’environnement (.env)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _quick_compress_to_b64_jpeg(img_bgr: np.ndarray, max_side: int = 320) -> Optional[str]:
    """Convertit un BGR en JPEG base64 (data URL), en niveaux de gris et réduit."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ok, buff = cv2.imencode(".jpg", gray3, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return None
        b64 = base64.b64encode(buff.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None

def _ocr_number_from_data_url(data_url: str) -> Optional[int]:
    """
    Envoie l’image (data URL) au modèle vision et extrait un entier 1–3 chiffres.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Lis UNIQUEMENT le nombre sur le panneau de limitation de vitesse."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=8,
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\b(\d{1,3})\b", txt)
        if not m:
            return None
        return int(m.group(1))
    except Exception as e:
        if DEBUG: print(f"[OCR] Erreur OCR: {e}")
        return None

# ========= Caméra (flux) =====================================================
CAM_INDEX_CANDIDATES = [0, 1]
WIDTH, HEIGHT, FPS = 640, 480, 30
cap = None
cap_lock = Lock()

# Dernier frame disponible pour capture manuelle
_last_frame = None
_last_frame_lock = Lock()

def open_camera():
    """Ouvre la caméra avec MJPG et applique WIDTH/HEIGHT/FPS."""
    for idx in CAM_INDEX_CANDIDATES:
        c = cv2.VideoCapture(idx)
        if not c.isOpened():
            c.release(); continue
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        c.set(cv2.CAP_PROP_FPS, FPS)
        time.sleep(0.2)
        ok, _ = c.read()
        if ok: return c
        c.release()
    return None

def ensure_camera():
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = open_camera()
    return cap

def release_camera():
    global cap
    with cap_lock:
        if cap is not None:
            try: cap.release()
            except Exception: pass
            cap = None

atexit.register(release_camera)

def generate_frames():
    """
    Génère le flux JPEG multipart pour /video_feed.
    AUCUNE détection/ocr ici → mode MANUEL.
    Stocke simplement le dernier frame pour /capture_speed_sign.
    """
    global _last_frame  # ← AJOUTER CETTE LIGNE

    if ensure_camera() is None:
        blank = (np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        ok, buffer = cv2.imencode('.jpg', blank)
        jpg = buffer.tobytes() if ok else b''
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        return

    while True:
        with cap_lock:
            if cap is None or not cap.isOpened():
                break
            success, frame = cap.read()
        if not success:
            release_camera()
            if ensure_camera() is None:
                break
            continue

        # Mémorise ce frame pour la capture manuelle
        with _last_frame_lock:
            # on stocke une copie compacte (pour éviter mutation)
            _last_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        jpg = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

# ========= État vitesse limite + API =========================================
_speed_limit_value: Optional[float] = None
_speed_limit_lock = RLock()

def set_speed_limit(value: Optional[Union[int, float]]):
    """
    Met à jour la vitesse limite côté serveur (value=None pour effacer).
    """
    global _speed_limit_value
    with _speed_limit_lock:
        if value is None:
            _speed_limit_value = None
        else:
            try:
                v = float(value)
                if v < 0: v = 0
                if v > 200: v = 200
                _speed_limit_value = v
            except Exception:
                pass

# ========= Flask app / Routes ================================================
app = Flask(__name__)

@app.route('/speed_limit', methods=['GET', 'POST'])
def speed_limit():
    """
    GET  -> { "speed_limit": 50 | null }
    POST -> body { "value": 50 } (ou { "speed_limit": 50 }) → met à jour.
    """
    global _speed_limit_value
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        value = data.get('value', data.get('speed_limit', None))
        set_speed_limit(value)
        with _speed_limit_lock:
            return jsonify({"ok": True, "speed_limit": _speed_limit_value}), 200
    with _speed_limit_lock:
        return jsonify({"speed_limit": _speed_limit_value}), 200

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('simulator.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    action = request.json.get('action')
    envoyer_message(action)
    print(f"Commande envoyée: {action}")
    return jsonify({"status": "success", "action": action})

# --------- NOUVEAU : capture manuelle + OCR ---------------------------------
@app.route('/capture_speed_sign', methods=['POST'])
def capture_speed_sign():
    # Récupère un snapshot courant
    with _last_frame_lock:
        frame = _last_frame.copy() if '_last_frame' in globals() and _last_frame is not None else None

    # 🔧 Fallback: on tente de lire directement la caméra si pas de frame en mémoire
    if frame is None:
        if ensure_camera() is not None:
            for _ in range(3):
                with cap_lock:
                    ok, frm = cap.read()
                if ok and frm is not None:
                    frame = frm
                    break
        if frame is None:
            return jsonify({"ok": False, "error": "Aucun frame disponible"}), 503

    # Essaie de trouver un panneau pour croper (sinon, image complète)
    boxes = detect_speed_limit_candidates(frame)
    used_crop = False
    roi = frame
    if boxes:
        # Choisit le plus grand (souvent le plus fiable)
        bx = max(boxes, key=lambda b: b[2]*b[3])
        x, y, w, h = bx
        h_img, w_img = frame.shape[:2]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(w_img, x + w); y1 = min(h_img, y + h)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1].copy()
            used_crop = True

    # Sauvegarde (utile pour debug)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(SNAP_DIR, f"capture_{ts}{'_crop' if used_crop else ''}.jpg")
    cv2.imwrite(out_path, roi)

    # OCR (data URL compact)
    data_url = _quick_compress_to_b64_jpeg(roi)
    if not data_url:
        return jsonify({"ok": False, "error": "Encodage image échoué"}), 500

    detected = _ocr_number_from_data_url(data_url)
    if detected is not None:
        set_speed_limit(detected)

    return jsonify({
        "ok": True,
        "detected": detected,
        "used_crop": used_crop,
        "saved_path": out_path
    }), 200

# ========= Chat (inchangé, avec tool-calling) ================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_vehicle_command",
            "description": "Envoyer une commande au véhicule/robot si l'utilisateur le demande",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "avance",
                            "recule",
                            "tourne_a_droite",
                            "tourne_a_gauche",
                            "klaxonne",
                            "clignotte",
                        ],
                        "description": "La commande normalisée à exécuter"
                    }
                },
                "required": ["command"],
                "additionalProperties": False
            }
        }
    }
]

SYSTEM_PROMPT = (
    "Tu es un assistant en français. "
    "Quand l'utilisateur exprime une intention de mouvement ou d'action, "
    "tu DOIS appeler l'outil `send_vehicle_command` avec la commande normalisée."
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get('message') or "").strip()
    if not user_message:
        return jsonify({"error": "message manquant"}), 400

    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        tools=TOOLS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    msg = completion.choices[0].message
    reply_text = (msg.content or "").strip()

    tool_calls = getattr(msg, "tool_calls", None) or []
    commandes_envoyees = []
    for call in tool_calls:
        if call.type == "function" and call.function and call.function.name == "send_vehicle_command":
            try:
                args = json.loads(call.function.arguments or "{}")
                cmd = args.get("command")
                if cmd:
                    try:
                        envoyer_message(cmd)
                    except NameError:
                        print(f"[WARN] envoyer_message(...) non défini. Aurait envoyé: {cmd}")
                    commandes_envoyees.append(cmd)
            except Exception as e:
                print(f"[ERR] parse tool args: {e}")

    if not reply_text and commandes_envoyees:
        reply_text = f"Commande envoyée : {', '.join(commandes_envoyees)}."

    return jsonify({
        "reply": reply_text or "OK",
        "commandes_envoyees": commandes_envoyees or None
    }), 200

# ========= Lancement =========================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
