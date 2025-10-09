# app.py — backend avec gestion des permissions (admin cadenas) + enforcement
from flask import Flask, render_template, request, jsonify, Response, send_file, session, flash, redirect, url_for
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
        port='/dev/serial0',
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
    """Retourne des bounding boxes plausibles de panneaux vitesse."""
    try:
        img = cv2.bilateralFilter(frame_bgr, d=7, sigmaColor=60, sigmaSpace=60)
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        img = cv2.addWeighted(img, 1.6, blur, -0.6, 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
            # Vérifie cœur blanc
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

        # Fallback Hough si rien trouvé
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

# ========= OCR & Vision ======================================================
SNAP_DIR = os.path.join(os.getcwd(), "snapshots_speed_sign")
os.makedirs(SNAP_DIR, exist_ok=True)
_OCR_EXEC = ThreadPoolExecutor(max_workers=1)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Auth / rôles ------------------------------------------------------------
SHARED_PASSWORD = os.getenv("APP_SHARED_PASSWORD", "change-me")
SECRET_KEY     = os.getenv("APP_SECRET_KEY",  "dev-key-change-me")
ADMIN_USERNAME = os.getenv("APP_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD", "changeme")

_online_users = set()
_online_lock = RLock()

def is_admin() -> bool:
    return session.get("user") == ADMIN_USERNAME

# -------- Permissions (in-memory) --------------------------------------------
__perm_lock = RLock()
__permissions: dict[str, bool] = {}  # pseudo -> allowed

def get_allowed(pseudo: str) -> bool:
    with __perm_lock:
        return __permissions.get(pseudo, False)

def set_allowed(pseudo: str, allowed: bool) -> bool:
    with __perm_lock:
        __permissions[pseudo] = bool(allowed)
        return __permissions[pseudo]

def toggle_allowed(pseudo: str) -> bool:
    with __perm_lock:
        cur = __permissions.get(pseudo, True)
        __permissions[pseudo] = not cur
        return __permissions[pseudo]

# --- Utils image/ocr ---------------------------------------------------------
def _quick_compress_to_b64_jpeg(img_bgr: np.ndarray, max_side: int = 320) -> Optional[str]:
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

def _img_path_to_data_url(path: str, max_side: int = 512) -> Optional[str]:
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        ok, buff = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        b64 = base64.b64encode(buff.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None

def _ocr_number_from_data_url(data_url: str) -> Optional[int]:
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

def _limit_to_5_words(text: str) -> str:
    if not text:
        return "—"
    words = re.findall(r"\S+", text.strip())
    return " ".join(words[:5]) if words else "—"

# ========= Caméra (flux) =====================================================
CAM_INDEX_CANDIDATES = [0, 1]
WIDTH, HEIGHT, FPS = 640, 480, 30
cap = None
cap_lock = Lock()
_last_frame = None
_last_frame_lock = Lock()

def open_camera():
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
    global _last_frame
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

        with _last_frame_lock:
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
app.secret_key = SECRET_KEY

def is_logged_in() -> bool:
    return bool(session.get("user"))

_ANON_PATHS = { "/login" }
def _is_anon_allowed(path: str) -> bool:
    if path == "/":
        return False
    if path.startswith("/static/"):  # fichiers statiques
        return True
    if path in ("/favicon.ico",):
        return True
    return path in _ANON_PATHS

@app.before_request
def require_auth():
    if _is_anon_allowed(request.path):
        return
    if is_logged_in():
        with _online_lock:
            _online_users.add(session["user"])
        return
    return redirect(url_for("login", next=request.path))

# ----------- Auth ------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        # Admin
        if username == ADMIN_USERNAME:
            if password != ADMIN_PASSWORD:
                flash("Identifiants incorrects.", "error")
                return redirect(url_for('login'))
            session["user"] = username
            with _online_lock: _online_users.add(username)
            # L'admin est toujours allowed (même si non stocké)
            set_allowed(username, True)
            nxt = request.args.get("next") or url_for("index")
            return redirect(nxt)

        # Utilisateur standard
        if not username:
            error = "Merci de renseigner un pseudo."
        elif password != SHARED_PASSWORD:
            error = "Mot de passe incorrect."
        else:
            session["user"] = username
            with _online_lock: _online_users.add(username)
            # Par défaut, on autorise si inconnu
            with __perm_lock:
                __permissions.setdefault(username, False)
            nxt = request.args.get("next") or url_for("index")
            return redirect(nxt)

    return render_template("login.html", error=error)

@app.route("/logout", methods=["POST", "GET"])
def logout():
    user_to_remove = session.get("user")
    with _online_lock:
        if user_to_remove in _online_users:
            _online_users.remove(user_to_remove)
    session.clear()
    return redirect(url_for("login"))

@app.route('/users/online', methods=['GET'])
def users_online():
    if not is_logged_in() or not is_admin():
        return jsonify({"error": "forbidden"}), 403
    me = session.get("user")
    with _online_lock:
        others = sorted(u for u in _online_users if u != me)
    return jsonify({"users": others})

# --------- Permissions API ---------------------------------------------------
@app.route("/permissions/me", methods=["GET"])
def permissions_me():
    """
    Utilisateur: renvoie son propre état {allowed: bool}
    Admin: peut fournir ?user=<pseudo> pour lire l'état d'un autre utilisateur.
    """
    if not is_logged_in():
        return jsonify({"error": "unauthorized"}), 401

    target = request.args.get("user")
    if target:
        if not is_admin():
            return jsonify({"error": "forbidden"}), 403
        return jsonify({"allowed": get_allowed(target)})

    return jsonify({"allowed": get_allowed(session["user"])})

@app.route("/admin/permissions/toggle", methods=["POST"])
def admin_permissions_toggle():
    if not is_logged_in() or not is_admin():
        return jsonify({"error": "forbidden"}), 403
    data = request.get_json(silent=True) or {}
    user = (data.get("user") or "").strip()
    if not user:
        return jsonify({"error": "user manquant"}), 400
    if user == ADMIN_USERNAME:
        return jsonify({"error": "impossible de modifier l'admin"}), 400
    new_state = toggle_allowed(user)
    return jsonify({"ok": True, "user": user, "allowed": new_state})

# --------- Speed limit API ---------------------------------------------------
@app.route('/speed_limit', methods=['GET', 'POST'])
def speed_limit():
    """
    GET  -> { "speed_limit": 50 | null }
    POST -> body { "value": 50 } (ou { "speed_limit": 50 })
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

# --------- Flux vidéo & pages -----------------------------------------------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('simulator.html',
                           user=session.get("user"),
                           is_admin=is_admin(),
                           admin_name=ADMIN_USERNAME)

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html",
                           user=session.get("user"),
                           is_admin=is_admin())

# --------- Enforcement + simulate -------------------------------------------
@app.route('/simulate', methods=['POST'])
def simulate():
    user = session.get("user")
    if not is_admin() and not get_allowed(user):
        return jsonify({"ok": False, "allowed": False, "error": "disabled_by_admin"}), 403
    action = (request.json or {}).get('action')
    envoyer_message(action)
    print(f"Commande envoyée: {action}")
    return jsonify({"status": "success", "action": action, "allowed": True})

# --------- Capture manuelle + OCR -------------------------------------------
@app.route('/capture_speed_sign', methods=['POST'])
def capture_speed_sign():
    user = session.get("user")
    if not is_admin() and not get_allowed(user):
        return jsonify({"ok": False, "allowed": False, "error": "disabled_by_admin"}), 403

    with _last_frame_lock:
        frame = _last_frame.copy() if '_last_frame' in globals() and _last_frame is not None else None

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

    boxes = detect_speed_limit_candidates(frame)
    used_crop = False
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    full_path = os.path.join(SNAP_DIR, f"capture_{ts}_full.jpg")
    cv2.imwrite(full_path, frame)

    roi = frame
    if boxes:
        bx = max(boxes, key=lambda b: b[2]*b[3])
        x, y, w, h = bx
        h_img, w_img = frame.shape[:2]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(w_img, x + w); y1 = min(h_img, y + h)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1].copy()
            used_crop = True

    crop_path = os.path.join(SNAP_DIR, f"capture_{ts}{'_crop' if used_crop else ''}.jpg")
    cv2.imwrite(crop_path, roi)

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
        "saved_path": crop_path,
        "full_path": full_path,
        "frame_size": {"w": int(frame.shape[1]), "h": int(frame.shape[0])}
    }), 200

# --------- Servir une image capturée ----------------------------------------
@app.route('/snap', methods=['GET'])
def snap():
    raw = (request.args.get('path') or '').strip()
    if not raw:
        return "path manquant", 400

    try:
        abs_path = os.path.realpath(raw)
        root = os.path.realpath(SNAP_DIR)
        if abs_path != root and not abs_path.startswith(root + os.sep):
            return "Chemin non autorisé", 403
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return "Fichier introuvable", 404
    except Exception:
        return "Chemin invalide", 400

    mimetype = "image/jpeg"
    if abs_path.lower().endswith(".png"):
        mimetype = "image/png"
    elif abs_path.lower().endswith(".webp"):
        mimetype = "image/webp"

    return send_file(abs_path, mimetype=mimetype, as_attachment=False, download_name=os.path.basename(abs_path))

# --------- Description IA ----------------------------------------------------
@app.route('/describe_image', methods=['POST'])
def describe_image():
    data = request.get_json(silent=True) or {}
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"ok": False, "error": "path manquant"}), 400

    try:
        abs_path = os.path.abspath(path)
        root = os.path.abspath(SNAP_DIR)
        if abs_path != root and not abs_path.startswith(root + os.sep):
            return jsonify({"ok": False, "error": "Chemin non autorisé"}), 403
        if not os.path.exists(abs_path):
            return jsonify({"ok": False, "error": "Fichier introuvable"}), 404
    except Exception:
        return jsonify({"ok": False, "error": "Chemin invalide"}), 400

    data_url = _img_path_to_data_url(abs_path, max_side=640)
    if not data_url:
        return jsonify({"ok": False, "error": "Lecture/encodage image échoué"}), 500

    def to_bool(v):
        if isinstance(v, bool): return v
        if isinstance(v, str): return v.strip().lower() in ("1","true","vrai","yes","on")
        return False

    brief_override = data.get("brief", None)
    if brief_override is None:
        brief_override = request.args.get("brief", None)
    mode = (data.get("mode") or request.args.get("mode") or "").strip().lower()

    brief = None
    if mode in ("sim", "simulateur", "simulator"):
        brief = True
    elif mode in ("chat", "chatbot"):
        brief = False
    elif brief_override is not None:
        brief = to_bool(brief_override)

    if brief is None:
        ref = (request.headers.get("Referer") or "").lower()
        if "simulator" in ref:
            brief = True
        elif "chatbot" in ref:
            brief = False
        else:
            brief = False

    prompt = (
        "Que vois-tu sur l'image ? Réponds en français, de manière complète et précise. "
        "Décris les objets, le contexte (route, météo, trafic), les panneaux visibles, "
        "et toute information pertinente pour la conduite."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=800,
        )
        description_full = (resp.choices[0].message.content or "").strip()
        description_out = _limit_to_5_words(description_full) if brief else description_full

        return jsonify({
            "ok": True,
            "description": description_out,
            "data_url": data_url,
            "description_full": description_full if brief else None
        }), 200
    except Exception as e:
        if DEBUG: print(f"[Describe] Erreur: {e}")
        return jsonify({"ok": False, "error": "Échec description"}), 500

# ========= Chat (avec enforcement) ===========================================
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
    user = session.get("user")
    if not is_admin() and not get_allowed(user):
        return jsonify({"ok": False, "allowed": False, "error": "disabled_by_admin"}), 403

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
        "ok": True,
        "allowed": True,
        "reply": reply_text or "OK",
        "commandes_envoyees": commandes_envoyees or None
    }), 200

# ========= Lancement =========================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
