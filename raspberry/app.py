from flask import Flask, render_template, request, jsonify, Response
import cv2
import atexit
import time
from threading import Lock
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import numpy as np
import base64
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

DEBUG = False

if DEBUG:
    pass
else:
    import serial
    # Configure le port série
    ser = serial.Serial(
        port='/dev/serial0',    # Peut aussi être /dev/ttyAMA0 ou /dev/ttyS0 selon le modèle
        baudrate=9600,
        timeout=1               # Temps d'attente pour lecture
    )

def envoyer_message(message):

    if DEBUG:
        pass
    else:
        ser.write((message + '\n').encode('utf-8'))
        print(f"Envoyé à l'Arduino : {message}")

# Adresse MAC de ton module HC-06 (à adapter)
HC06_MAC_ADDRESS = '00:14:03:06:0C:02'
PORT = 1  # Le port RFCOMM standard

# Active/désactive la détection (pratique pour tester)
ENABLE_SPEED_SIGN_DETECT = True

# Seuils HSV pour le rouge (anneau rouge du panneau)
_LOWER_RED_1 = np.array([0, 70, 50])
_UPPER_RED_1 = np.array([10, 255, 255])
_LOWER_RED_2 = np.array([170, 70, 50])
_UPPER_RED_2 = np.array([180, 255, 255])

# Seuils d'"intérieur blanc" (faible saturation, forte luminosité)
_SAT_MAX_WHITE = 60
_VAL_MIN_WHITE = 160

# Contraintes géométriques
_MIN_AREA = 800          # taille minimale du contour
_MIN_PERIM = 100         # périmètre minimal
_MIN_CIRCULARITY = 0.5   # 1.0 = cercle parfait


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
        # Lissage léger pour réduire le bruit
        img = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Masque rouge (deux plages HSV pour couvrir le wrap 0°/180°)
        mask1 = cv2.inRange(hsv, _LOWER_RED_1, _UPPER_RED_1)
        mask2 = cv2.inRange(hsv, _LOWER_RED_2, _UPPER_RED_2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        # Nettoyage morphologique
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN,  kernel, iterations=1)

        # Contours du rouge
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < _MIN_AREA:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < _MIN_PERIM:
                continue

            circ = _circularity(area, perim)
            if circ < _MIN_CIRCULARITY:
                continue

            # Cercle englobant
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < 12:  # évite les mini faux-positifs
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Vérifie qu'il y a un cœur "blanc" (faible saturation, forte valeur) à l'intérieur
            # Crée un masque de disque intérieur (60% du rayon)
            inner_radius = max(int(radius * 0.6), 8)
            center = (int(cx), int(cy))
            h_img, w_img = hsv.shape[:2]

            # ROI sécurisée autour du cercle
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
            # Moyennes pondérées par le masque intérieur
            inner_pixels = mask_inner > 0
            if inner_pixels.sum() == 0:
                continue

            mean_sat = float(sat[inner_pixels].mean())
            mean_val = float(val[inner_pixels].mean())

            looks_white_inside = (mean_sat <= _SAT_MAX_WHITE and mean_val >= _VAL_MIN_WHITE)
            if not looks_white_inside:
                continue

            boxes.append((x, y, w, h))

        return boxes
    except Exception:
        # Sécurité : en cas d'erreur, on ne bloque pas le flux
        return []

import os
from datetime import datetime

SNAP_DIR = os.path.join(os.getcwd(), "snapshots_speed_sign")
os.makedirs(SNAP_DIR, exist_ok=True)

# Anti-spam simple : délai minimal entre deux captures (en secondes)
CAPTURE_COOLDOWN_S = 2.0
_last_capture_ts = 0.0
# OCR async
_OCR_EXEC = ThreadPoolExecutor(max_workers=1)
_ocr_inflight = False

def _quick_compress_to_b64_jpeg(path: str) -> Optional[str]:
    """
    Charge l'image, convertit en niveaux de gris, redimensionne à max 320px,
    encode en JPEG qualité 60, renvoie une data URL prête à envoyer.
    """
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        # niveaux de gris -> 1 canal (moins d'entropie) puis retour BGR pour jpg
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        max_side = max(h, w)
        if max_side > 320:
            scale = 320.0 / max_side
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Remet en 3 canaux pour encode JPEG classique
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ok, buff = cv2.imencode(".jpg", gray3, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return None
        b64 = base64.b64encode(buff.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None

def _ocr_job(image_path: str):
    """Tâche exécutée dans le thread pool : prépare l'image et appelle l'IA."""
    global _ocr_inflight
    try:
        data_url = _quick_compress_to_b64_jpeg(image_path)
        if not data_url:
            if DEBUG: print("[OCR] Prétraitement/encodage échoué.")
            return
        _analyze_speed_sign_and_set_limit_from_data_url(data_url)
    finally:
        _ocr_inflight = False  # libère le slot même si erreur

def _enqueue_ocr(image_path: str):
    """Si pas déjà en cours, programme un OCR en arrière-plan."""
    global _ocr_inflight
    if _ocr_inflight:
        return
    _ocr_inflight = True
    _OCR_EXEC.submit(_ocr_job, image_path)


def _maybe_save_crop(frame: np.ndarray, bbox):
    global _last_capture_ts
    try:
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w_img, x + w)
        y1 = min(h_img, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        crop = frame[y0:y1, x0:x1].copy()

        # Cooldown
        now = datetime.now().timestamp()
        if (now - _last_capture_ts) < CAPTURE_COOLDOWN_S:
            return
        _last_capture_ts = now

        # Nom horodaté
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(SNAP_DIR, f"speed_sign.jpg")
        ok = cv2.imwrite(filename, crop)
        if not ok:
            if DEBUG:
                print("[OCR] Échec d'écriture du fichier image.")
            return

        # ➜ Analyse immédiate + mise à jour de la limite
        #_analyze_speed_sign_and_set_limit(filename)
        _enqueue_ocr(filename) 

    except Exception as e:
        if DEBUG:
            print(f"[OCR] Erreur _maybe_save_crop: {e}")
        pass

# Charge les variables du fichier .env à la racine du projet
load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Définition de l'outil que le modèle peut appeler ---
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
    "Quand l'utilisateur exprime une intention de mouvement ou d'action (ex: avancement, reculer, tourner à droite, "
    "tourner à gauche, klaxonner, clignoter/‘blink’), tu DOIS appeler l'outil "
    "`send_vehicle_command` avec la commande normalisée correspondante parmi "
    "['avance','recule','tourne_a_droite','tourne_a_gauche','klaxonne','clignotte'].\n"
    "Exemples:\n"
    "- « avance un peu » -> command=avance\n"
    "- « recule » -> command=recule\n"
    "- « tourne à droite » -> command=tourne_a_droite\n"
    "- « fais clignoter la led », « fais blinker la led » -> command=clignotte\n"
    "Réponds aussi un court message à l'utilisateur. "
    "Si la demande n'est pas une commande, réponds normalement sans appeler l'outil."
)

def _analyze_speed_sign_and_set_limit_from_data_url(data_url: str):
    """
    Version optimisée qui reçoit directement une data URL (JPEG compressé).
    Utilise le même prompt strict. Appelle set_speed_limit(...) si un entier (1-3 chiffres) est trouvé.
    """
    try:
        messages = [
            {"role": "system",
             "content": "Tu es un extracteur OCR ultra-strict. "
                        "Réponds UNIQUEMENT le nombre imprimé au centre du panneau (ex: 30). "
                        "Aucune unité, aucun mot, aucun symbole."},
        ]
        user_content = [
            {"type": "text", "text": "Quel est le nombre sur le panneau ? Réponds uniquement avec ce nombre."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=5,  # sortie minuscule
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\b(\d{1,3})\b", txt)
        if not m:
            if DEBUG: print(f"[OCR] Réponse non exploitable: {txt!r}")
            return
        value = int(m.group(1))
        set_speed_limit(value)  # protège déjà et borne 0..200 :contentReference[oaicite:3]{index=3}
        if DEBUG: print(f"[OCR] Limite détectée: {value} -> set_speed_limit({value})")
    except Exception as e:
        if DEBUG: print(f"[OCR] Erreur OCR: {e}")


def _analyze_speed_sign_and_set_limit(image_path: str):
    """
    Envoie l'image du panneau au modèle vision d'OpenAI, récupère le nombre
    (ex: 30, 50, 110) et met à jour la vitesse limite via set_speed_limit(...).
    En cas d'échec (pas de nombre détecté), ne change rien.
    """
    try:
        # Encode l'image en data URL (base64) pour chat.completions avec image
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"

        # Invite très stricte: on veut UNIQUEMENT le nombre, sans unité/texte.
        messages = [
            {"role": "system",
             "content": "Tu es un extracteur OCR ultra-strict. "
                        "Réponds UNIQUEMENT le nombre imprimé au centre du panneau (ex: 30). "
                        "Aucune unité, aucun mot, aucun symbole."},
            {"role": "user",
             "content": [
                 {"type": "text",
                  "text": "Quel est le nombre sur le panneau ? Réponds uniquement avec ce nombre."},
                 {"type": "image_url", "image_url": {"url": data_url}}
             ]}
        ]

        # Utilise un modèle vision (gpt-4o-mini convient très bien pour l'OCR simple)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=messages
        )
        txt = (resp.choices[0].message.content or "").strip()

        # Sécurise: on ne garde que le premier entier à 1-3 chiffres
        m = re.search(r"\b(\d{1,3})\b", txt)
        if not m:
            if DEBUG:
                print(f"[OCR] Rien d'exploitable dans la réponse: {txt!r}")
            return

        value = int(m.group(1))
        # Mets à jour la limite (ta fonction borne déjà 0..200) :contentReference[oaicite:1]{index=1}
        set_speed_limit(value)

        if DEBUG:
            print(f"[OCR] Limite détectée: {value} km/h -> set_speed_limit({value})")

    except Exception as e:
        if DEBUG:
            print(f"[OCR] Erreur analyse panneau: {e}")


#---- Paramètres caméra (adapte si besoin) ----
CAM_INDEX_CANDIDATES = [0, 1]       # essaie /dev/video0 puis /dev/video1
WIDTH, HEIGHT = 640, 480            # 1280x720 marche aussi si MJPG
FPS = 30

cap = None
cap_lock = Lock()

def open_camera():
    """Ouvre la caméra USB avec V4L2 + MJPG, et applique largeur/hauteur/FPS."""
    global cap
    # Essaie plusieurs index au cas où /dev/video1 serait utilisé
    for idx in CAM_INDEX_CANDIDATES:
        c = cv2.VideoCapture(idx)  # backend V4L2 sur RPi
        if not c.isOpened():
            c.release()
            continue

        # FourCC MJPG (évite la conversion CPU YUYV -> BGR)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        c.set(cv2.CAP_PROP_FPS, FPS)

        # Petit délai pour que les réglages s’appliquent
        time.sleep(0.2)

        # Vérifie qu’on lit bien une image
        ok, _ = c.read()
        if ok:
            return c
        c.release()
    return None

def ensure_camera():
    """(Ré)ouvre la caméra si besoin, thread-safe."""
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = open_camera()
    return cap

def release_camera():
    global cap
    with cap_lock:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None

atexit.register(release_camera)

def generate_frames():
    """Génère un flux MJPEG pour /video_feed."""
    # S’assure que la caméra est prête
    if ensure_camera() is None:
        # Renvoie un unique cadre noir si échec (pour feedback)
        import numpy as np
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

            if ENABLE_SPEED_SIGN_DETECT:
                boxes = detect_speed_limit_candidates(frame)
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Limitation (probable)", (x, max(0, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    _maybe_save_crop(frame, (x, y, w, h))

            #ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            #if not ret:
            #    continue
            #jpg = buffer.tobytes()
            #yield (b'--frame\r\n'
            #    b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

        if not success:
            # Tentative de reconnexion (caméra débranchée/rebranchée)
            release_camera()
            if ensure_camera() is None:
                break
            continue

        # Ré-encodage JPEG (rapide si source est MJPG)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        jpg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

# --- ajoute près des autres imports/globals ---
from threading import RLock

# Valeur de vitesse limite + verrou
_speed_limit_value = None
_speed_limit_lock = RLock()

def set_speed_limit(value: int | float | None):
    """
    Met à jour la vitesse limite côté serveur (utilisable depuis n'importe quel code Python).
    value=None permet d'effacer/mettre '—'.
    """
    global _speed_limit_value
    with _speed_limit_lock:
        if value is None:
            _speed_limit_value = None
        else:
            try:
                v = float(value)
                # garde dans un intervalle raisonnable (optionnel)
                if v < 0: v = 0
                if v > 200: v = 200
                _speed_limit_value = v
            except Exception:
                # si non convertible, on ignore
                pass

@app.route('/speed_limit', methods=['GET', 'POST'])
def speed_limit():
    """
    GET  -> renvoie la vitesse limite courante: { "speed_limit": 50 } ou null
    POST -> met à jour via JSON: { "value": 50 } ou { "speed_limit": 50 }
    """
    global _speed_limit_value
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        value = data.get('value', data.get('speed_limit', None))
        set_speed_limit(value)
        with _speed_limit_lock:
            return jsonify({"ok": True, "speed_limit": _speed_limit_value}), 200

    # GET
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
    #sock.send(action+ '\n')

    if DEBUG:
        pass
    else:
        envoyer_message(action)
    
    print(f"Commande envoyée: {action}")
    return jsonify({"status": "success", "action": action})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get('message') or "").strip()
    if not user_message:
        return jsonify({"error": "message manquant"}), 400

    # Appel du modèle avec tool calling
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

    # Exécuter toutes les tool calls retournées (s'il y en a)
    commandes_envoyees = []
    tool_calls = getattr(msg, "tool_calls", None) or []
    for call in tool_calls:
        if call.type == "function" and call.function and call.function.name == "send_vehicle_command":
            try:
                args = json.loads(call.function.arguments or "{}")
                cmd = args.get("command")
                if cmd:
                    try:
                        envoyer_message(cmd)  # <-- ta fonction
                    except NameError:
                        # En dev si la fonction n'est pas importée
                        print(f"[WARN] envoyer_message(...) non défini. Aurait envoyé: {cmd}")
                    commandes_envoyees.append(cmd)
            except Exception as parse_err:
                print(f"[ERR] parse tool args: {parse_err}")

    # Si le modèle n'a pas parlé (contenu vide), on renvoie au moins un accusé
    if not reply_text and commandes_envoyees:
        reply_text = f"Commande envoyée : {', '.join(commandes_envoyees)}."

    return jsonify({
        "reply": reply_text or "OK",
        "commandes_envoyees": commandes_envoyees or None
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

