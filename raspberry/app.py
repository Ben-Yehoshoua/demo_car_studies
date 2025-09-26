from flask import Flask, render_template, request, jsonify, Response
import cv2
import atexit
import time
from threading import Lock
from dotenv import load_dotenv
from openai import OpenAI
import os
import json

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

