from flask import Flask, render_template, request, jsonify, Response
import bluetooth
import cv2
import atexit
import time
from threading import Lock
import serial

# Configure le port série
ser = serial.Serial(
    port='/dev/serial0',    # Peut aussi être /dev/ttyAMA0 ou /dev/ttyS0 selon le modèle
    baudrate=9600,
    timeout=1               # Temps d'attente pour lecture
)

def envoyer_message(message):
    ser.write((message + '\n').encode('utf-8'))
    print(f"Envoyé à l'Arduino : {message}")


# Adresse MAC de ton module HC-06 (à adapter)
HC06_MAC_ADDRESS = '00:14:03:06:0C:02'
PORT = 1  # Le port RFCOMM standard

# try:
    # print(f"Connexion à {HC06_MAC_ADDRESS}...")
    # sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    # sock.connect((HC06_MAC_ADDRESS, PORT))
    # time.sleep(5)
    # print("Connecté au module HC-06")
    # # Configurer le serveur Bluetooth
    # server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    # server_socket.bind(("", bluetooth.PORT_ANY))
    # server_socket.listen(2)

    # port = server_socket.getsockname()[1]
    # bluetooth.advertise_service(server_socket, "LEDControl",
                            # service_classes=[bluetooth.SERIAL_PORT_CLASS],
                            # profiles=[bluetooth.SERIAL_PORT_PROFILE])

    # #print(f"En attente de connexion Bluetooth sur le port {port}...")

    # #client_socket, client_info = server_socket.accept()
    
    # #print(f"Connexion acceptée de {client_info}")

# except bluetooth.btcommon.BluetoothError as e:
    # print(f"Erreur Bluetooth : {e}")


app = Flask(__name__)

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
    envoyer_message(action)
    print(f"Commande envoyée: {action}")
    return jsonify({"status": "success", "action": action})

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    return jsonify({"reply": f"Tu as dit: {message}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

