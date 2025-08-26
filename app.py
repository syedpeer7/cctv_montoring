"""
prototype.py
Simple CCTV prototype: YOLOv8 detection + tracker -> store crops, embeddings, OCR plates.
Run: python prototype.py
"""

import os, time, math, sqlite3, uuid, threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# === Config ===
# Add up to 4 sources: 0 = laptop webcam; "http://phone-ip:8080/video" for phone cameras
CAM_SOURCES = [
    0,  # Laptop webcam
    "http://10.189.3.89:8080/video",  # Phone 1 IP Webcam URL
    "http://192.0.0.4:8080/video",  # Phone 2 IP Webcam URL
    "http://192.168.1.12:8080/video"   # Phone 3 IP Webcam URL
]
SAVE_DIR = Path("data")
SAVE_DIR.mkdir(exist_ok=True)
DB_PATH = "db.sqlite"
DETECTION_CONF = 0.35
EMBED_DIM = 512  # target dim for embeddings

# === DB helpers ===
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sightings(
        id TEXT PRIMARY KEY, ts REAL, cam TEXT, track_id INTEGER, cls TEXT, bbox TEXT, image_path TEXT, plate TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS embeddings(
        sight_id TEXT, kind TEXT, vector BLOB
    )""")
    conn.commit()
    return conn

conn = init_db()

# === Optional libs and models (face, reid, ocr) ===
try:
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(allowed_modules=['detection','recognition'])
    face_app.prepare(ctx_id=-1, det_size=(640,640))
    print("[INFO] InsightFace loaded")
except Exception as e:
    face_app = None
    print("[WARN] InsightFace not available:", e)

try:
    from torchreid.utils import FeatureExtractor
    reid_extractor = FeatureExtractor(model_name='osnet_x1_0', model_path='', device='cpu')
    use_torchreid = True
    print("[INFO] TorchReID feature extractor loaded")
except Exception as e:
    reid_extractor = None
    use_torchreid = False
    print("[WARN] TorchReID not available:", e)

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    use_paddle = True
    print("[INFO] PaddleOCR loaded")
except Exception:
    try:
        import pytesseract
        use_paddle = False
        print("[INFO] pytesseract fallback available")
    except Exception:
        ocr = None
        use_paddle = False
        print("[WARN] No OCR available")

if not use_torchreid:
    try:
        import torch
        import torchvision.transforms as T
        from torchvision import models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resnet = models.resnet50(pretrained=True).to(device).eval()
        preprocess = T.Compose([
            T.Resize((256,128)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        def resnet_embed(pil_img):
            x = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet(x)
            vec = feat.cpu().numpy().ravel()
            vec = vec / (np.linalg.norm(vec)+1e-6)
            if len(vec) >= EMBED_DIM:
                return vec[:EMBED_DIM]
            else:
                v = np.zeros(EMBED_DIM, dtype=np.float32)
                v[:len(vec)] = vec
                return v
        print("[INFO] ResNet fallback embeddings ready")
    except Exception as e:
        resnet_embed = None
        print("[WARN] No embedding model loaded:", e)

# === Utility functions and gallery ===
def crop_box(bgr, xyxy):
    x1,y1,x2,y2 = map(int, xyxy)
    h,w = bgr.shape[:2]
    x1,x2 = max(0,x1), min(w,x2)
    y1,y2 = max(0,y1), min(h,y2)
    return bgr[y1:y2, x1:x2]

def save_snapshot(img_bgr, cam, track_id):
    fn = SAVE_DIR / f"{cam}_{track_id}_{int(time.time()*1000)}.jpg"
    cv2.imwrite(str(fn), img_bgr)
    return str(fn)

def vector_to_bytes(v):
    return v.tobytes()

GALLERY = []
def add_to_gallery(sight_id, kind, vec):
    GALLERY.append({'id': sight_id, 'kind': kind, 'vec': vec})

def search_gallery(vec, kind='reid', topk=5):
    sims = []
    q = vec / (np.linalg.norm(vec)+1e-6)
    for g in GALLERY:
        if g['kind'] != kind: continue
        v = g['vec']
        v = v / (np.linalg.norm(v)+1e-6)
        sims.append((np.dot(q, v), g))
    sims.sort(key=lambda x: -x[0])
    return sims[:topk]

print("[INFO] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# === Main camera processing loop (same as before) ===
def process_stream(cam_id, source):
    cap = cv2.VideoCapture(source)
    cam_name = str(source)
    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Frame read failed from {source}")
            time.sleep(0.5)
            continue
        results = model.track(frame, persist=True, conf=DETECTION_CONF)
        if not results:
            cv2.imshow(f"cam{cam_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            cv2.imshow(f"cam{cam_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        try:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)
        except Exception:
            xyxy_arr, confs, clss, ids = [], [], [], []
            for b in boxes:
                try:
                    xyxy_arr.append(b.xyxy[0].cpu().numpy())
                    confs.append(float(b.conf.cpu().numpy()))
                    clss.append(int(b.cls.cpu().numpy()))
                    ids.append(int(b.id) if hasattr(b, "id") else -1)
                except Exception:
                    pass
            xyxy_arr = np.array(xyxy_arr)
            confs = np.array(confs)
            clss = np.array(clss)
            ids = np.array(ids)

        for i, box in enumerate(xyxy_arr):
            cls_idx = int(clss[i]); conf = float(confs[i])
            tid = int(ids[i]) if len(ids)>0 else -1
            label = model.names[cls_idx]
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"{label}#{tid} {conf:.2f}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

            if label == "person" or label in ("car","truck","bus","motorbike","bicycle"):
                crop = crop_box(frame, box)
                if crop.size == 0: continue
                sight_id = str(uuid.uuid4())
                img_path = save_snapshot(crop, cam_name, tid)
                vec = None
                try:
                    if use_torchreid:
                        vecs = reid_extractor(crop)
                        vec = vecs[0].cpu().numpy()
                    else:
                        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        vec = resnet_embed(pil)
                except Exception as e:
                    print("[WARN] reid embed failed:", e)
                face_vec = None
                if face_app is not None:
                    try:
                        faces = face_app.get(crop)
                        if len(faces)>0:
                            face_vec = faces[0].embedding / (np.linalg.norm(faces[0].embedding)+1e-6)
                    except Exception as e:
                        print("[WARN] face embed failed", e)
                plate_text = None
                if label != "person" and (ocr is not None or 'pytesseract' in globals()):
                    try:
                        if use_paddle:
                            res = ocr.ocr(crop, cls=True)
                            texts = []
                            for line in res:
                                for item in line:
                                    if isinstance(item, list) and len(item)>=2:
                                        texts.append(item[1][0])
                            plate_text = " ".join(texts).upper().strip()
                        else:
                            import pytesseract
                            txt = pytesseract.image_to_string(crop)
                            plate_text = txt.strip().upper()
                    except Exception as e:
                        print("[WARN] OCR failed:", e)
                c = conn.cursor()
                c.execute("INSERT INTO sightings(id, ts, cam, track_id, cls, bbox, image_path, plate) VALUES (?,?,?,?,?,?,?,?)",
                          (sight_id, time.time(), cam_name, int(tid), label, ",".join(map(str,map(int,box))), img_path, plate_text))
                if vec is not None:
                    c.execute("INSERT INTO embeddings(sight_id, kind, vector) VALUES (?,?,?)",
                              (sight_id, "reid", vector_to_bytes(np.array(vec,dtype=np.float32))))
                    add_to_gallery(sight_id, "reid", np.array(vec,dtype=np.float32))
                if face_vec is not None:
                    c.execute("INSERT INTO embeddings(sight_id, kind, vector) VALUES (?,?,?)",
                              (sight_id, "face", vector_to_bytes(np.array(face_vec,dtype=np.float32))))
                    add_to_gallery(sight_id, "face", np.array(face_vec,dtype=np.float32))
                conn.commit()

        cv2.imshow(f"cam{cam_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"cam{cam_id}")

# === Launch multiple cameras in parallel ===
if __name__ == "__main__":
    print("Starting streams on:", CAM_SOURCES)
    threads = []
    for idx, src in enumerate(CAM_SOURCES):
        t = threading.Thread(target=process_stream, args=(idx, src))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
