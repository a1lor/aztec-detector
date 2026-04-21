#!/usr/bin/env python3
"""
server.py — API Backend FastAPI pour le modèle YOLOv8 Segmentation
Endpoints :
  POST /predict              → Segmentation d'une image uploadée
  GET  /health               → État du serveur
  GET  /training/status       → Rapport d'entraînement en temps réel
  WS   /ws/training          → WebSocket pour les rapports en direct
  GET  /models               → Liste des modèles disponibles
  POST /models/load          → Charger un modèle spécifique
Usage:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
  # Ou avec le script directement :
  python api/server.py --model runs/segment/seg_train/weights/best.pt --port 8000
"""
import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

# SAHI — tiling pour images larges (planches, manuscrits)
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("⚠️  SAHI non installé. pip install sahi pour activer le tiling.")
# ============================================================
# Configuration
# ============================================================
DEFAULT_MODEL_PATH = "best_fixed.pt"
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = Path("runs/reports")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# ============================================================
# Application FastAPI
# ============================================================
app = FastAPI(
    title="YOLO Segmentation API",
    description="API de segmentation d'images avec YOLOv8 — 307 classes",
    version="1.0.0",
)
# CORS — autorise ton frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod, restreindre à ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve static assets and HTML frontend
_static_dir   = BASE_DIR / "static"
_template_dir = BASE_DIR / "templates"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/", include_in_schema=False)
async def frontend():
    """Serve the main HTML interface."""
    html_file = _template_dir / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Interface introuvable.")
    return FileResponse(str(html_file))
# ============================================================
# État global
# ============================================================
class AppState:
    def __init__(self):
        self.model       = None          # YOLO model
        self.sahi_model  = None          # SAHI wrapper pour le tiling
        self.model_path: Optional[str] = None
        self.device: str = "0"
        self.is_loading: bool = False

state = AppState()

# Seuil (px) au-dessus duquel le tiling est activé automatiquement
AUTO_TILING_THRESHOLD = 900
# TILE_SIZE doit correspondre à imgsz utilisé lors de l'entraînement.
# Modèle actuel (best_fixed.pt) → imgsz=512 → TILE_SIZE=512
# Après réentraînement avec imgsz=640 → passer TILE_SIZE à 640
TILE_SIZE    = 640
TILE_OVERLAP = 0.3

def load_model(model_path: str, device: str = "0"):
    """Charge le modèle YOLO + le wrapper SAHI pour le tiling."""
    from ultralytics import YOLO
    import torch
    state.is_loading = True
    try:
        # Résolution du device : CUDA > MPS > CPU
        if torch.cuda.is_available() and device.isdigit():
            resolved_device = device
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
        state.model      = YOLO(model_path)
        state.model_path = model_path
        state.device     = resolved_device
        print(f"✅ Modèle chargé : {model_path} (device: {resolved_device})")

        if SAHI_AVAILABLE:
            try:
                if torch.cuda.is_available() and device.isdigit():
                    sahi_device = f"cuda:{device}"
                elif torch.backends.mps.is_available():
                    sahi_device = "cpu"  # SAHI ne supporte pas MPS, forcer CPU
                else:
                    sahi_device = "cpu"
                state.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=model_path,
                    confidence_threshold=0.01,   # filtrage dynamique en post-processing
                    device=sahi_device,
                )
                print(f"✅ SAHI prêt (tiling, device: {sahi_device})")
            except Exception as e:
                print(f"⚠️  SAHI non disponible : {e}")
                state.sahi_model = None
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        raise
    finally:
        state.is_loading = False
# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
async def health():
    """Vérifie l'état du serveur."""
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "model_path": state.model_path,
        "device": state.device,
        "tiling_available": SAHI_AVAILABLE and state.sahi_model is not None,
    }
@app.get("/models")
async def list_models():
    """Liste tous les modèles .pt disponibles dans runs/."""
    models = []
    runs_dir = Path("runs")
    if runs_dir.exists():
        for pt_file in runs_dir.rglob("*.pt"):
            stat = pt_file.stat()
            models.append({
                "path": str(pt_file),
                "name": pt_file.name,
                "size_mb": round(stat.st_size / 1e6, 2),
                "modified": stat.st_mtime,
            })
    return {"models": sorted(models, key=lambda x: x["modified"], reverse=True)}
@app.post("/models/load")
async def load_model_endpoint(
    model_path: str = Query(..., description="Chemin vers le modèle .pt"),
    device: str = Query("0", description="GPU device"),
):
    """Charge un modèle spécifique."""
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Modèle introuvable : {model_path}")
    try:
        load_model(model_path, device)
        return {"status": "loaded", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Seuil de confiance"),
    iou: float  = Query(0.7,  ge=0.0, le=1.0, description="Seuil IoU NMS"),
    tiling: Optional[bool] = Query(None, description="Forcer le tiling (None = auto)"),
    return_masks: bool = Query(False, description="Inclure les masques en base64"),
    return_image: bool = Query(False, description="Inclure l'image annotée en base64"),
):
    """
    Inférence sur une image.
    - tiling=None  → auto (activé si image > 900px)
    - tiling=true  → force le tiling SAHI (recommandé pour planches)
    - tiling=false → inférence directe YOLOv8 standard
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Aucun modèle chargé. Utilisez POST /models/load d'abord.",
        )
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {ext}. Formats acceptés : {ALLOWED_EXTENSIONS}",
        )
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image illisible")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture image : {e}")

    img_h, img_w = img.shape[:2]

    # Décide si on utilise le tiling
    use_tiling = (
        tiling is True
        or (tiling is None and max(img_w, img_h) > AUTO_TILING_THRESHOLD)
    )
    if use_tiling and state.sahi_model is None:
        use_tiling = False   # fallback si SAHI non disponible

    start_time = time.time()
    detections = []

    # ── Mode tiling SAHI ──────────────────────────────────────────────
    if use_tiling:
        state.sahi_model.confidence_threshold = conf
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Taille de tuile adaptée à la taille de l'image
        tile_size = TILE_SIZE if max(img_w, img_h) <= 2000 else 800

        sahi_result = get_sliced_prediction(
            pil_img,
            state.sahi_model,
            slice_height=tile_size,
            slice_width=tile_size,
            overlap_height_ratio=TILE_OVERLAP,
            overlap_width_ratio=TILE_OVERLAP,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=iou,
            verbose=0,
        )

        for j, pred in enumerate(sahi_result.object_prediction_list):
            if pred.score.value < conf:
                continue
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            detections.append({
                "id": j,
                "class_id": int(pred.category.id),
                "class_name": pred.category.name,
                "confidence": round(float(pred.score.value), 4),
                "bbox": {
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                },
                "bbox_normalized": {
                    "x_center": round((x1 + x2) / 2 / img_w, 6),
                    "y_center": round((y1 + y2) / 2 / img_h, 6),
                    "width":    round((x2 - x1) / img_w, 6),
                    "height":   round((y2 - y1) / img_h, 6),
                },
            })

    # ── Mode standard YOLOv8 ─────────────────────────────────────────
    else:
        results = state.model.predict(
            source=img,
            conf=conf,
            iou=iou,
            device=state.device,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            masks = result.masks
            for j in range(len(boxes)):
                det = {
                    "id": j,
                    "class_id": int(boxes.cls[j]),
                    "class_name": result.names[int(boxes.cls[j])],
                    "confidence": round(float(boxes.conf[j]), 4),
                    "bbox": {
                        "x1": round(float(boxes.xyxy[j][0]), 2),
                        "y1": round(float(boxes.xyxy[j][1]), 2),
                        "x2": round(float(boxes.xyxy[j][2]), 2),
                        "y2": round(float(boxes.xyxy[j][3]), 2),
                    },
                    "bbox_normalized": {
                        "x_center": round(float(boxes.xywhn[j][0]), 6),
                        "y_center": round(float(boxes.xywhn[j][1]), 6),
                        "width":    round(float(boxes.xywhn[j][2]), 6),
                        "height":   round(float(boxes.xywhn[j][3]), 6),
                    },
                }
                if masks is not None and j < len(masks):
                    mask_xy = masks.xy[j]
                    if len(mask_xy) > 0:
                        det["polygon"] = [
                            {"x": round(float(pt[0]), 2), "y": round(float(pt[1]), 2)}
                            for pt in mask_xy
                        ]
                    if return_masks:
                        mask_np = (masks.data[j].cpu().numpy() * 255).astype(np.uint8)
                        _, mask_encoded = cv2.imencode(".png", mask_np)
                        det["mask_base64"] = base64.b64encode(mask_encoded).decode("utf-8")
                detections.append(det)

    inference_time = round((time.time() - start_time) * 1000, 2)

    # Image annotée (optionnel, mode standard uniquement)
    annotated_image_b64 = None
    if return_image and not use_tiling:
        annotated = results[0].plot()
        _, img_encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_image_b64 = base64.b64encode(img_encoded).decode("utf-8")

    return {
        "image_name": file.filename,
        "image_size": {"width": img_w, "height": img_h},
        "inference_time_ms": inference_time,
        "tiling_used": use_tiling,
        "num_detections": len(detections),
        "detections": detections,
        "annotated_image_base64": annotated_image_b64,
    }
@app.get("/training/status")
async def training_status():
    """
    Retourne le rapport d'entraînement en cours (lu depuis le fichier JSON).
    Le script train.py met à jour ce fichier à chaque epoch.
    """
    metrics_file = REPORTS_DIR / "training_metrics.json"
    if not metrics_file.exists():
        return {"status": "no_training", "message": "Aucun entraînement en cours ou terminé."}
    try:
        with open(metrics_file, "r") as f:
            report = json.load(f)
        return report
    except json.JSONDecodeError:
        return {"status": "error", "message": "Fichier de rapport corrompu."}
@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """
    WebSocket pour les rapports d'entraînement en temps réel.
    Le frontend peut se connecter ici pour recevoir les mises à jour.
    """
    await websocket.accept()
    metrics_file = REPORTS_DIR / "training_metrics.json"
    last_content = ""
    try:
        while True:
            # Lire le fichier de métriques
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    content = f.read()
                # Envoyer seulement si le contenu a changé
                if content != last_content:
                    last_content = content
                    try:
                        data = json.loads(content)
                        await websocket.send_json(data)
                    except json.JSONDecodeError:
                        pass
                # Arrêter si l'entraînement est terminé
                try:
                    data = json.loads(content)
                    if data.get("status") == "completed":
                        await websocket.send_json({"status": "completed", "message": "Entraînement terminé"})
                        break
                except json.JSONDecodeError:
                    pass
            # Attendre avant de re-vérifier (polling toutes les 2 secondes)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        print("🔌 WebSocket client déconnecté")
# ============================================================
# Démarrage
# ============================================================
@app.on_event("startup")
async def startup():
    """Charge le modèle au démarrage si disponible."""
    model_path = Path(DEFAULT_MODEL_PATH)
    if model_path.exists():
        print(f"🚀 Chargement du modèle au démarrage : {model_path}")
        load_model(str(model_path))
    else:
        print(f"⚠️  Modèle par défaut introuvable : {model_path}")
        print("   Utilisez POST /models/load pour charger un modèle.")
def main():
    parser = argparse.ArgumentParser(description="API YOLO Segmentation")
    global DEFAULT_MODEL_PATH
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    # Mettre à jour le chemin du modèle par défaut
    DEFAULT_MODEL_PATH = args.model
    state.device = args.device
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,  # 1 worker pour éviter de charger le modèle plusieurs fois
    )
if __name__ == "__main__":
    main()
