from ultralytics import YOLO

# Charger ton modèle entraîné
model = YOLO("runs/detect/train3/weights/best.pt")

# Faire une prédiction sur ton image test
results = model.predict(
    source="test_images",
    conf=0.25,
    iou=0.5,
    save=True,
    show=True
)

# Résumé des résultats AVEC identification automatique
for r in results:
    print("\n📸 Image :", r.path)

    # Si rien n'a été détecté
    if r.boxes.cls.numel() == 0:
        print("⚠️ Aucun élément reconnu")
        continue

    # On récupère la prédiction la plus confiante
    best_idx = r.boxes.conf.argmax()
    class_id = int(r.boxes.cls[best_idx])
    confidence = float(r.boxes.conf[best_idx])
    class_name = r.names[class_id]

    threshold = 0.8

    if confidence < threshold:
        print(f"⚠️ Unsure prediction: {class_name} ({confidence * 100:.1f}%)")
    else:
        print(f"✅ Identified element: {class_name} ({confidence * 100:.1f}%)")