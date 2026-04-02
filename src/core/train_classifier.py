import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
AUG_ROT_DIR = ROOT / "data" / "processed " / "aug_rot"

CLASSES = ["cacahuatl", "teocomitl", "tepetla", "tlapexohuiloni"]

# ── Hyperparamètres ───────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_EPOCHS  = 60
NUM_WORKERS = 4
VAL_RATIO   = 0.2
LR          = 3e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset personnalisé pour aug_rot ─────────────────────────────────────────
class AugRotDataset(Dataset):
    """Lit les images depuis aug_rot — dossiers nommés 'NNNN-classname'."""

    def __init__(self, aug_rot_root: Path, classes: list, transform=None):
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.samples = []

        for folder in sorted(aug_rot_root.iterdir()):
            if not folder.is_dir():
                continue
            class_name = folder.name.split("-", 1)[-1]
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            for img_path in sorted(folder.glob("*.png")):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def make_weighted_sampler(dataset, indices):
    """WeightedRandomSampler : chaque classe a la même probabilité d'être tirée."""
    labels = [dataset.samples[i][1] for i in indices]
    class_counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def main():
    full_dataset = AugRotDataset(AUG_ROT_DIR, CLASSES, transform=train_transform)
    val_dataset  = AugRotDataset(AUG_ROT_DIR, CLASSES, transform=val_transform)

    print(f"Total images : {len(full_dataset)}")
    for cls, idx in full_dataset.class_to_idx.items():
        count = sum(1 for _, l in full_dataset.samples if l == idx)
        print(f"  {cls} ({idx}) : {count} images")

    # Split train / val
    n_val   = int(len(full_dataset) * VAL_RATIO)
    n_train = len(full_dataset) - n_val
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    from torch.utils.data import Subset
    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(val_dataset,  val_idx)

    # ── Anti-biais : WeightedRandomSampler ────────────────────────────────
    sampler = make_weighted_sampler(full_dataset, train_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── Modèle ───────────────────────────────────────────────────────────────
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    # ── Anti-biais : label smoothing dans CrossEntropyLoss ────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    out_dir = ROOT / "models" / "clf_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path    = out_dir / "resnet18_best.pt"
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(model(inputs), 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        train_acc = correct / total

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                _, preds = torch.max(model(inputs), 1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step()

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | loss={running_loss/total:.4f} "
              f"acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                        "class_to_idx":    full_dataset.class_to_idx}, best_path)
            print(f"  ✅ Sauvegardé (val_acc={val_acc:.3f})")

    print(f"\nMeilleure val_acc : {best_val_acc:.4f}")
    print(f"Poids : {best_path}")


if __name__ == "__main__":
    main()
