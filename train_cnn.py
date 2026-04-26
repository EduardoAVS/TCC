import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SimpleFrameDataset


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    fpr: float


@dataclass
class EpochResult:
    metrics: Metrics
    labels: list
    preds: list


def build_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def compute_metrics(total_loss, total_samples, labels, preds, positive_scores):
    loss = total_loss / total_samples if total_samples else 0.0
    accuracy = float(accuracy_score(labels, preds))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(labels, positive_scores))
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return Metrics(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        auc=auc,
        fpr=fpr,
    )


def write_metrics_table(output_path, rows):
    fieldnames = ["epoch", "split", "loss", "accuracy", "precision", "recall", "f1_score", "auc", "fpr"]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_loss_curve(output_path, train_losses, test_losses):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker="o", label="train loss")
    plt.plot(epochs, test_losses, marker="o", label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(output_path, labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_labels = ["negative", "positive"]
    tick_marks = [0, 1]
    plt.xticks(tick_marks, tick_labels)
    plt.yticks(tick_marks, tick_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    labels_list = []
    preds_list = []
    positive_scores = []

    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        labels_list.extend(labels.detach().cpu().tolist())
        preds_list.extend(preds.detach().cpu().tolist())
        positive_scores.extend(probs.detach().cpu().tolist())

    metrics = compute_metrics(total_loss, total_samples, labels_list, preds_list, positive_scores)
    return EpochResult(metrics=metrics, labels=labels_list, preds=preds_list)


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight CNN prototype on SO-TAD.")
    parser.add_argument("--root", required=True, help="Dataset root containing train/, test/ and optionally Appendix.txt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--max-frames-per-video", type=int, default=24)
    parser.add_argument("--train-video-limit", type=int, default=None)
    parser.add_argument("--test-video-limit", type=int, default=None)
    parser.add_argument("--output-dir", default="runs/simple_cnn")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    transform = build_transforms(args.image_size)
    train_dataset = SimpleFrameDataset(
        root=args.root,
        mode="train",
        transform=transform,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        video_limit=args.train_video_limit,
    )
    test_dataset = SimpleFrameDataset(
        root=args.root,
        mode="test",
        transform=transform,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        video_limit=args.test_video_limit,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("device:", device)
    print("train samples:", len(train_dataset))
    print("test samples:", len(test_dataset))

    best_acc = -1.0
    metrics_rows = []
    train_losses = []
    test_losses = []
    metrics_path = os.path.join(args.output_dir, "metrics_table.csv")
    loss_curve_path = os.path.join(args.output_dir, "loss_curve.png")
    confusion_matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        eval_result = run_epoch(model, test_loader, criterion, device)
        train_metrics = train_result.metrics
        eval_metrics = eval_result.metrics
        train_losses.append(train_metrics.loss)
        test_losses.append(eval_metrics.loss)
        metrics_rows.extend([
            {
                "epoch": epoch,
                "split": "train",
                "loss": train_metrics.loss,
                "accuracy": train_metrics.accuracy,
                "precision": train_metrics.precision,
                "recall": train_metrics.recall,
                "f1_score": train_metrics.f1_score,
                "auc": train_metrics.auc,
                "fpr": train_metrics.fpr,
            },
            {
                "epoch": epoch,
                "split": "test",
                "loss": eval_metrics.loss,
                "accuracy": eval_metrics.accuracy,
                "precision": eval_metrics.precision,
                "recall": eval_metrics.recall,
                "f1_score": eval_metrics.f1_score,
                "auc": eval_metrics.auc,
                "fpr": eval_metrics.fpr,
            },
        ])
        write_metrics_table(metrics_path, metrics_rows)
        save_loss_curve(loss_curve_path, train_losses, test_losses)
        save_confusion_matrix(confusion_matrix_path, eval_result.labels, eval_result.preds)
        print(
            "epoch {}/{} | "
            "train loss {:.4f} acc {:.4f} prec {:.4f} rec {:.4f} f1 {:.4f} auc {:.4f} fpr {:.4f} | "
            "test loss {:.4f} acc {:.4f} prec {:.4f} rec {:.4f} f1 {:.4f} auc {:.4f} fpr {:.4f}".format(
                epoch,
                args.epochs,
                train_metrics.loss,
                train_metrics.accuracy,
                train_metrics.precision,
                train_metrics.recall,
                train_metrics.f1_score,
                train_metrics.auc,
                train_metrics.fpr,
                eval_metrics.loss,
                eval_metrics.accuracy,
                eval_metrics.precision,
                eval_metrics.recall,
                eval_metrics.f1_score,
                eval_metrics.auc,
                eval_metrics.fpr,
            )
        )

        if eval_metrics.accuracy > best_acc:
            best_acc = eval_metrics.accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_accuracy": best_acc,
                },
                os.path.join(args.output_dir, "best.pt"),
            )

    print("metrics table saved to:", metrics_path)
    print("loss curve saved to:", loss_curve_path)
    print("confusion matrix saved to:", confusion_matrix_path)


if __name__ == "__main__":
    main()
