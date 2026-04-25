import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
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


def build_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def compute_metrics(total_loss, total_samples, tp, fp, fn, correct):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = correct / total_samples if total_samples else 0.0
    loss = total_loss / total_samples if total_samples else 0.0
    return Metrics(loss=loss, accuracy=accuracy, precision=precision, recall=recall)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0

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
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct += (preds == labels).sum().item()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    return compute_metrics(total_loss, total_samples, tp, fp, fn, correct)


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
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        eval_metrics = run_epoch(model, test_loader, criterion, device)
        print(
            "epoch {}/{} | train loss {:.4f} acc {:.4f} prec {:.4f} rec {:.4f} | "
            "test loss {:.4f} acc {:.4f} prec {:.4f} rec {:.4f}".format(
                epoch,
                args.epochs,
                train_metrics.loss,
                train_metrics.accuracy,
                train_metrics.precision,
                train_metrics.recall,
                eval_metrics.loss,
                eval_metrics.accuracy,
                eval_metrics.precision,
                eval_metrics.recall,
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


if __name__ == "__main__":
    main()
