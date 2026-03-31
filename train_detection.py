import torch
import torch.nn as nn

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Hyperparameters
    # -----------------------
    batch_size = 32
    lr = 3e-3
    num_epochs = 40

    # -----------------------
    # Data
    # -----------------------
    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = load_data(
        "drive_data/val",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
    )

    # -----------------------
    # Model, loss, optimizer
    # -----------------------
    model = Detector().to(device)

    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            depth_labels = batch["depth"].to(device)
            seg_labels = batch["track"].to(device).long()

            optimizer.zero_grad()

            logits, depth_preds = model(images)

            seg_loss = seg_criterion(logits, seg_labels)
            depth_loss = depth_criterion(depth_preds, depth_labels)
            loss = 3 * seg_loss + depth_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        metric = DetectionMetric()

        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                depth_labels = batch["depth"].to(device)
                seg_labels = batch["track"].to(device).long()

                preds, depth_preds = model.predict(images)
                metric.add(
                    preds.cpu(),
                    seg_labels.cpu(),
                    depth_preds.cpu(),
                    depth_labels.cpu(),
                )

        results = metric.compute()
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
            f"IOU: {results['iou']:.4f} | "
            f"Depth: {results['abs_depth_error']:.4f} | "
            f"Lane Depth: {results['tp_depth_error']:.4f}"
        )

    # -----------------------
    # Save model
    # -----------------------
    save_model(model)
    print("Model saved!")


if __name__ == "__main__":
    train()