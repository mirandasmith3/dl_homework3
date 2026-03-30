import torch
import torch.nn as nn

from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Hyperparameters
    # -----------------------
    batch_size = 64
    lr = 1e-3
    num_epochs = 10

    # -----------------------
    # Data
    # -----------------------
    train_loader = load_data(
        "classification_data/train",
        transform_pipeline="aug",
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = load_data(
        "classification_data/val",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False
    )


    # -----------------------
    # Model, loss, optimizer
    # -----------------------
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        metric = AccuracyMetric()

        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                preds = model.predict(images)
                metric.update(preds.cpu(), labels.cpu())

        acc = metric.compute()

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    # -----------------------
    # Save model
    # -----------------------
    save_model(model)
    print("Model saved!")


if __name__ == "__main__":
    train()