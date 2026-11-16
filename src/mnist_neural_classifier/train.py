
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import MNISTMLP


# 1. Load MNIST dataset
dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor() # converts images to PyTorch tensors when accessed
)

# 2. Split dataset
train_data, val_data = random_split(
    dataset,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42)
)

# 3. Prepare batches
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False
)

# 4. Create model, loss function and optimizer
model = MNISTMLP()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# print the number of training and validation images
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

# save training history and best validation loss for later use
history = []
best_val_loss = float("inf")
# where to save the best model and training metrics
Path("checkpoints").mkdir(exist_ok=True)


# 5. Training and validation loop
for epoch in range(3):

    # TRAINING
    model.train()
    total_loss = 0.0
    correct = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (
            predictions.argmax(dim=1) == labels
        ).sum().item()

    train_loss = total_loss / len(train_data)
    train_accuracy = correct / len(train_data)

    print(
        f"Epoch {epoch + 1} | "
        f"Training Loss: {train_loss:.4f} | "
        f"Training Accuracy: {train_accuracy:.2%}"
    )

    # VALIDATION
    model.eval()
    val_correct = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            predictions = model(images)
            loss = loss_fn(predictions, labels)

            val_loss_sum += loss.item() * len(labels)
            val_correct += (
                predictions.argmax(dim=1) == labels
            ).sum().item()

    val_loss = val_loss_sum / len(val_data)
    val_accuracy = val_correct / len(val_data)

    print(
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.2%}"
    )

    # RECORD RESULTS
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    # SAVE BEST MODEL
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pt"
        )


# 6. Save configuration and metrics AFTER training
config = {
    "epochs": 3,
    "batch_size": 64,
    "learning_rate": 0.01
}

Path("config.json").write_text(
    json.dumps(config, indent=2)
)

Path("metrics.json").write_text(
    json.dumps(history, indent=2)
)

print("Model, configuration and metrics saved.")




# Load an image the model has not trained on
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

image, actual_digit = test_data[7]

model.eval()
with torch.no_grad():
    scores = model(image.unsqueeze(0))
    predicted_digit = scores.argmax(dim=1).item()

print("Actual digit:", actual_digit)
print("Model prediction:", predicted_digit)