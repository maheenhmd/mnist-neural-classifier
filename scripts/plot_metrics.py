
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Load saved training history
history = json.loads(
    Path("outputs/metrics.json").read_text()
)

epochs = [row["epoch"] for row in history]

train_loss = [row["train_loss"] for row in history]
val_loss = [row["val_loss"] for row in history]

train_acc = [row["train_accuracy"] * 100 for row in history]
val_acc = [row["val_accuracy"] * 100 for row in history]

# Create two graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss graph
axes[0].plot(epochs, train_loss, marker="o", label="Training")
axes[0].plot(epochs, val_loss, marker="o", label="Validation")
axes[0].set_title("Loss vs Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Accuracy graph
axes[1].plot(epochs, train_acc, marker="o", label="Training")
axes[1].plot(epochs, val_acc, marker="o", label="Validation")
axes[1].set_title("Accuracy vs Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()

fig.tight_layout()
fig.savefig("outputs/training_curves.png", dpi=150)
plt.close(fig)

print("Training graphs saved successfully!")
