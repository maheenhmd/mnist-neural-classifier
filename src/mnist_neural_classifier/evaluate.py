
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_neural_classifier.model import MNISTMLP

# 1. Load test dataset
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

# 2. Load trained model
model = MNISTMLP()
model.load_state_dict(
    torch.load("checkpoints/best_model.pt", weights_only=True)
)
model.eval()

# 3. Build confusion matrix
confusion = torch.zeros(10, 10, dtype=torch.int64)

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images).argmax(dim=1)

        for actual, predicted in zip(labels, predictions):
            confusion[actual, predicted] += 1

# 4. Visualize and save
fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(confusion.numpy(), cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel("Predicted Digit")
ax.set_ylabel("Actual Digit")
ax.set_title("MNIST Confusion Matrix")

for i in range(10):
    for j in range(10):
        ax.text(j, i, str(confusion[i, j].item()),
                ha="center", va="center", fontsize=7)

Path("outputs").mkdir(exist_ok=True)
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close(fig)

print("Confusion matrix saved to outputs/confusion_matrix.png")
print("Correct:", confusion.diag().sum().item())
print("Total:", confusion.sum().item())





# 5. Collect 10 incorrect predictions
wrong_examples = []

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images).argmax(dim=1)

        for image, actual, predicted in zip(images, labels, predictions):
            if actual != predicted:
                wrong_examples.append((
                    image.squeeze().numpy(),
                    actual.item(),
                    predicted.item()
                ))

            if len(wrong_examples) == 10:
                break

        if len(wrong_examples) == 10:
            break

# 6. Visualize incorrect predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for ax, (image, actual, predicted) in zip(
    axes.flat, wrong_examples
):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Actual: {actual} | Predicted: {predicted}")
    ax.axis("off")

fig.suptitle("10 Incorrect MNIST Predictions")
fig.tight_layout()

plt.savefig("outputs/incorrect_predictions.png", dpi=150)
plt.close(fig)

print("Saved 10 incorrect predictions.")
