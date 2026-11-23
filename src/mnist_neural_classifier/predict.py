
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_neural_classifier.model import MNISTMLP

# Load 10,000 test images
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Divide test images into batches of 100
test_loader = DataLoader(
    test_data,
    batch_size=100,
    shuffle=False
)

# Create model and load trained weights
model = MNISTMLP()

model.load_state_dict(
    torch.load("checkpoints/best_model.pt", weights_only=True)
)

model.eval()

# Evaluate all test images
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        scores = model(images)
        predictions = scores.argmax(dim=1)

        correct += (predictions == labels).sum().item()

# Calculate overall test accuracy
total = len(test_data)
incorrect = total - correct
accuracy = correct / total

print("Total test images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", incorrect)
print(f"Test accuracy: {accuracy:.2%}")
