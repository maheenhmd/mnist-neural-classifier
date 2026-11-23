
import argparse

import torch
from torchvision import datasets, transforms

from mnist_neural_classifier.model import MNISTMLP


def predict_index(index: int):
    # Load MNIST test images
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    if not 0 <= index < len(test_data):
        raise ValueError("Index must be between 0 and 9999")

    image, actual = test_data[index]

    # Load trained model
    model = MNISTMLP()
    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pt",
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    # Predict one image
    with torch.inference_mode():
        scores = model(image.unsqueeze(0))
        predicted = scores.argmax(dim=1).item()

    return actual, predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=7)
    args = parser.parse_args()

    actual, predicted = predict_index(args.index)

    print(f"Actual digit: {actual}")
    print(f"Predicted digit: {predicted}")
