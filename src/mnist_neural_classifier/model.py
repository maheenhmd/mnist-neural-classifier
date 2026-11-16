import torch
from torch import nn


# recognize handwritten digits from the MNIST dataset
class MNISTMLP(nn.Module):

    def __init__(self):
        super().__init__()

        # Define the layers through which images will pass
        self.network = nn.Sequential(

            # Convert each 28x28 image into 784 pixel values
            nn.Flatten(),

            # Input: 784 pixels -> Hidden: 128 neurons
            nn.Linear(784, 128),

            # Add non-linearity to learn complex patterns
            nn.ReLU(),

            # Hidden: 128 neurons -> Output: 10 digit scores
            nn.Linear(128, 10)
        )

    # Define how input data moves through the network
    def forward(self, x):
        return self.network(x)


# Run this block only when model.py is executed directly
# It will not run when another file imports MNISTMLP
if __name__ == "__main__":

    # Create a model with initially random weights
    model = MNISTMLP()

    # Create 32 fake images to test the model's input/output shapes
    # Shape: (batch_size, channels, height, width)
    dummy_images = torch.randn(32, 1, 28, 28)

    # Pass images through the neural network (forward pass)
    output = model(dummy_images)

    # Verify that the model produces 10 scores per image
    print("Input shape:", dummy_images.shape)
    print("Output shape:", output.shape)
