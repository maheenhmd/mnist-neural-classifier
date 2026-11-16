from torchvision import datasets, transforms

# Load the original MNIST training dataset
mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor() # image will be converted to a PyTorch tensor when we access it
)

# Inspect one example
image, label = mnist[0]

print("Total images:", len(mnist))
print("Image shape:", image.shape)
print("Correct label:", label)
print("Image data type:", image.dtype)
print("Pixel range:", image.min().item(), image.max().item())
