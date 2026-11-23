
import torch

from mnist_neural_classifier.model import MNISTMLP


def test_output_shape():
    model = MNISTMLP()

    images = torch.randn(4, 1, 28, 28)
    outputs = model(images)

    assert outputs.shape == (4, 10)


def test_deterministic_inference():
    torch.manual_seed(42)

    model = MNISTMLP()
    model.eval()

    image = torch.ones(1, 1, 28, 28)

    with torch.no_grad():
        output1 = model(image)
        output2 = model(image)

    assert torch.equal(output1, output2)
