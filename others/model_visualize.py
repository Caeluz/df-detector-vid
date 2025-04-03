from torchsummary import summary
import torch
from torchviz import make_dot
from step_04_deepfake_data import DeepFakeDetector

# Load the model

if __name__ == "__main__":
    model = DeepFakeDetector()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 128, 128).to(device)

    # Perform a forward pass to get the model output
    output = model(dummy_input)

    # Visualize the model
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('model_architecture')
