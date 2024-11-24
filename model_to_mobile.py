import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile

# Define your model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18(weights=None)  # Load without pretrained weights
        # Adjust the final layer to your number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 81)  # Adjust according to the number of classes

    def forward(self, x):
        return self.model(x)

# Create the model instance
model = MyModel()

# Load the state dictionary
state_dict = torch.load('classification_model.pth', map_location=torch.device('cpu'), weights_only=True)

# Adjust the keys if they are missing a prefix or need other modifications
adjusted_state_dict = {}
for key in state_dict.keys():
    new_key = f"model.{key}" if key not in model.state_dict() else key
    adjusted_state_dict[new_key] = state_dict[key]

# Load the adjusted state dictionary into the model
model.load_state_dict(adjusted_state_dict, strict=False)

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor
example = torch.rand(1, 3, 100, 100)  # Ensure this size matches your training input size

# Trace the model
traced_script_module = torch.jit.script(model)

# Optimize for mobile
traced_script_module_optimized = optimize_for_mobile(traced_script_module)

# Save the optimized model
traced_script_module_optimized._save_for_lite_interpreter("mobile_classification_model.ptl")
print("Modelo guardado como TorchScript optimizado exitosamente.")
