import torch
import torchvision.transforms as transforms
from PIL import Image

# Load class names
classes = ['aguacate', 'ajo', 'almendra india', 'arandano', 'arandano azul', 'berenjena', 'cacahuate', 'cacao', 'cafe', 'calabacin', 'calabaza', 'calabaza amarga', 'calabaza moscada', 'camote', 'caqui', 'carambola', 'cebolla', 'cereza', 'cereza negra', 'chile', 'ciruela', 'ciruela de playa', 'coco', 'coliflor', 'durazno', 'durian', 'espinaca', 'frambuesa', 'fresa', 'granada', 'granadilla', 'granadina', 'grosella negra', 'guanabana', 'guayaba', 'guisante', 'higo', 'higo del desierto', 'jalapeno', 'jengibre', 'jitomate', 'jocote', 'kiwano', 'kiwi', 'lechuga', 'lima', 'limon americano', 'lychee', 'maiz', 'mamey', 'mandarina', 'mango', 'manzana', 'melon', 'melon galia', 'mora artica', 'nabo', 'naranja', 'nuez de brazil', 'papa', 'papaya', 'paprika', 'pepino', 'pera', 'pera yali', 'pimiento', 'pina', 'pitahaya', 'platano', 'pomelo', 'rabano', 'remolacha', 'repollo', 'sandia', 'soja', 'toronja', 'tuna', 'uva', 'vainilla', 'zanahoria', 'zarzamora']

# Load the TorchScript model
model = torch.jit.load('mobile_classification_model.ptl')
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Ensure this matches your training size
    transforms.ToTensor(),
])

def get_prediction(img_path):
    img = Image.open(img_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0)  # Add batch dimension
    print(f"Image {img_path} transformed successfully.")
    
    with torch.no_grad():
        output = model(img_transformed)
        _, predicted = torch.max(output, 1)
        label_idx = predicted.item()
        label = classes[label_idx]

    print(f"Predicted Label for {img_path}: {label}")
    return img, label

# Example usage
img_path = './data/testset/Image_10 (8).jpg'
#img_path = 'testset/Peach0011.png'
#img_path = 'testset/Banana08.png'
img, label = get_prediction(img_path)
print(f"Prediction: {label}")

# Optionally display the image with the prediction
img.show()
