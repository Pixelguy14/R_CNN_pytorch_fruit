'''
2024
ÁNGEL EMMANUEL URBINA NAVARRETE
JOSÉ JULIÁN SIERRA ÁLVAREZ
'''
# Importacion de bibliotecasimport torch
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# Load class names
classes = ['aguacate', 'ajo', 'almendra india', 'arandano', 'arandano azul', 'berenjena', 'cacahuate', 'cacao', 'cafe', 'calabacin', 'calabaza', 'calabaza amarga', 'calabaza moscada', 'camote', 'caqui', 'carambola', 'cebolla', 'cereza', 'cereza negra', 'chile', 'ciruela', 'ciruela de playa', 'coco', 'coliflor', 'durazno', 'durian', 'espinaca', 'frambuesa', 'fresa', 'granada', 'granadilla', 'granadina', 'grosella negra', 'guanabana', 'guayaba', 'guisante', 'higo', 'higo del desierto', 'jalapeno', 'jengibre', 'jitomate', 'jocote', 'kiwano', 'kiwi', 'lechuga', 'lima', 'limon americano', 'lychee', 'maiz', 'mamey', 'mandarina', 'mango', 'manzana', 'melon', 'melon galia', 'mora artica', 'nabo', 'naranja', 'nuez de brazil', 'papa', 'papaya', 'paprika', 'pepino', 'pera', 'pera yali', 'pimiento', 'pina', 'pitahaya', 'platano', 'pomelo', 'rabano', 'remolacha', 'repollo', 'sandia', 'soja', 'toronja', 'tuna', 'uva', 'vainilla', 'zanahoria', 'zarzamora']

# Load the classification model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load('classification_model.pth'))
model.to(device)
model.eval()

print("Model loaded successfully.")

# Transformation for the test image
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Ensure this matches your training size
    transforms.ToTensor(),
])

def get_prediction(img_path):
    img = Image.open(img_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    print(f"Image {img_path} transformed successfully.")
    
    with torch.no_grad():
        output = model(img_transformed)
        _, predicted = torch.max(output, 1)
        label_idx = predicted.item()
        label = classes[label_idx]

    print(f"Predicted Label for {img_path}: {label}")
    return img, label

def draw_label(image, label):
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), label, fill="red")
    return image

# Create output directory if it does not exist
output_dir = 'output_testset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process all images in the testset folder
testset_dir = 'testset'
for filename in os.listdir(testset_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        test_image_path = os.path.join(testset_dir, filename)
        img, label = get_prediction(test_image_path)
        img_with_label = draw_label(img, label)
        
        # Save the image with label
        output_path = os.path.join(output_dir, f'output_{filename}')
        img_with_label.save(output_path)
        print(f"Image saved to {output_path}")

        # Display the image using matplotlib
        plt.imshow(img_with_label)
        plt.axis('off')
        plt.show()
