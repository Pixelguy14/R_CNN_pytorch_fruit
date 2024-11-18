import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from modelVOC import get_model_instance_segmentation
import matplotlib.pyplot as plt

# Load class names
classes = ['__background__', 'manzana', 'platano', 'carambola', 'guayaba', 'kiwi', 'mango', 'melon', 'naranja', 'durazno', 'pera', 'caqui', 'pitahaya', 'ciruela', 'granada', 'jitomate']

# Load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = len(classes)
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.to(device)
model.eval()

print("Model loaded successfully.")

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_prediction(img_path, threshold):
    img = Image.open(img_path).convert("RGB")
    img_transformed = transform(img).to(device)
    print("Image transformed successfully.")
    with torch.no_grad():
        prediction = model([img_transformed])
    pred_classes = [classes[i] for i in list(prediction[0]['labels'].cpu().numpy())]
    pred_scores = list(prediction[0]['scores'].cpu().numpy())
    pred_bboxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].cpu().numpy())]
    
    print("Predicted Classes: ", pred_classes)
    print("Predicted Scores: ", pred_scores)
    print("Predicted Bounding Boxes: ", pred_bboxes)
    
    pred_t = [pred_scores.index(x) for x in pred_scores if x > threshold]
    if pred_t:
        pred_t = pred_t[-1]
        pred_bboxes = pred_bboxes[:pred_t+1]
        pred_classes = pred_classes[:pred_t+1]
    else:
        pred_bboxes, pred_classes = [], []
    return img, pred_bboxes, pred_classes

def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=3)
        draw.text(box[0], label, fill="red")
    return image

# Test the model on a new image
test_image_path = 'testset/Orange0011.png' 
threshold = 0.1  # Confidence threshold
img, boxes, labels = get_prediction(test_image_path, threshold)
print("Boxes and labels drawn.")
img_with_boxes = draw_boxes(img, boxes, labels)

# Save the image with bounding boxes
output_path = 'output_testset/output_image.png'
img_with_boxes.save(output_path)
print(f"Image saved to {output_path}")

# Display the image using matplotlib
plt.imshow(img_with_boxes)
plt.axis('off')
plt.show()
