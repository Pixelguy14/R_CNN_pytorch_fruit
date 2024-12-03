import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
from datasetVOC import VOCDataset, transform
from modelVOC import get_model_instance_segmentation

# Paths
data_dir = 'dataset/images'
ann_dir = 'dataset/annotations/voc_annotations'

# Define the class names
classes = ['__background__', 'manzana', 'platano', 'carambola', 'guayaba', 'kiwi', 'mango', 'melon', 'naranja', 'durazno', 'pera', 'caqui', 'pitahaya', 'ciruela', 'granada', 'jitomate']

# Load dataset
dataset = VOCDataset(root=data_dir, ann_dir=ann_dir, transform=transform, classes=classes)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

# Define model
num_classes = len(classes)
model = get_model_instance_segmentation(num_classes)

# Move model to device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 1

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                pred_labels = output['labels'][output['scores'] > 0.1].cpu().numpy()  # Lower confidence threshold
                true_labels = target['labels'].cpu().numpy()
                
                if len(pred_labels) > 0 and len(true_labels) >= len(pred_labels):
                    all_preds.extend(pred_labels)
                    all_labels.extend(true_labels[:len(pred_labels)])
                
    if all_preds and all_labels:
        labels_list = list(range(num_classes))
        cm = confusion_matrix(all_labels, all_preds, labels=labels_list)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', labels=labels_list, zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)
        class_report = classification_report(
            all_labels, all_preds, target_names=classes, labels=labels_list, zero_division=0
        )

        print("Confusion Matrix:\n", cm)
        print("Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, Accuracy: {:.4f}".format(precision, recall, fscore, accuracy))
        print("Classification Report:\n", class_report)
    else:
        print("No valid predictions found for evaluation.")
    
    model.train()

for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {losses.item()}")

    # Evaluate after each epoch
    evaluate(model, data_loader, device, num_classes)

# Save the model
torch.save(model.state_dict(), 'model.pth')
