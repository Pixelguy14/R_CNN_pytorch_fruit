'''
2024
ÁNGEL EMMANUEL URBINA NAVARRETE
JOSÉ JULIÁN SIERRA ÁLVAREZ
'''
# Importacion de bibliotecas
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Redimensionar imagenes
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Cargar sets de datos
dataset = torchvision.datasets.ImageFolder(root='/home/pixelguy14/Documentos/SemAgo-Dic-2024/Topicos_Sistemas_Comp/Proyecto_Final/clean_dataset', transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Definicion de un modelo de CNN (red neuronal convolucional)
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 81)  # Numero de clases = 81 'aguacate', 'ajo', 'almendra india', 'arandano', 'arandano azul', 'berenjena', 'cacahuate', 'cacao', 'cafe', 'calabacin', 'calabaza', 'calabaza amarga', 'calabaza moscada', 'camote', 'caqui', 'carambola', 'cebolla', 'cereza', 'cereza negra', 'chile', 'ciruela', 'ciruela de playa', 'coco', 'coliflor', 'durazno', 'durian', 'espinaca', 'frambuesa', 'fresa', 'granada', 'granadilla', 'granadina', 'grosella negra', 'guanabana', 'guayaba', 'guisante', 'higo', 'higo del desierto', 'jalapeno', 'jengibre', 'jitomate', 'jocote', 'kiwano', 'kiwi', 'lechuga', 'lima', 'limon americano', 'lychee', 'maiz', 'mamey', 'mandarina', 'mango', 'manzana', 'melon', 'melon galia', 'mora artica', 'nabo', 'naranja', 'nuez de brazil', 'papa', 'papaya', 'paprika', 'pepino', 'pera', 'pera yali', 'pimiento', 'pina', 'pitahaya', 'platano', 'pomelo', 'rabano', 'remolacha', 'repollo', 'sandia', 'soja', 'toronja', 'tuna', 'uva', 'vainilla', 'zanahoria', 'zarzamora'

# Crear un directorio para guardar las imagenes de las matrices de confusion
output_dir = 'confusion_matrices'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Entrenamiento principal
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
continue_training = True

while continue_training:
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        all_labels = []
        all_preds = []
        
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
        
        avg_loss = epoch_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        class_report = classification_report(all_labels, all_preds, target_names=dataset.classes)
        
        print(f"Epoca {epoch + 1}, Perdida Promedio: {avg_loss:.4f}")
        print(f"Exactitud: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")
        print(f"Reporte de Clasificacion:\n{class_report}")
        
        # Calcular la matriz de confusion
        cm = confusion_matrix(all_labels, all_preds)
        
        # Graficar y guardar la matriz de confusion como imagen
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Matriz de Confusión - Época {epoch + 1}')
        plt.savefig(os.path.join(output_dir, f'cf_{epoch + 1}.png'))
        plt.close()
        
        user_input = input("¿Quieres entrenar otra epoca? (si/no): ").strip().lower()
        if user_input != 'si':
            continue_training = False
            break
    if continue_training:
        print("Entrenando otra epoca...")
    
# Guardar el modelo entrenado
torch.save(model.state_dict(), 'classification_model.pth')
print("Modelo guardado exitosamente.")
