'''
2024
ÁNGEL EMMANUEL URBINA NAVARRETE
JOSÉ JULIÁN SIERRA ÁLVAREZ
'''

# Importacion de bibliotecas
import torch
import torchvision
import torchvision.transforms as transforms
import torch_directml  # Agregar torch-directml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Configurar el dispositivo DirectML
dml_device = torch_directml.device()
print("Usando el dispositivo:", dml_device)

# Redimensionar imagenes
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Cargar sets de datos
dataset = torchvision.datasets.ImageFolder(root='./data/clean_dataset', transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Definicion de un modelo de CNN (red neuronal convolucional)
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 81)  # Numero de clases = 81
model = model.to(dml_device)  # Mover el modelo al dispositivo DirectML

# Crear un directorio para guardar las imagenes de las matrices de confusion
output_dir = 'confusion_matrices'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Entrenamiento principal
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10
continue_training = True

while continue_training:
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        all_labels = []
        all_preds = []
        aux = 0
        
        for images, labels in data_loader:
            # Mover tensores al dispositivo DirectML
            images = images.to(dml_device)
            labels = labels.to(dml_device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())  # Mover a CPU para métricas
            all_preds.extend(preds.cpu().numpy())
            aux += 1
            if aux % 100 == 1:
                print('Hecho:', aux)
        
        avg_loss = epoch_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        class_report = classification_report(all_labels, all_preds, target_names=dataset.classes)
        
        print(f"Época {epoch + 1}, Pérdida Promedio: {avg_loss:.4f}")
        print(f"Exactitud: {accuracy:.4f}")
        print(f"Precisión: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")
        print(f"Reporte de Clasificación:\n{class_report}")

        # Guardar el modelo entrenado
        torch.save(model.state_dict(), f'classification_model_epoch_{epoch + 1}.pth')
        print(f'Modelo {epoch + 1} guardado exitosamente.')
        
        # Calcular la matriz de confusión
        cm = confusion_matrix(all_labels, all_preds)
        
        # Graficar y guardar la matriz de confusión como imagen
        plt.figure(figsize=(40, 32))  # Modificar para aumentar tamaño de matriz de confusión
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Matriz de Confusión - Época {epoch + 1}')
        plt.savefig(os.path.join(output_dir, f'cf_{epoch + 1}.png'))
        plt.close()

        end_time = time.time()  # Registrar el final de la iteración
        iteration_time = end_time - start_time  # Calcular el tiempo transcurrido
        print(f"Época terminada. Tiempo transcurrido: {iteration_time:.4f} s")
        
        user_input = 'si'
        if user_input != 'si' and (epoch + 1) < 12:
            continue_training = False
            break
if continue_training:
    print("Entrenando otra época...")
