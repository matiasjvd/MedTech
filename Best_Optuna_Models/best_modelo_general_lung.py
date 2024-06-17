import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
sys.path.append("C:/Users/Matias/Desktop/Tesis/Tesis-Codes/utils")
dataset_dir = r"C:/Users/Matias/Desktop/Tesis/Dataset_Neoplasias"

# Definición del modelo
class Net(nn.Module):
    def __init__(self, num_filters1, num_filters2, num_filters3, kernel_size, dropout_rate, num_classes, input_size=256):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters1, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters1, num_filters2, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters2, num_filters3, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        output_size = input_size
        for _ in range(3):  # Dado que tenemos tres bloques de Conv + MaxPool
            output_size = (output_size - kernel_size + 2 * 1) / 1 + 1  # Aplicamos la fórmula del tamaño de salida
            output_size = output_size // 2  # MaxPooling divide el tamaño por 2

        output_features = int(output_size * output_size * num_filters3)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_features, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(device, train_loader, val_loader):
    num_classes = 2  # Actualizar según el número de clases
    model = Net(num_filters1=16, num_filters2=256, num_filters3=512, kernel_size=5, dropout_rate=0.388875361181391, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00015436358741558522)
    criterion = nn.NLLLoss()

    # Entrenamiento
    model.train()
    for epoch in range(10):
        steps = 0
        for images, labels in train_loader:
            if steps >= 182:  # steps_per_epoch
                break
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluación en conjunto de validación
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    return model