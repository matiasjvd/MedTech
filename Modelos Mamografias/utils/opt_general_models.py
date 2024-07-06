import torch
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
import sys
sys.path.append("C:/Users/Matias/Desktop/MedTech/Modelos Mamografias/utils")
import creacion_df_torch as qol

dataset_dir = r"C:/Users/Matias/Desktop/MedTech/Dataset_Consolidado"

class Net(nn.Module):
    def __init__(self, num_filters1, num_filters2, num_filters3, num_filters4, kernel_size, dropout_rate, num_classes, input_size=256):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, num_filters1, kernel_size, padding=1),  # Ajustado para 1 canal
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters1, num_filters2, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters2, num_filters3, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters3, num_filters4, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        output_size = input_size
        for _ in range(4):  # Dado que tenemos cuatro bloques de Conv + MaxPool
            output_size = (output_size - kernel_size + 2 * 1) / 1 + 1  # Aplicamos la fórmula del tamaño de salida
            output_size = output_size // 2  # MaxPooling divide el tamaño por 2

        output_features = int(output_size * output_size * num_filters4)

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

# Función de optimización
def objective(trial):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2  
    df = qol.crear_dataframe_binario(dataset_dir)
    
    # Hiperparámetros
    num_filters1 = trial.suggest_categorical('num_filters1', [32, 64, 128])
    num_filters2 = trial.suggest_categorical('num_filters2', [128, 256, 512])
    num_filters3 = trial.suggest_categorical('num_filters3', [128, 256, 512])
    num_filters4 = trial.suggest_categorical('num_filters4', [128, 256, 512])
    kernel_size = trial.suggest_int('kernel_size', 3, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [20, 40, 60])
    steps_per_epoch = trial.suggest_int('steps_per_epoch', 100, 150)

    train_loader, val_loader, test_loader = qol.prepare_data_loaders(df, batch_size, grayscale=True)

    # Calcular los pesos para la función de pérdida
    class_counts = df['etiqueta'].value_counts()
    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights.values, dtype=torch.float).to(device)

    model = Net(num_filters1, num_filters2, num_filters3, num_filters4, kernel_size, dropout_rate, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(weight=class_weights)

    # Entrenamiento
    model.train()
    for epoch in range(10):
        steps = 0
        for images, labels in train_loader:
            if steps >= steps_per_epoch:
                break
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    # Evaluación
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calcular el F1 score ponderado
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return f1