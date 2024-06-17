import torch
torch.cuda.memory_cached()
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import optuna
import sys
sys.path.append("C:/Users/Matias/Desktop/Tesis/Tesis-Codes/utils")
import creacion_df_torch as qol
dataset_dir = r"C:/Users/Matias/Desktop/Tesis/Dataset_Neoplasias"



class Net(nn.Module):
    def __init__(self, num_filters1, num_filters2, num_filters3, num_filters4, kernel_size, dropout_rate, num_classes, input_size=256):
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
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters3, num_filters4, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        output_size = input_size
        for _ in range(4):  # Dado que tenemos dos bloques de Conv + MaxPool
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
 


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2  # Actualizar según el número de clases
    df = qol.crear_dataframe_binario(dataset_dir, filtro_cancer= 'oral')
    # Hiperparámetros
    num_filters1 = trial.suggest_categorical('num_filters1', [32, 64, 128])
    num_filters2 = trial.suggest_categorical('num_filters2', [128, 256, 512])
    num_filters3 = trial.suggest_categorical('num_filters3', [128, 256, 512])
    num_filters4 = trial.suggest_categorical('num_filters4', [128, 256, 512])
    kernel_size = trial.suggest_int('kernel_size', 3, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [20, 40, 80])
    steps_per_epoch = trial.suggest_int('steps_per_epoch', 100, 300)

    train_loader, val_loader, test_loader = qol.prepare_data_loaders(df, batch_size, m_type = 'bin')

    model = Net(num_filters1, num_filters2,num_filters3,num_filters4, kernel_size, dropout_rate, num_classes).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy