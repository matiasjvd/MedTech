import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import optuna
import sys

sys.path.append("C:/Users/Matias/Desktop/Tesis/Tesis-Codes/utils")
import creacion_df_torch as qol

# Definición de la red neuronal
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
        for _ in range(3):
            output_size = (output_size - kernel_size + 2 * 1) / 1 + 1
            output_size = output_size // 2

        output_features = int(output_size * output_size * num_filters3)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Función objetivo para Optuna
def objective(trial, cancer='all'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = "C:/Users/Matias/Desktop/Tesis/Dataset_Neoplasias"
    df = qol.crear_dataframe_multiclase(dataset_dir, filtro_cancer=cancer)
    num_classes = df['etiqueta'].nunique()
    
    num_filters1 = trial.suggest_categorical('num_filters1', [16, 32, 64])
    num_filters2 = trial.suggest_categorical('num_filters2', [64, 128, 256])
    num_filters3 = trial.suggest_categorical('num_filters3', [64, 128, 256])
    kernel_size = trial.suggest_int('kernel_size', 3, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])  # Reducir tamaño del lote
    steps_per_epoch = trial.suggest_int('steps_per_epoch', 100, 200)

    train_loader, val_loader, test_loader = qol.prepare_data_loaders(df, batch_size, m_type='multiclase')

    model = Net(num_filters1, num_filters2, num_filters3, kernel_size, dropout_rate, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

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
            # Liberar memoria de la GPU
            del images, labels, output
            torch.cuda.empty_cache()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Liberar memoria de la GPU
            del images, labels, outputs
            torch.cuda.empty_cache()

    accuracy = correct / total
    return accuracy

