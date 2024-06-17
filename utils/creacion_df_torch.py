import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, to_pil_image

# Crear un DataFrame a partir de las imágenes en el directorio
def crear_dataframe_binario(dataset_dir, filtro_cancer = 'all'):
    data = {'ruta': [], 'etiqueta': [], 'tipo_cancer': []}
    for neoplasia in ['ACA', 'SCC', 'benigno']:
        etiqueta = 'maligno' if neoplasia in ['ACA', 'SCC'] else 'benigno'
        neoplasia_dir = os.path.join(dataset_dir, neoplasia)
        for filename in os.listdir(neoplasia_dir):
            ruta_completa = os.path.join(neoplasia_dir, filename)

            #Definimos el tipo de cancer por si quisieramos entrenarlo con algun tipo particular
            parts = filename.split('_')
            # Determinar el tipo de cáncer y la etiqueta basado en el nombre del archivo
            if len(parts) > 2 and parts[0] == 'oral':
                tipo_cancer = parts[0]  # 'oral'
            elif len(parts) == 3:
                tipo_cancer, neoplasia_foto, _ = parts  # Formato antiguo
            else:
                print(f"Archivo {filename} ignorado por tener un formato incorrecto.")
                continue

            data['ruta'].append(ruta_completa)
            data['etiqueta'].append(etiqueta)
            data['tipo_cancer'].append(tipo_cancer)
            
    data = pd.DataFrame(data)
    if filtro_cancer != 'all':
        data = data[data['tipo_cancer']==filtro_cancer]

    return data



# Clase para cargar imágenes y etiquetas desde DataFrame
class ImageDatasetBin(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['ruta']
        label = 1 if self.df.iloc[index]['etiqueta'] == 'maligno' else 0
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Función para crear un DataFrame desde el directorio del dataset
def crear_dataframe_multiclase(dataset_dir, filtro_cancer = 'all'):
    data = {'ruta': [], 'etiqueta': [], 'tipo_cancer': []}
    for categoria in os.listdir(dataset_dir):  # Asume que cada subcarpeta es una categoría
        categoria_dir = os.path.join(dataset_dir, categoria)
        for filename in os.listdir(categoria_dir):
            parts = filename.split('_')
            
            # Determinar el tipo de cáncer y la etiqueta basado en el nombre del archivo
            if len(parts) > 2 and parts[0] == 'oral':
                tipo_cancer = parts[0]  # 'oral'
                etiqueta = parts[1]     # 'benigno' o 'maligno'
            elif len(parts) == 3:
                tipo_cancer, etiqueta, _ = parts  # Formato antiguo
            else:
                print(f"Archivo {filename} ignorado por tener un formato incorrecto.")
                continue

            ruta_completa = os.path.join(categoria_dir, filename)
            data['ruta'].append(ruta_completa)
            data['etiqueta'].append(etiqueta)
            data['tipo_cancer'].append(tipo_cancer)
    if filtro_cancer != 'all':
        data = data[data['tipo_cancer'] == filtro_cancer]
        
    return pd.DataFrame(data)


# Clase del dataset para cargar imágenes y etiquetas desde el DataFrame
class ImageDatasetMul(Dataset):
    def __init__(self, dataframe, transform=None, class_to_idx=None):
        self.df = dataframe
        self.transform = transform
        self.class_to_idx = class_to_idx or {v: k for k, v in enumerate(sorted(dataframe['etiqueta'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['ruta']
        label = self.class_to_idx[self.df.iloc[index]['etiqueta']]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    


#_______________________________________________ACA SE DEFINEN LAS FUNCIONES QUE SIRVEN PARA AMBOS DF_______________________________________________

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def prepare_data_loaders(df, batch_size=20, m_type = 'bin'):
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=46, stratify=df['etiqueta'])
    val_df, test_df = train_test_split(test_val_df, test_size=0.33, random_state=46, stratify=test_val_df['etiqueta'])

    if m_type == 'bin':
        train_ds = ImageDatasetBin(train_df, transform=get_transforms())
        val_ds = ImageDatasetBin(val_df, transform=get_transforms())
        test_ds = ImageDatasetBin(test_df, transform=get_transforms())

    else:
        train_ds = ImageDatasetMul(train_df, transform=get_transforms())
        val_ds = ImageDatasetMul(val_df, transform=get_transforms())
        test_ds = ImageDatasetMul(test_df, transform=get_transforms())


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def evaluate_model(device, model, test_loader):
    # Evaluación en conjunto de prueba
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy




#_______________________________________________ACA SE DEFINEN LAS FUNCIONES QUE MUESTRA EL TRABAJO DE LA RED NEURONAL_______________________________________________

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_image, class_idx=None):
        self.model.eval()
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        input_image.requires_grad = True

        output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        weights = np.mean(gradients, axis=(2, 3))[0]
        grad_cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights):
            grad_cam += w * activations[0, i, :, :]

        grad_cam = np.maximum(grad_cam, 0)
        grad_cam = cv2.resize(grad_cam, (input_image.shape[2], input_image.shape[3]))

        if np.max(grad_cam) != 0:
            grad_cam = grad_cam - np.min(grad_cam)
            grad_cam = grad_cam / np.max(grad_cam)
        else:
            grad_cam = np.zeros_like(grad_cam)

        return grad_cam

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    return image