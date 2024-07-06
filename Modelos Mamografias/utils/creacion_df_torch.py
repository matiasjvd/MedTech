import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np

def crear_dataframe_binario(dataset_dir):
    data = {'ruta': [], 'etiqueta': []}
    
    for categoria in ['benign', 'malignant']:
        etiqueta = categoria
        categoria_dir = os.path.join(dataset_dir, categoria)
        
        for filename in os.listdir(categoria_dir):
            ruta_completa = os.path.join(categoria_dir, filename)
            
            # Ignorar archivos que no son imágenes
            if not (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))):
                print(f"Archivo {filename} ignorado por no ser una imagen.")
                continue

            data['ruta'].append(ruta_completa)
            data['etiqueta'].append(etiqueta)
    
    data = pd.DataFrame(data)
    return data


# Clase para cargar imágenes y etiquetas desde DataFrame
class ImageDatasetBinGrayscale(Dataset):
    def __init__(self, dataframe, transform=None, grayscale=False):
        self.df = dataframe
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['ruta']
        label = 1 if self.df.iloc[index]['etiqueta'] == 'malignant' else 0
        if self.grayscale:
            image = Image.open(img_path).convert('L')  # Convertir a grayscale
        else:
            image = Image.open(img_path).convert('RGB')  # Convertir a RGB
        
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
#_______________________________________________ACA SE DEFINEN LAS FUNCIONES QUE SIRVEN PARA AMBOS DF_______________________________________________

def get_transforms(grayscale=False):
    if grayscale:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def prepare_data_loaders(df, batch_size=20, grayscale=False):
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=46, stratify=df['etiqueta'])
    val_df, test_df = train_test_split(test_val_df, test_size=0.33, random_state=46, stratify=test_val_df['etiqueta'])

    train_ds = ImageDatasetBinGrayscale(train_df, transform=get_transforms(grayscale=grayscale), grayscale=grayscale)
    val_ds = ImageDatasetBinGrayscale(val_df, transform=get_transforms(grayscale=grayscale), grayscale=grayscale)
    test_ds = ImageDatasetBinGrayscale(test_df, transform=get_transforms(grayscale=grayscale), grayscale=grayscale)

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