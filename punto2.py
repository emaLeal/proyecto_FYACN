# ✅ Importar librerías necesarias
import numpy as np
import torch
import matplotlib.pyplot as plt
import helper
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict


# 1. Preprocesamiento de datos
# Convertimos las imágenes a tensores y las normalizamos a valores entre -1 y 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Cargar el dataset MNIST (imágenes de dígitos escritos a mano)
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)



# 3. Obtener un lote de datos para probar la red
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Construir la red neuronal con 3 capas ocultas
# Estructura: 784 → 400 → 200 → 100 → 10
input_size = 784                  # 28x28 píxeles = 784 entradas
hidden_sizes = [400, 200, 100]    # Tres capas ocultas con distintas unidades
output_size = 10                  # 10 clases (dígitos del 0 al 9)

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),  # Capa 1: entrada → 400
    ('relu1', nn.ReLU()),                              # Activación ReLU

    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),  # Capa 2: 400 → 200
    ('relu2', nn.ReLU()),

    ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),  # Capa 3: 200 → 100
    ('relu3', nn.ReLU()),

    ('output', nn.Linear(hidden_sizes[2], output_size)),   # Salida: 100 → 10
    ('softmax', nn.Softmax(dim=1))                         # Activación Softmax
]))

# 5. Imprimir arquitectura de la red
print(model)

# 6. Hacer una predicción con una imagen
# Aplanar la imagen de [1, 28, 28] → [1, 784]
images.resize_(images.shape[0], 1, 784)

# Seleccionamos una sola imagen del batch para probar
ps = model.forward(images[0, :])  # Probabilidades predichas para cada clase

# 7. Mostrar la imagen y las probabilidades de predicción
helper.view_classify(images[0].view(1, 28, 28), ps)
