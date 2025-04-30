import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import OrderedDict

#Función que retorna un pokemon aleatorio
def choose_pokemon(data):
    size = len(data)
    return data['Name'][random.randint(0, size)]

def translate_types(type):
    translated_types = {
        'Ground': 'Tierra',
        'Dark': 'Siniestro',
        'Fairy': 'Hada',
        'Fighting': 'Lucha',
        'Flying': 'Volador',
        'Water': 'Agua',
        'Fire': 'Fuego',
        'Ghost': 'Fantasma',
        'Bug': 'Bicho',
        'Ice': 'Hielo',
        'Psychic': 'Psiquico',
        'Rock': 'Roca',
        'Steel': 'Acero',
        'Grass': 'Planta',
        'Normal': 'Normal',
        'Electric': 'Electrico',
        'Dragon': 'Dragon',
        'Poison': 'Veneno'
    }
    new_types = []
    for i in type:
        new_types.append(translated_types[i])
    return new_types

# Rutas de Relevancia
dataset_dir = "data"  
csv_path = os.path.join(dataset_dir, "pokemon.csv")
image_dir = os.path.join(dataset_dir, "images")

# 2. Leer CSV y preparar etiquetas
df = pd.read_csv(csv_path)
type_column = [col for col in df.columns if 'type' in col.lower()][0]
df["type"] = df[type_column]



types = sorted(df["type"].dropna().unique())
type_to_idx = {t: i for i, t in enumerate(types)}
idx_to_type = {i: t for t, i in type_to_idx.items()}
types_spanish = translate_types(type_to_idx)


# Carga Pokemon
pokemon_name = choose_pokemon(df)
filename = pokemon_name.lower().replace(" ", "-") + ".png"
img_path = os.path.join(image_dir, filename)

if not os.path.exists(img_path):
    raise FileNotFoundError(f"No se encontró la imagen: {img_path}")

# 4. Preprocesamiento
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).view(1, -1)

# 5. Modelo (sin entrenar)
input_size = 3 * 28 * 28
hidden_sizes = [400, 200, 100]
output_size = len(types)

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

# 6. Inferencia
with torch.no_grad():
    output = model(image_tensor).squeeze()  # [18]

print(model)

# 7. Visualización: imagen + gráfico de predicciones
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# Mostrar imagen
ax1.imshow(image)
ax1.axis('off')
ax1.set_title(f"Imagen: {pokemon_name.capitalize()}")

# Mostrar barras de probabilidad
ax2.bar(types, output.numpy(), color='skyblue')
ax2.set_ylim(0, 1)
ax2.set_xticklabels(types_spanish, rotation=45, ha="right")
ax2.set_ylabel("Probabilidad")
ax2.set_title("Predicción de Tipo Pokémon")

plt.tight_layout()
plt.show()
