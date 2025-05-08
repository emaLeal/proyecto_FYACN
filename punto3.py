import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter
import torch.nn.functional as F
from collections import OrderedDict
import helper  # Si tienes el helper.view_classify del curso

# --- Función auxiliar para elegir un Pokémon aleatorio ---
def choose_pokemon(data):
    size = len(data)
    return data['Name'][random.randint(0, size - 1)]

# --- Traducción de tipos al español ---
def translate_types(type_dict):
    translated_types = {
        'Ground': 'Tierra', 'Dark': 'Siniestro', 'Fairy': 'Hada',
        'Fighting': 'Lucha', 'Flying': 'Volador', 'Water': 'Agua',
        'Fire': 'Fuego', 'Ghost': 'Fantasma', 'Bug': 'Bicho',
        'Ice': 'Hielo', 'Psychic': 'Psíquico', 'Rock': 'Roca',
        'Steel': 'Acero', 'Grass': 'Planta', 'Normal': 'Normal',
        'Electric': 'Eléctrico', 'Dragon': 'Dragón', 'Poison': 'Veneno'
    }
    return [translated_types[k] for k in type_dict]



# --- Dataset personalizado ---
class PokemonDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.types = sorted(self.df['type'].dropna().unique())
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['Name']
        label = self.type_to_idx[row['type']]
        filename = name.lower().replace(" ", "-") + ".png"
        image_path = os.path.join(self.image_dir, filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image.view(-1), label  # Aplanamos la imagen aquí

# --- Configuración y datos ---
dataset_dir = "data"
csv_path = os.path.join(dataset_dir, "pokemon.csv")
image_dir = os.path.join(dataset_dir, "images")

df = pd.read_csv(csv_path)
df["type"] = df["Type1"]  # <-- Aquí decides usar solo Type1 de forma clara
types = sorted(df["type"].dropna().unique())
type_to_idx = {t: i for i, t in enumerate(types)}
idx_to_type = {i: t for t, i in type_to_idx.items()}
types_spanish = translate_types(type_to_idx)
# --- Preprocesamiento ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = PokemonDataset(df, image_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# --- Modelo ---
input_size = 3 * 128 * 128
hidden_sizes = [512, 256, 128]
output_size = len(types)

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
    ('relu3', nn.ReLU()),
    ('output', nn.Linear(hidden_sizes[2], output_size))
]))

# Si el modelo ya fue guardado, lo cargamos
model_path = "pokemon_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Modelo cargado desde disco. Saltando entrenamiento.")
    skip_training = True
else:
    skip_training = False

labels = df["type"].map(type_to_idx)
class_counts = Counter(labels)
total = max(class_counts.values())

# Calcula pesos inversos a la frecuencia
weights = [total / class_counts[i] for i in range(len(type_to_idx))]
weights = torch.tensor(weights, dtype=torch.float32)

# --- Entrenamiento ---
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 30
steps = 0
if not skip_training:
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            steps += 1
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss = 0

    torch.save(model.state_dict(), model_path)
    print("Modelo entrenado y guardado.")

        
# Guardar el modelo entrenado
torch.save(model.state_dict(), "pokemon_model.pth")
print("Modelo guardado como pokemon_model.pth")
# --- Validación con imagen aleatoria ---
pokemon_name = choose_pokemon(df)
pokemon_row = df[df["Name"] == pokemon_name].iloc[0]
true_type = pokemon_row["type"]
true_type_idx = type_to_idx[true_type]

filename = pokemon_name.lower().replace(" ", "-") + ".png"
img_path = os.path.join(image_dir, filename)

image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).view(1, -1)

val_loss = 0
val_accuracy = 0
with torch.no_grad():
    output = model(image_tensor)
    loss = criterion(output, torch.tensor([true_type_idx]))  # Etiqueta real
    ps = F.softmax(output, dim=1)
    top_p, top_class = ps.topk(1, dim=1)
    predicted_type = idx_to_type[top_class.item()]

print(f"Loss de evaluación para {pokemon_name}: {loss.item():.4f}")

model.train()
print(f"Pokémon elegido: {pokemon_name}")
print(f"Tipo real: {true_type} ({true_type_idx})")
print(f"Predicción del tipo para {pokemon_name}: {predicted_type}")

# --- Visualización ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(image)
ax1.axis('off')
ax1.set_title(f"Imagen: {pokemon_name.capitalize()}")
colors = ['skyblue'] * len(types_spanish)
colors[true_type_idx] = 'green'  # marca el tipo real
ax2.bar(types_spanish, ps.numpy()[0], color=colors)
ax2.set_ylim(0, 1)
ax2.set_xticklabels(types_spanish, rotation=45, ha="right")
ax2.set_ylabel("Probabilidad")
ax2.set_title("Predicción de Tipo Pokémon")
plt.tight_layout()
plt.show()