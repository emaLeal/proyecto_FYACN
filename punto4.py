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
import helper  # Solo si lo usas

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

        return image, label

# --- Red convolucional ---
class CNNClassifier(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# --- Configuración y datos ---
dataset_dir = "data"
csv_path = os.path.join(dataset_dir, "pokemon.csv")
image_dir = os.path.join(dataset_dir, "images")

df = pd.read_csv(csv_path)
df["type"] = df["Type1"]  # Usa solo Type1
types = sorted(df["type"].dropna().unique())
type_to_idx = {t: i for i, t in enumerate(types)}
idx_to_type = {i: t for t, i in type_to_idx.items()}
types_spanish = translate_types(type_to_idx)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_dataset = PokemonDataset(df, image_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# --- Modelo ---
output_size = len(types)
model = CNNClassifier(output_size)

model_path = "pokemon_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Modelo cargado desde disco. Saltando entrenamiento.")
    skip_training = True
else:
    skip_training = False

# --- Pérdida ponderada ---
labels = df["type"].map(type_to_idx)
class_counts = Counter(labels)
total = max(class_counts.values())
weights = [total / class_counts[i] for i in range(len(type_to_idx))]
weights = torch.tensor(weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 30

if not skip_training:
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validación
        val_loss = 0
        val_accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                ps = F.softmax(outputs, dim=1)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class.squeeze() == labels
                val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Época {epoch+1}/{epochs}.. "
              f"Loss Entrenamiento: {running_loss:.3f}.. "
              f"Loss Validación: {val_loss:.3f}.. "
              f"Precisión Validación: {val_accuracy / len(val_loader):.3f}")

    torch.save(model.state_dict(), model_path)
    print("Modelo entrenado y guardado.")

# --- Evaluación con imagen aleatoria ---
pokemon_name = choose_pokemon(df)
pokemon_row = df[df["Name"] == pokemon_name].iloc[0]
true_type = pokemon_row["type"]
true_type_idx = type_to_idx[true_type]

filename = pokemon_name.lower().replace(" ", "-") + ".png"
img_path = os.path.join(image_dir, filename)
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(image_tensor)
    loss = criterion(output, torch.tensor([true_type_idx]))
    ps = F.softmax(output, dim=1)
    top_p, top_class = ps.topk(1, dim=1)
    predicted_type = idx_to_type[top_class.item()]

print(f"Loss de evaluación para {pokemon_name}: {loss.item():.4f}")
print(f"Pokémon elegido: {pokemon_name}")
print(f"Tipo real: {true_type} ({true_type_idx})")
print(f"Predicción del tipo: {predicted_type}")

# --- Visualización ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(image)
ax1.axis('off')
ax1.set_title(f"{pokemon_name}")

colors = ['skyblue'] * len(types_spanish)
colors[true_type_idx] = 'green'
ax2.bar(types_spanish, ps.numpy()[0], color=colors)
ax2.set_ylim(0, 1)
ax2.set_xticklabels(types_spanish, rotation=45, ha="right")
ax2.set_ylabel("Probabilidad")
ax2.set_title("Predicción de Tipo Pokémon")
plt.tight_layout()
plt.show()
