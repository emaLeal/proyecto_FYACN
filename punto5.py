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

class CNNClassifier(nn.Module):
    def __init__(self, output_size, hidden_layers=[256], drop_p=0.5):
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
        
        self.hidden_layers = nn.ModuleList()
        input_size = 128 * 16 * 16  
        
        for hidden_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
            
        self.output = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        # Forward convolucional
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        # Forward capas ocultas con dropout
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        x = self.output(x)
        return F.log_softmax(x, dim=1)

# --- Función de validación (estilo Part 5) ---
def validation(model, val_loader, criterion):
    val_loss = 0
    accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            
            ps = torch.exp(outputs)  # Convertir de log-softmax a probabilidades
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return val_loss, accuracy

# --- Configuración principal ---
dataset_dir = "data"
csv_path = os.path.join(dataset_dir, "pokemon.csv")
image_dir = os.path.join(dataset_dir, "images")

# Cargar y preparar datos
df = pd.read_csv(csv_path)
df["type"] = df["Type1"]  # Usa solo Type1
types = sorted(df["type"].dropna().unique())
type_to_idx = {t: i for i, t in enumerate(types)}
idx_to_type = {i: t for t, i in type_to_idx.items()}
types_spanish = translate_types(type_to_idx)

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Crear datasets y dataloaders
full_dataset = PokemonDataset(df, image_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# --- Configuración del modelo ---
hidden_layers = [512, 256]  # Capas ocultas configurables
drop_p = 0.5                # Probabilidad de dropout
output_size = len(types)
model = CNNClassifier(output_size, hidden_layers=hidden_layers, drop_p=drop_p)

# Manejo de pesos para clases desbalanceadas
labels = df["type"].map(type_to_idx)
class_counts = Counter(labels)
total = max(class_counts.values())
weights = torch.tensor([total / class_counts[i] for i in range(len(type_to_idx))], dtype=torch.float32)
criterion = nn.NLLLoss(weight=weights)  

optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 30
print_every = 10  # Mostrar resultados cada X batches
model_path = "pokemon_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Modelo cargado desde disco. Saltando entrenamiento.")
    skip_training = True
else:
    skip_training = False

if not skip_training:
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        for batch_i, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Validación periódica
            if batch_i % print_every == 0:
                val_loss, val_accuracy = validation(model, val_loader, criterion)
                
                print(f"Época: {epoch+1}/{epochs}.. "
                      f"Batch: {batch_i}/{len(train_loader)}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "
                      f"Val Loss: {val_loss/len(val_loader):.3f}.. "
                      f"Val Accuracy: {val_accuracy/len(val_loader):.3f}")
                
                running_loss = 0
                model.train()

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

# Inferencia (estilo Part 5)
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    ps = torch.exp(output)  # Convertir log-softmax a probabilidades
    top_p, top_class = ps.topk(1, dim=1)
    predicted_type = idx_to_type[top_class.item()]

print(f"\nPokémon elegido: {pokemon_name}")
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