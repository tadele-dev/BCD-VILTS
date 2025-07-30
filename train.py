import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from bcdvilts_model import BCDVILTSModel  

# --------------------------
# CONFIGURATION
# --------------------------
CSV_PATH = 'train.csv'  
IMAGE_DIR = 'bcd-dataset/images'  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'bcdvilts_trained.pth'

# --------------------------
# Our DATASET DEFINITION
# --------------------------
class BCDFusionDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_id'])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = row['text']
        label = int(row['label'])
        return image, text, label

# --------------------------
# Our TRANSFORMS AND TOKENIZER
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --------------------------
# LOADING THE DATASET
# --------------------------
dataset = BCDFusionDataset(CSV_PATH, IMAGE_DIR, tokenizer, transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------
# INITIALIZE MODEL
# --------------------------
model = BCDVILTSModel()
model.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------
# OUR TRAINING LOOP
# --------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for images, texts, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Tokenize text
        text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

        # Forward pass
        outputs = model(images, texts)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

# --------------------------
# SAVEING OUR MODEL
# --------------------------
torch.save(model.state_dict(), My_MODEL_SAVE_PATH)
print(f"Model saved to {MY_MODEL_SAVE_PATH}")
