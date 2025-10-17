# multimodal_train.py

import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# =========================================================
# 1️⃣  Environment setup
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# =========================================================
# 2️⃣  Paths (EDIT THESE TWO LINES)
# =========================================================
reports_path = r"C:\Users\Vinatha\Desktop\indiana_cxr_multimodal_project\indiana_cxr_multimodal\indiana_reports.csv"
proj_path    = r"C:\Users\Vinatha\Desktop\indiana_cxr_multimodal_project\indiana_cxr_multimodal\indiana_projections.csv"
img_dir      = r"C:\Users\Vinatha\Desktop\indiana_cxr_multimodal_project\indiana_cxr_multimodal\images\images_normalized"

# =========================================================
# 3️⃣  Load and prepare data
# =========================================================
reports_df = pd.read_csv(reports_path)
proj_df = pd.read_csv(proj_path)
df = pd.merge(proj_df, reports_df, on="uid", how="inner")

# Label extraction (binary pneumonia)
df["label"] = df["findings"].apply(lambda x: 1 if "pneumonia" in str(x).lower() else 0)

# =========================================================
# 4️⃣  Named Entity Recognition (NER)
# =========================================================
print("🔍 Extracting clinical entities...")
ner_pipeline = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

def extract_entities(text):
    try:
        ents = ner_pipeline(str(text))
        return [ent["word"] for ent in ents]
    except Exception as e:
        print(f"NER extraction error: {e}")
        return []

df["entities"] = df["impression"].apply(extract_entities)
print("✅ Entity extraction complete!")

# =========================================================
# 5️⃣  Train/test split
# =========================================================
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# =========================================================
# 6️⃣  Tokenizer + Image Transform
# =========================================================
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================================================
# 7️⃣  Dataset Class
# =========================================================
class IndianaMultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer, img_transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.img_transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])

        if not os.path.exists(img_path):
            print(f"⚠️ Missing image: {img_path}")
            image = torch.zeros(3, 224, 224)
        else:
            image = Image.open(img_path).convert("RGB")
            if self.img_transform:
                image = self.img_transform(image)

        text = str(row["impression"])
        enc = self.tokenizer(text, padding="max_length",
                             truncation=True, max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        entities = " ".join(row["entities"])[:200]
        label = int(row["label"])
        return image, input_ids, attention_mask, label, entities

# =========================================================
# 8️⃣  Dataloaders
# =========================================================
train_dataset = IndianaMultimodalDataset(train_df, img_dir, tokenizer, img_transform)
test_dataset = IndianaMultimodalDataset(test_df, img_dir, tokenizer, img_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# =========================================================
# 9️⃣  Model definition
# =========================================================
class MultimodalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, 128)

        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.txt_fc = nn.Linear(self.text_encoder.config.hidden_size, 128)
        self.entity_fc = nn.Linear(self.text_encoder.config.hidden_size, 64)
        self.classifier = nn.Linear(128 + 128 + 64, 1)

    def forward(self, image, input_ids, attention_mask, entity_emb):
        img_feat = self.img_encoder(image)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = self.txt_fc(txt_out.last_hidden_state[:, 0, :])
        ent_feat = self.entity_fc(entity_emb)
        fused = torch.cat([img_feat, txt_feat, ent_feat], dim=1)
        return self.classifier(fused)

# =========================================================
# 🔟  Model training setup
# =========================================================
model = MultimodalNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def encode_entities(entity_texts):
    enc = tokenizer(entity_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.text_encoder(**enc)
    return out.last_hidden_state[:, 0, :]

# =========================================================
# 🔁  Training loop
# =========================================================
num_epochs = 2
max_batches = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, input_ids, attention_mask, labels, entities) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        if batch_idx >= max_batches:
            break
        images, input_ids = images.to(device), input_ids.to(device)
        attention_mask, labels = attention_mask.to(device), labels.float().unsqueeze(1).to(device)
        entity_emb = encode_entities(list(entities))

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask, entity_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / (batch_idx+1):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (images, input_ids, attention_mask, labels, entities) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            images, input_ids = images.to(device), input_ids.to(device)
            attention_mask, labels = attention_mask.to(device), labels.to(device)
            entity_emb = encode_entities(list(entities))
            outputs = model(images, input_ids, attention_mask, entity_emb)
            preds = (torch.sigmoid(outputs) > 0.5).long().cpu()
            correct += (preds.squeeze() == labels.cpu()).sum().item()
            total += labels.size(0)

    print(f"✅ Test Accuracy: {correct / total:.4f}")

# =========================================================
# 💾 Save model and metrics
# =========================================================
torch.save(model.state_dict(), "multimodal_model.pth")
print("💾 Model saved as multimodal_model.pth")

# Final metrics
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, input_ids, attention_mask, labels, entities in test_loader:
        images, input_ids = images.to(device), input_ids.to(device)
        attention_mask = attention_mask.to(device)
        entity_emb = encode_entities(list(entities))
        outputs = model(images, input_ids, attention_mask, entity_emb)
        preds = (torch.sigmoid(outputs) > 0.5).long().cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
print(f"📊 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print("✅ Training complete.")