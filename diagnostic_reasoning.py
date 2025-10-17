import os
import random
import glob
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from torch.nn.functional import cosine_similarity

# ====================================================
# DEVICE SETUP
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Using device:", device)

# ====================================================
# PATH SETUP
# ====================================================
base_dir = r"C:\Users\Vinatha\Desktop\indiana_cxr_multimodal_project"

# ✅ Real dataset folders
img_dir_dcm = os.path.join(base_dir, "images", "images_normalized")  # for .dcm.png
img_dir_jpeg_train = os.path.join(base_dir, "pneumonia_classifier", "train")  # for .jpeg
img_dir_jpeg_test = os.path.join(base_dir, "pneumonia_classifier", "test")   # for .jpeg

# Combine all possible image folders
img_dirs = [img_dir_dcm, img_dir_jpeg_train, img_dir_jpeg_test]

reports_path = os.path.join(base_dir, "indiana_cxr_multimodal", "indiana_reports.csv")
proj_path = os.path.join(base_dir, "indiana_cxr_multimodal", "indiana_projections.csv")

# ====================================================
# LOAD CSV DATA
# ====================================================
reports_df = pd.read_csv(reports_path)
proj_df = pd.read_csv(proj_path)
df = pd.merge(proj_df, reports_df, on='uid', how='inner')

df['label'] = df['findings'].apply(lambda x: 1 if "pneumonia" in str(x).lower() else 0)
print(f"✅ Merged dataset: {len(df)} samples")

# ====================================================
# TOKENIZER & IMAGE TRANSFORM
# ====================================================
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====================================================
# CUSTOM DATASET (RECURSIVE IMAGE SEARCH)
# ====================================================
class IndianaMultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dirs, tokenizer, img_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.tokenizer = tokenizer
        self.img_transform = img_transform

        # Recursively gather all image paths
        self.all_images = []
        for d in self.img_dirs:
            for ext in ["*.jpeg", "*.jpg", "*.png"]:
                self.all_images += glob.glob(os.path.join(d, "**", ext), recursive=True)

        if len(self.all_images) == 0:
            raise FileNotFoundError("❌ No image files found in any provided directory!")

        print(f"📸 Total available images: {len(self.all_images)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row.get("filename", ""))

        found_path = None
        for img_path in self.all_images:
            if filename in img_path:
                found_path = img_path
                break

        if not found_path:
            found_path = random.choice(self.all_images)
            # comment out if you don't want console spam
            print(f"⚠️ Using fallback image: {os.path.basename(found_path)}")

        try:
            image = Image.open(found_path).convert("RGB")
        except:
            image = Image.new("RGB", (128, 128), color=(0, 0, 0))

        if self.img_transform:
            image = self.img_transform(image)

        text = str(row.get("impression", ""))
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=128, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        entities = str(row.get("findings", ""))[:200]
        label = int(row.get("label", 0))
        uid = row.get("uid", idx)

        return image, input_ids, attention_mask, label, entities, uid

# ====================================================
# DATALOADER SETUP
# ====================================================
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = IndianaMultimodalDataset(train_df, img_dirs, tokenizer, img_transform)
test_dataset = IndianaMultimodalDataset(test_df, img_dirs, tokenizer, img_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print(f"📊 Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# ====================================================
# MODEL DEFINITION
# ====================================================
class MultimodalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, 128)
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.txt_fc = nn.Linear(self.text_encoder.config.hidden_size, 128)
        self.entity_fc = nn.Linear(self.text_encoder.config.hidden_size, 128)
        self.classifier = nn.Linear(128 + 128 + 128, 1)

    def forward(self, image, input_ids, attention_mask, entity_emb):
        img_feat = self.img_encoder(image)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = self.txt_fc(txt_out.last_hidden_state[:, 0, :])
        ent_feat = self.entity_fc(entity_emb)
        fused = torch.cat([img_feat, txt_feat, ent_feat], dim=1)
        return self.classifier(fused)

# ====================================================
# TRAINING SETUP
# ====================================================
model = MultimodalNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

def encode_entities(entity_texts):
    enc = tokenizer(entity_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.text_encoder(**enc)
    return out.last_hidden_state[:, 0, :]

# ====================================================
# TRAIN LOOP
# ====================================================
epochs = 2
print("\n🚀 Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, input_ids, attention_mask, labels, entities, _ in train_loader:
        images, input_ids, attention_mask, labels = (
            images.to(device),
            input_ids.to(device),
            attention_mask.to(device),
            labels.float().to(device),
        )

        entity_emb = encode_entities(list(entities))
        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask, entity_emb)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "multimodal_model_trained_final.pth")
print("\n✅ Model trained and saved as multimodal_model_trained_final.pth")

# ====================================================
# INFERENCE (DIAGNOSTIC REASONING)
# ====================================================
model.eval()
results = []
anomaly_threshold = 0.3
anomaly_count = 0

def rank_diagnoses(entities, pred_prob):
    ent = entities.lower()
    score = pred_prob
    if "tumor" in ent:
        score += 0.1
    if "lymph" in ent or "mass" in ent or "nodule" in ent:
        score += 0.1
    if "infection" in ent or "pneumonia" in ent:
        score += 0.05
    return min(score, 1.0)

with torch.no_grad():
    for images, input_ids, attention_mask, labels, entities, uids in test_loader:
        images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
        entity_emb = encode_entities(list(entities))
        outputs = model(images, input_ids, attention_mask, entity_emb)
        probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
        probs = [probs] if probs.ndim == 0 else probs

        txt_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = model.txt_fc(txt_outputs.last_hidden_state[:, 0, :])
        ent_feat = model.entity_fc(entity_emb)
        sim = cosine_similarity(txt_feat, ent_feat, dim=1)
        anomalies = (sim < anomaly_threshold)
        anomaly_count += anomalies.sum().item()

        for ent, prob, uid, label in zip(list(entities), probs, uids, labels):
            score = rank_diagnoses(ent, prob)
            results.append({
                "uid": str(uid),
                "entities": ent,
                "probability": float(prob),
                "ranked_score": float(score),
                "label": int(label)
            })

# Save results
pd.DataFrame(results).to_csv("diagnostic_reasoning_results_final.csv", index=False)
print(f"\n✅ Diagnostic results saved to diagnostic_reasoning_results_final.csv")
print(f"🧩 Total anomalies detected (cosine similarity < {anomaly_threshold}): {anomaly_count}")

top_cases = sorted(results, key=lambda x: x['ranked_score'], reverse=True)[:5]
print("\n🔍 Top 5 High-Risk Cases:")
for c in top_cases:
    print(f"UID: {c['uid']} | Score: {c['ranked_score']:.3f} | Entities: {c['entities'][:70]}")
