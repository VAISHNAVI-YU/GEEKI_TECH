# ====================================================
# anomaly_detection_standalone.py
# ====================================================
import os
import glob
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from sklearn.model_selection import train_test_split
import torch.nn as nn

# ====================================================
# DEVICE SETUP
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Using device:", device)

# ====================================================
# PATH SETUP
# ====================================================
base_dir = r"C:\Users\Vinatha\Desktop\indiana_cxr_multimodal_project"

# Paths from your project
reports_path = os.path.join(base_dir, "indiana_cxr_multimodal", "indiana_reports.csv")
proj_path = os.path.join(base_dir, "indiana_cxr_multimodal", "indiana_projections.csv")

# Image directories
img_dir_dcm = os.path.join(base_dir, "images", "images_normalized")
img_dir_train = os.path.join(base_dir, "pneumonia_classifier", "train")
img_dir_test = os.path.join(base_dir, "pneumonia_classifier", "test")
img_dirs = [img_dir_dcm, img_dir_train, img_dir_test]

# ====================================================
# TOKENIZER & TRANSFORM
# ====================================================
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====================================================
# DATASET CLASS (same as before)
# ====================================================
class IndianaMultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dirs, tokenizer, img_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.tokenizer = tokenizer
        self.img_transform = img_transform

        self.all_images = []
        for d in self.img_dirs:
            for ext in ["*.jpeg", "*.jpg", "*.png"]:
                self.all_images += glob.glob(os.path.join(d, "**", ext), recursive=True)
        if len(self.all_images) == 0:
            raise FileNotFoundError("❌ No image files found in provided directories!")
        print(f"📸 Found {len(self.all_images)} total images")

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
            found_path = self.all_images[idx % len(self.all_images)]

        try:
            image = Image.open(found_path).convert("RGB")
        except:
            image = Image.new("RGB", (128, 128))

        if self.img_transform:
            image = self.img_transform(image)

        text = str(row.get("impression", ""))
        encoding = tokenizer(text, padding="max_length", truncation=True,
                             max_length=128, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        entities = str(row.get("findings", ""))[:200]
        label = int(row.get("label", 0))
        uid = row.get("uid", idx)
        return image, input_ids, attention_mask, label, entities, uid

# ====================================================
# LOAD DATA & SPLIT
# ====================================================
reports_df = pd.read_csv(reports_path)
proj_df = pd.read_csv(proj_path)
df = pd.merge(proj_df, reports_df, on='uid', how='inner')
df['label'] = df['findings'].apply(lambda x: 1 if "pneumonia" in str(x).lower() else 0)

_, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
test_dataset = IndianaMultimodalDataset(test_df, img_dirs, tokenizer, img_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ====================================================
# MODEL DEFINITION (same architecture)
# ====================================================
class MultimodalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, 128)
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.txt_fc = nn.Linear(self.text_encoder.config.hidden_size, 128)
        self.entity_fc = nn.Linear(self.text_encoder.config.hidden_size, 64)  # match old model
        self.classifier = nn.Linear(128 + 128 + 64, 1)  # total 320

    def forward(self, image, input_ids, attention_mask, entity_emb):
        img_feat = self.img_encoder(image)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = self.txt_fc(txt_out.last_hidden_state[:, 0, :])
        ent_feat = self.entity_fc(entity_emb)
        fused = torch.cat([img_feat, txt_feat, ent_feat], dim=1)
        return self.classifier(fused)


# ====================================================
# LOAD TRAINED MODEL
# ====================================================
model_path = os.path.join(base_dir, "multimodal_model.pth")
model = MultimodalNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Loaded trained model from {model_path}")

# ====================================================
# HELPER: ENCODE ENTITY TEXTS
# ====================================================
def encode_entities(entity_texts):
    enc = tokenizer(entity_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.text_encoder(**enc)
    return out.last_hidden_state[:, 0, :]

# ====================================================
# ANOMALY DETECTION LOOP
# ====================================================
print("\n🔍 Running anomaly detection...")

anomaly_threshold = 0.3
anomaly_count = 0

with torch.no_grad():
    for images, input_ids, attention_mask, labels, entities, uids in test_loader:
        images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

        entity_emb = encode_entities(list(entities))
        ent_feat = model.entity_fc(entity_emb)
        txt_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = model.txt_fc(txt_outputs.last_hidden_state[:, 0, :])

        # 🔧 Dimension alignment (128 → 64)
        if txt_feat.size(1) != ent_feat.size(1):
            txt_feat_proj = torch.nn.functional.linear(txt_feat, torch.eye(ent_feat.size(1), txt_feat.size(1)).to(device))
        else:
            txt_feat_proj = txt_feat

        sim = cosine_similarity(txt_feat_proj, ent_feat, dim=1)
        anomalies = (sim < anomaly_threshold)
        anomaly_count += anomalies.sum().item()

print(f"✅ Total anomalies detected (cosine sim < {anomaly_threshold}): {anomaly_count}")
print("🚀 Anomaly detection pipeline complete!")
