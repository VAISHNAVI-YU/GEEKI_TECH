import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.model_selection import train_test_split

# 1. Load Data
base_dir = r"C:\Users\Vinatha\Desktop\indiana_cxr_multimodal"
img_dir = os.path.join(base_dir, "images", "images_normalized")
reports_path = os.path.join(base_dir, "indiana_reports.csv")
proj_path = os.path.join(base_dir, "indiana_projections.csv")

reports_df = pd.read_csv(reports_path)
proj_df = pd.read_csv(proj_path)
df = pd.merge(proj_df, reports_df, on='uid', how='inner')

# 2. NLP Entity Extraction Using stable public NER model
print("Extracting clinical entities using stable NER model...")
ner_pipeline = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

def extract_entities(text):
    try:
        ents = ner_pipeline(str(text))
        return [ent["word"] for ent in ents]
    except Exception as e:
        print(f"Error in NLP extraction: {e}")
        return []

df["entities"] = df["impression"].apply(extract_entities)
print("Sample extracted entities:")
print(df[["impression", "entities"]].head(2))

# 3. Generate Label
def extract_pneumonia_label(text):
    return 1 if "pneumonia" in str(text).lower() else 0
df['label'] = df['findings'].apply(extract_pneumonia_label)

# 4. Split Train/Test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# --- SAMPLE DATA FOR FAST TRAINING ---
train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# 5. Tokenizer and Transforms
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. Custom Dataset Including Entities
class IndianaMultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer, img_transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.img_transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)

        report_text = str(row['impression'])
        encoding = self.tokenizer(
            report_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        label = int(row['label'])
        entities = " ".join(row['entities'])[:200]
        return image, input_ids, attention_mask, label, entities


print("Creating datasets...")
train_dataset = IndianaMultimodalDataset(train_df, img_dir, tokenizer, img_transform)
test_dataset = IndianaMultimodalDataset(test_df, img_dir, tokenizer, img_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# 7. Define the Multimodal Model with fixed forward method
class MultimodalNet(nn.Module):
    def __init__(self, text_model_name='emilyalsentzer/Bio_ClinicalBERT'):
        super().__init__()
        self.img_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, 128)

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.txt_fc = nn.Linear(self.text_encoder.config.hidden_size, 128)

        self.entity_fc = nn.Linear(self.text_encoder.config.hidden_size, 64)

        self.classifier = nn.Linear(128 + 128 + 64, 1)

    def forward(self, image, input_ids, attention_mask, entity_emb):
        img_feat = self.img_encoder(image)                     # [batch, 128]
        txt_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = self.txt_fc(txt_outputs.last_hidden_state[:, 0, :])  # [batch, 128]

        entity_feat = self.entity_fc(entity_emb)                # Project entity embedding to 64 dims

        fused = torch.cat([img_feat, txt_feat, entity_feat], dim=1)    # [batch, 320]
        out = self.classifier(fused)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Helper function to encode extracted entities
def encode_entities(entity_texts):
    encoding = tokenizer(entity_texts, return_tensors='pt', padding=True, truncation=True, max_length=64)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model.text_encoder(**encoding)
    return outputs.last_hidden_state[:, 0, :]  # shape [batch_size, 768]

# 8. Training Loop with batch limit for faster iterations
num_epochs = 3
max_batches = 50  # Limit batches per epoch to speed up training

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, input_ids, attention_mask, labels, entities) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        entity_emb = encode_entities(list(entities))

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask, entity_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / (batch_idx + 1):.4f}")

    # Evaluation on test set with batch limit
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (images, input_ids, attention_mask, labels, entities) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            entity_emb = encode_entities(list(entities))

            outputs = model(images, input_ids, attention_mask, entity_emb)
            preds = (torch.sigmoid(outputs) > 0.5).long().cpu()

            correct += (preds.squeeze() == labels.cpu()).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")

print("Training complete.")
