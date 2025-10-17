import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Paths
base_dir = r"C:\Users\Vinatha\Desktop\indiana_cxr_multimodal"
img_dir = os.path.join(base_dir, "images", "images_normalized")
reports_path = os.path.join(base_dir, "indiana_reports.csv")
proj_path = os.path.join(base_dir, "indiana_projections.csv")

# Load CSVs and print columns
reports_df = pd.read_csv(reports_path)
proj_df = pd.read_csv(proj_path)
print('\nReports columns:', reports_df.columns)
print('Projections columns:', proj_df.columns)

# Merge
df = pd.merge(proj_df, reports_df, on='uid', how='inner')

# Print samples and available images
print('\nFirst 10 filenames in CSV:', df['filename'].head(10).tolist())
image_files = [os.path.basename(f) for f in os.listdir(img_dir) if f.endswith('.png')]
print('First 10 images in folder:', image_files[:10])
image_set = set(image_files)

# Filter for existing images only
df = df[df['filename'].apply(lambda x: x in image_set)].reset_index(drop=True)
print(f'Number of samples after filtering for existing images: {len(df)}')

if len(df) == 0:
    print("No image filenames in CSV matched the images in your folder! Please check your file extensions and naming.")
    exit()

# --- 4. Label construction (do this before train_test_split) ---
def extract_pneumonia_label(text):
    text_combined = str(text).lower()
    return 1 if "pneumonia" in text_combined else 0

df['label'] = df['findings'].apply(extract_pneumonia_label)

# --- 5. Train/Test split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# --- 6. Tokenizer & Transforms ---
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 7. Dataset ---
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
        return image, input_ids, attention_mask, label

# --- 8. Dataloaders ---
train_dataset = IndianaMultimodalDataset(train_df, img_dir, tokenizer, img_transform)
test_dataset = IndianaMultimodalDataset(test_df, img_dir, tokenizer, img_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# --- 9. Test a train batch Output ---
for images, input_ids, attention_mask, labels in train_loader:
    print('Train Images:', images.shape)
    print('Train Input IDs:', input_ids.shape)
    print('Train Attention mask:', attention_mask.shape)
    print('Train Labels:', labels)
    break
