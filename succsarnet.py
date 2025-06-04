# SuccSarNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score

# read in data
df = pd.read_csv('final_mra.csv')
ids = df['image_id']

df.drop(columns=['Team', 'image_id', 'Type'], inplace=True)

# feature engineering datetime 
df['Date'] = pd.to_datetime(df['Date'], utc=True)

df['Year'] = df['Date'].dt.year

df['Month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)

df['Date_num'] = df['Date'].dt.day

df['Day_of_week_sin'] = np.sin(2 * np.pi * df['Date'].dt.day_of_week / 7)
df['Day_of_week_cos'] = np.cos(2 * np.pi * df['Date'].dt.day_of_week / 7)

df['Hour_sin'] = np.sin(2 * np.pi * df['Date'].dt.hour / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Date'].dt.hour / 24)

df.drop(columns='Date', axis=1, inplace=True)

# binary encoding response 
df['Success'] = df['Success'].map({'Success': 1, 'Failure': 0})
df['Success'].value_counts()

# split for test/val set
success_df = df[df['Success'] == 1]
failure_df = df[df['Success'] == 0]

half = int(0.075*len(df)) // 2

test_val_df = pd.concat([
    success_df.sample(n=half, random_state=599),
    failure_df.sample(n=half, random_state=599)
])

val_df, test_df = train_test_split(test_val_df, test_size=2/3, stratify=test_val_df['Success'], random_state=599)

train_df = df.drop(index=test_val_df.index)

# categorical one hot encoding/numerical scaling 
categorical_cols = df.select_dtypes(include='object').drop(columns=['image_id']).columns.tolist()
numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).drop(columns=['Success']).columns.tolist()

# one hot encoding all categorical cols 
train_cat = pd.get_dummies(train_df[categorical_cols], drop_first=True).reset_index()
val_cat = pd.get_dummies(val_df[categorical_cols], drop_first=True).reset_index()
test_cat = pd.get_dummies(test_df[categorical_cols], drop_first=True).reset_index()

# reindexing val/test set to match train 
val_cat = val_cat.reindex(columns=train_cat.columns, fill_value=0)
test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

# scaling all numeric 
scaler = StandardScaler()
train_num = pd.DataFrame(scaler.fit_transform(train_df[numeric_cols]), columns=numeric_cols)
val_num = pd.DataFrame(scaler.transform(val_df[numeric_cols]), columns=numeric_cols)
test_num = pd.DataFrame(scaler.transform(test_df[numeric_cols]), columns=numeric_cols)

# splitting 
X_train = pd.concat([train_num, train_cat], axis=1).astype('float32')
X_val = pd.concat([val_num, val_cat], axis=1).astype('float32')
X_test = pd.concat([test_num, test_cat], axis=1).astype('float32')

# for reinitializing fusion model in REGAL
X_train.to_csv('x_train.csv', index=False) 

y_train = train_df['Success'].values
y_val = val_df['Success'].values
y_test = test_df['Success'].values

img_train = train_df['image_id'].values
img_val = val_df['image_id'].values
img_test = test_df['image_id'].values

# multimodal set class  
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, labels, img_ids, img_dir, transform_pipeline):
        self.df = dataframe
        self.labels = labels
        self.img_ids = img_ids 
        self.img_dir = img_dir
        self.transform_pipeline = transform_pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, f'{self.img_ids[idx]}.tif')
        image = self.transform_pipeline(Image.open(img_path).convert('RGB'))

        tabular = torch.tensor(row.values, dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return image, tabular, label
    
# image transformation pipeline 
transform_pl = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# data loader 
img_path = 'gis_files'
train_set = MultimodalDataset(X_train, y_train, img_train, img_path, transform_pl)
val_set = MultimodalDataset(X_val, y_val, img_val, img_path, transform_pl)
test_set = MultimodalDataset(X_test, y_test, img_test, img_path, transform_pl)

train_loader = DataLoader(train_set, batch_size = 32)
val_loader = DataLoader(val_set, batch_size = 32)
test_loader = DataLoader(test_set, batch_size = 32) 

# resnet + fusion model 
class FusionModel(nn.Module):
    def __init__(self, tabular_input_dim):
        super().__init__()

        # resnet
        resnet = models.resnet50(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  
        self.cnn_out_dim = resnet.fc.in_features
        self.img_embedding = nn.Linear(self.cnn_out_dim, 128)

        # fusion mlp 
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + tabular_input_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False), 
            nn.Linear(64, 32),
            nn.ReLU(inplace=False), 
            nn.Linear(32, 1)  
        )

    def forward(self, image, tabular):
        img_features = self.cnn_backbone(image).reshape(image.size(0), -1)
        img_embeds = self.img_embedding(img_features)
        tab_embeds = self.tabular_mlp(tabular)
        combined = torch.cat((img_embeds, tab_embeds), dim=1)
        return self.fusion_mlp(combined).squeeze(1)

# gpu init     
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# succsarnet init 
succsarnet = FusionModel(tabular_input_dim=len(X_train.columns))
succsarnet = succsarnet.to(device)

# loss func 
pos_weight = torch.tensor([len(failure_df) / len(success_df)], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(succsarnet.parameters())

# training 
for epoch in range(10):
    succsarnet.train()
    total_loss = 0
    for images, tabular, labels in train_loader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)

        logits = succsarnet(images, tabular)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}')

    succsarnet.eval()
    val_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, tabular, labels in val_loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            
            logits = succsarnet(images, tabular)
            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = criterion(logits, labels)
            val_loss += loss.item()
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    pr_auc = average_precision_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds) 
            
    print(f'Val Loss: {val_loss / len(val_loader):.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, PR-AUC: {pr_auc:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

# evaluating on test set 
succsarnet.eval()
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for images, tabular, labels in test_loader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)

        logits = succsarnet(images, tabular)
        probs = torch.sigmoid(logits)
        
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)
pr_auc = average_precision_score(all_labels, all_probs) 
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f'\n=== Test Set Performance ===')
print(f'Accuracy     : {accuracy:.4f}')
print(f'Precision    : {precision:.4f}')
print(f'Recall       : {recall:.4f}')
print(f'ROC-AUC      : {roc_auc:.4f}')
print(f'PR-AUC       : {pr_auc:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')

# saving model 
torch.save({
    'model_state_dict': succsarnet.state_dict(),
    'scaler': scaler
}, 
'succsarnet.pth')

# running model on full set:
full_cat = pd.get_dummies(df[categorical_cols], drop_first=True)
full_cat = full_cat.reindex(columns=train_cat.columns, fill_value=0)
full_num = pd.DataFrame(scaler.transform(df[numeric_cols]), columns=numeric_cols)

X_full = pd.concat([full_num, full_cat], axis=1).astype('float32')
y_full = df['Success'].values
img_full = ids.values

full_set = MultimodalDataset(X_full, y_full, img_full, img_path, transform_pl)
full_loader = DataLoader(full_set, batch_size=32)

# getting embeddings of resnet backbone 
def get_embeddings(self, image, tabular):
    img_features = self.cnn_backbone(image).reshape(image.size(0), -1)
    img_embeds = self.img_embedding(img_features)
    tab_embeds = self.tabular_mlp(tabular)
    logits = self.fusion_mlp(torch.cat((img_embeds, tab_embeds), dim=1)).squeeze(1)
    return img_embeds, torch.sigmoid(logits)

succsarnet.eval()
embeddings = []
predictions = []

with torch.no_grad():
    for images, tabular, _ in full_loader:
        images, tabular = images.to(device), tabular.to(device)
        img_embeds, probs = succsarnet.get_embeddings(images, tabular)

        embeddings.append(img_embeds.cpu().numpy())
        predictions.append(probs.cpu().numpy())

embeddings = np.vstack(embeddings)
predictions = np.hstack(predictions)

# Save to dataframe
embedding_cols = [f'embed_{i}' for i in range(128)]
df_embed = pd.DataFrame(embeddings, columns=embedding_cols)
df_embed['predicted_success'] = predictions
df_final = pd.concat([df.reset_index(drop=True), df_embed], axis=1)
df_final.to_csv("df_with_preds_and_embeddings.csv", index=False)


