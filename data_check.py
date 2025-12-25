import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import Adam

#PHẦN 1: THIẾT LẬP VÀ TIỀN XỬ LÝ
DATA_PATH = 'D:/CIC_IDS_Data' 
SAVE_GRAPH_PATH = 'D:/CIC_IDS_Data/final_graph_data.pt'

all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
df_list = []
print("Processing: Reading CSV files...")
for filename in all_files:
    try:
        df = pd.read_csv(filename, low_memory=False) 
        df_list.append(df)
    except Exception as e: print(f"Error: {e}")

df_combined = pd.concat(df_list, axis=0, ignore_index=True)
df_combined.columns = df_combined.columns.str.strip().str.replace(' ', '_')
df_combined.replace([np.inf, -np.inf, 'NaN', 'Infinity'], np.nan, inplace=True)
df_combined.fillna(0, inplace=True)

# Loại bỏ metadata và gộp nhãn
cols_to_drop = ['Flow_ID', 'Source_IP', 'Source_Port', 'Destination_IP', 'Destination_Port', 'Protocol', 'Timestamp']
df_combined.drop(columns=[c for c in cols_to_drop if c in df_combined.columns], inplace=True)

def consolidate_label(label):
    label = str(label).strip().upper()
    if label == 'BENIGN': return 'Benign'
    if 'DOS' in label or 'HEARTBLEED' in label: return 'DoS'
    if 'DDOS' in label: return 'DDoS'
    if 'INFILTRATION' in label: return 'Infiltration'
    if 'PATATOR' in label: return 'Brute_Force'
    if 'BOT' in label or 'WEB ATTACK' in label: return 'Other_Attack'
    return 'Other'

df_combined['Label_Category'] = df_combined['Label'].apply(consolidate_label)

# Cân bằng dữ liệu (Systematic Sampling cho Benign)
df_benign = df_combined[df_combined['Label_Category'] == 'Benign']
df_attack = df_combined[df_combined['Label_Category'] != 'Benign']
step_size = max(1, len(df_benign) // 500000)
df_benign_balanced = df_benign.iloc[::step_size, :].copy()
df_inf_boosted = pd.concat([df_attack[df_attack['Label_Category'] == 'Infiltration']] * 100, ignore_index=True)
df_final = pd.concat([df_benign_balanced, df_attack[df_attack['Label_Category'] != 'Infiltration'], df_inf_boosted]).sort_index()

# Chuẩn hóa
scaler = RobustScaler()
x_scaled = scaler.fit_transform(df_final.drop(columns=['Label', 'Label_Category']))
label_map = {cat: i for i, cat in enumerate(df_final['Label_Category'].unique())}
y_tensor = torch.tensor(df_final['Label_Category'].map(label_map).values, dtype=torch.long)

# PHẦN 2: XÂY DỰNG ĐỒ THỊ
print("Processing: Constructing Graph...")
num_nodes = x_scaled.shape[0]
src, dst = [], []
for i in range(num_nodes - 1):
    src.extend([i, i]); dst.extend([i + 1, i + 2 if i + 2 < num_nodes else i + 1])

edge_index = torch.tensor([src, dst], dtype=torch.long)
graph_data = Data(x=torch.tensor(x_scaled, dtype=torch.float), edge_index=edge_index, y=y_tensor)
torch.save(graph_data, SAVE_GRAPH_PATH)

#PHẦN 3: HUẤN LUYỆN MÔ HÌNH GNN
class GCN_IDS(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN_IDS, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.classifier = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(self.classifier(x), dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_IDS(graph_data.num_node_features, len(label_map)).to(device)
graph_data = graph_data.to(device)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Chia mask 80/20
indices = torch.randperm(graph_data.num_nodes)
train_mask = indices[:int(graph_data.num_nodes * 0.8)]
test_mask = indices[int(graph_data.num_nodes * 0.8):]

print(f"Training on {device}...")
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad(); out = model(graph_data)
    loss = criterion(out[train_mask], graph_data.y[train_mask])
    loss.backward(); optimizer.step()
    if epoch % 10 == 0: print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')

#PHẦN 4: ĐÁNH GIÁ
print("\nEvaluating Model...")
model.eval()
with torch.no_grad():
    out = model(graph_data)
    pred = out.argmax(dim=1)

y_true = graph_data.y[test_mask].cpu().numpy()
y_pred = pred[test_mask].cpu().numpy()
target_names = list(label_map.keys())

print(f"\nAccuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
print(classification_report(y_true, y_pred, target_names=target_names))

# Vẽ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.title('Confusion Matrix - GNN Performance')
plt.show()
