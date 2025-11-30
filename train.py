import torch
import os
import glob
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

# 自作モジュール
from step_to_dataset import process_step_to_pyg
from model import BRepNetLite

# データセットのパス設定 (環境に合わせて書き換えてください)
DATA_ROOT = "./data/breps"
STEP_DIR = os.path.join(DATA_ROOT, "step")
SEG_DIR = os.path.join(DATA_ROOT, "seg")

def load_fusion360_seg_labels(seg_path, num_faces):
    """
    Fusion 360 Gallery Datasetの .seg ファイルから正解ラベル(y)を取得する
    形式:
      5
      0
      1
      ... (1行に1つのクラスID)
    """
    if not os.path.exists(seg_path):
        print(f"Label file not found: {seg_path}")
        return None

    try:
        with open(seg_path, 'r') as f:
            # 各行を読み込み、空白を除去して整数に変換
            labels_list = [int(line.strip()) for line in f if line.strip().isdigit()]
    except Exception as e:
        print(f"Error reading .seg file: {e}")
        return None

    # STEPから読み取った面の数と、ラベルの数が一致するか確認
    if len(labels_list) != num_faces:
        print(f"Warning: Face count mismatch! STEP:{num_faces} vs SEG:{len(labels_list)}")
        
        # 安全策: 数が合わない場合のパディング/切り詰め処理
        if len(labels_list) > num_faces:
            labels_list = labels_list[:num_faces]
        else:
            # 足りない場合は "Other" (仮に7) で埋める
            labels_list += [7] * (num_faces - len(labels_list))

    return torch.tensor(labels_list, dtype=torch.long)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. データセット構築ループ ---
    dataset = []
    
    # STEPファイルを検索
    step_files = glob.glob(os.path.join(STEP_DIR, "*.stp"))[:10] # まずは10個だけでテスト
    
    print(f"Found {len(step_files)} step files. Processing...")

    for step_path in step_files:
        # 対応する .seg ファイルのパスを作成
        # ファイル名 (例: 1234.step -> 1234.seg)
        base_name = os.path.splitext(os.path.basename(step_path))[0]
        seg_path = os.path.join(SEG_DIR, base_name + ".seg")
        
        # 1. STEP -> Graph
        data = process_step_to_pyg(step_path)
        if data is None: continue
            
        # 2. SEG -> Labels
        labels = load_fusion360_seg_labels(seg_path, data.num_nodes)
        if labels is None: continue
            
        data.y = labels
        dataset.append(data)
        print(f"Loaded: {base_name} (Nodes: {data.num_nodes})")

    if len(dataset) == 0:
        print("No valid data found.")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- 2. モデル定義 ---
    # クラス数は .seg 内の最大値から推測するか、Fusion360の定義(通常8〜10クラス)に合わせる
    NUM_CLASSES = 10 
    model = BRepNetLite(num_node_features=8, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    # --- 3. 学習 ---
    print("\nStart Training...")
    for epoch in range(50):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}')

    # 保存
    torch.save(model.state_dict(), "brepnet_poc.pth")
    print("Training finished & Model saved.")

if __name__ == "__main__":
    train()