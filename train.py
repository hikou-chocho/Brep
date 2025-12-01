import torch
import os
import glob
import random
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from collections import Counter
# sklearn is not required; using manual accuracy calculation

# 自作モジュール
from step_to_dataset import process_step_to_pyg
from model import BRepNetModern

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
        # print warning at caller
        return None

    try:
        with open(seg_path, 'r') as f:
            # 各行を読み込み、空白を除去して整数に変換
            labels_list = [int(line.strip()) for line in f if line.strip().isdigit()]
    except Exception:
        return None

    # STEPから読み取った面の数と、ラベルの数が一致するか確認
    if len(labels_list) != num_faces:
        # 安全策: 数が合わない場合のパディング/切り詰め処理
        if len(labels_list) > num_faces:
            labels_list = labels_list[:num_faces]
        else:
            # 足りない場合は "Other" (仮に7) で埋める
            labels_list += [7] * (num_faces - len(labels_list))

    return torch.tensor(labels_list, dtype=torch.long)


def calc_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total if total > 0 else 0.0


def load_labels(seg_path, num_faces):
    return load_fusion360_seg_labels(seg_path, num_faces)


def train_proper():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. データ収集 (全ファイル対象)
    step_files = glob.glob(os.path.join(STEP_DIR, "*.stp"))
    dataset = []
    print(f"Loading {len(step_files)} files...")

    # データ量が多すぎる環境向けにサンプリングで削減（既定で30%に削減）
    try:
        sample_ratio = float(os.environ.get('DATA_SAMPLE_RATIO', '1.0'))
    except Exception:
        sample_ratio = 0.3
    if sample_ratio <= 0 or sample_ratio > 1:
        sample_ratio = 0.3
    keep_n = max(1, int(len(step_files) * sample_ratio))
    if len(step_files) > keep_n:
        random.shuffle(step_files)
        step_files = step_files[:keep_n]
    print(f"Sampling dataset: keeping {len(step_files)} of {len(glob.glob(os.path.join(STEP_DIR, '*.stp')))} files ({sample_ratio*100:.0f}%)")
    for i, step_path in enumerate(step_files):
        base_name = os.path.splitext(os.path.basename(step_path))[0]
        seg_path = os.path.join(SEG_DIR, base_name + ".seg")
        if i % 10 == 0:
            print(f"Processing {i}/{len(step_files)}...", end="\r")

        data = process_step_to_pyg(step_path)
        if data is None:
            continue

        labels = load_labels(seg_path, data.num_nodes)
        if labels is None:
            continue
        data.y = labels
        dataset.append(data)

    print(f"\nTotal valid graphs: {len(dataset)}")

    if len(dataset) == 0:
        print("No valid data found.")
        return

    # 2. Train / Test 分割 (8:2)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    
    # バッチサイズを大きめに設定して BatchNorm を効かせる
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. モデル定義
    # Node features: 基本特徴量 v6 = 9 (IsClosed を追加)
    # Edge features: [Convex, Concave, Smooth] = 3
    # --- クラス重みを計算して損失に反映する ---
    NUM_CLASSES = 10
    # 集計関数
    def calculate_class_weights(dataset, num_classes):
        print("Calculating class weights...")
        all_labels = []
        for data in dataset:
            if hasattr(data, 'y'):
                all_labels.extend(data.y.tolist())
        counts = Counter(all_labels)
        total = len(all_labels) if len(all_labels) > 0 else 1
        weights = torch.zeros(num_classes, dtype=torch.float)
        for cls_idx in range(num_classes):
            c = counts.get(cls_idx, 0)
            if c > 0:
                weights[cls_idx] = total / (num_classes * c)
            else:
                weights[cls_idx] = 1.0
        # Smooth weights to avoid extreme penalties: sqrt and clamp
        weights = torch.sqrt(weights)
        weights = torch.clamp(weights, max=10.0)
        print(f"Smoothed Class Weights: {weights}")
        return weights

    class_weights = calculate_class_weights(train_dataset, NUM_CLASSES).to(device)

    model = BRepNetModern(num_node_features=9, num_edge_features=3, num_classes=NUM_CLASSES, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # 4. 学習ループ（Best-test-acc 保存ロジックを追加）
    print("\nStart Training Loop...")
    best_acc = 0.0
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            # 重み付き損失を使用（クラス分布に応じたペナルティ）
            loss = F.nll_loss(out, batch.y, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 毎エポックで評価してベストモデルを保存
        train_acc = calc_accuracy(train_loader, model, device)
        test_acc = calc_accuracy(test_loader, model, device)
        print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")

        if test_acc > best_acc:
            best_acc = test_acc
            try:
                torch.save(model.state_dict(), "brepnet_best.pth")
                print(f"  >>> Best Model Saved! ({best_acc:.2%})")
            except Exception as e:
                print(f"Failed to save best model: {e}")


if __name__ == "__main__":
    train_proper()