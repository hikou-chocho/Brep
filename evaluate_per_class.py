import os
import glob
import torch
import numpy as np
from torch_geometric.loader import DataLoader

# 自作モジュール
from step_to_dataset import process_step_to_pyg
from model import BRepNetModern

# 設定
DATA_ROOT = "./data/breps"
STEP_DIR = os.path.join(DATA_ROOT, "step")
SEG_DIR = os.path.join(DATA_ROOT, "seg")
# モデルパスは環境変数 MODEL_PATH で上書き可
MODEL_PATH = os.environ.get('MODEL_PATH', '')
NUM_CLASSES = 10
SAMPLE_N = int(os.environ.get('EVAL_SAMPLE_N', '500'))
BATCH_SIZE = int(os.environ.get('EVAL_BATCH_SIZE', '32'))

CLASS_NAMES = [
    "ExtrudeSide (0)", "ExtrudeEnd (1)", "CutSide (2)", "CutEnd (3)",
    "Fillet (4)", "Chamfer (5)", "Hole (6)", "Other (7)", "Gear (8)", "Toroidal (9)"
]


def load_labels(seg_path, num_faces):
    if not os.path.exists(seg_path):
        return None
    try:
        with open(seg_path, 'r') as f:
            labels = [int(line.strip()) for line in f if line.strip().isdigit()]
    except Exception:
        return None
    if len(labels) != num_faces:
        if len(labels) > num_faces:
            labels = labels[:num_faces]
        else:
            labels += [7] * (num_faces - len(labels))
    return torch.tensor(labels, dtype=torch.long)


def find_model_path():
    candidates = []
    if MODEL_PATH:
        candidates.append(MODEL_PATH)
    # common names
    candidates.extend(["brepnet_weighted_best.pth", "brepnet_best.pth", "brepnet.pth"])
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_file = find_model_path()
    if model_file is None:
        print("No model file found. Set MODEL_PATH env var or place 'brepnet_weighted_best.pth' / 'brepnet_best.pth' in repo root.")
        return

    print(f"Loading model from: {model_file}")
    model = BRepNetModern(num_node_features=9, num_edge_features=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # collect step files (support both .step and .stp)
    step_files = glob.glob(os.path.join(STEP_DIR, "*.step")) + glob.glob(os.path.join(STEP_DIR, "*.stp"))
    if len(step_files) == 0:
        print("No STEP files found in:", STEP_DIR)
        return

    # sample from end or random
    if SAMPLE_N and SAMPLE_N > 0 and len(step_files) > SAMPLE_N:
        step_files = step_files[-SAMPLE_N:]

    dataset = []
    print("Preparing test data...")
    for step_path in step_files:
        base_name = os.path.splitext(os.path.basename(step_path))[0]
        seg_path = os.path.join(SEG_DIR, base_name + ".seg")
        data = process_step_to_pyg(step_path)
        if data is None:
            continue
        labels = load_labels(seg_path, data.x.shape[0])
        if labels is None:
            continue
        data.y = labels
        dataset.append(data)

    if len(dataset) == 0:
        print("No valid graphs for evaluation.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    all_preds = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)

    overall_acc = (all_preds == all_labels).mean()
    print(f"Overall Accuracy: {overall_acc:.2%}")

    # confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for a, p in zip(all_labels, all_preds):
        if 0 <= a < NUM_CLASSES and 0 <= p < NUM_CLASSES:
            cm[a, p] += 1

    # per-class metrics
    precisions = []
    recalls = []
    f1s = []
    supports = cm.sum(axis=1)

    for cls in range(NUM_CLASSES):
        tp = cm[cls, cls]
        pred_pos = cm[:, cls].sum()
        actual_pos = supports[cls]
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / actual_pos if actual_pos > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # print report
    print("\nPer-Class Report:")
    print(f"{'Class':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print('-' * 60)
    for i in range(NUM_CLASSES):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
        print(f"{name:<25} {precisions[i]:6.2f} {recalls[i]:6.2f} {f1s[i]:6.2f} {supports[i]:8d}")

    # show confusion matrix summary for Hole (6)
    hole_idx = 6
    if 0 <= hole_idx < NUM_CLASSES:
        hole_recall = recalls[hole_idx]
        print('\n' + '='*40)
        print(f"Hole (class {hole_idx}) Recall: {hole_recall:.2%} (TP={cm[hole_idx, hole_idx]}, Support={supports[hole_idx]})")
        print('='*40)

    # print confusion matrix (small)
    print('\nConfusion Matrix (rows=true, cols=pred):')
    print(cm)


if __name__ == '__main__':
    evaluate()
