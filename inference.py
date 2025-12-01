import sys
import os
import torch
import numpy as np

# 自作モジュール
from step_to_dataset import process_step_to_pyg
from model import BRepNetModern

# クラス定義
CLASS_NAMES = [
    "ExtrudeSide (側面)",    # 0
    "ExtrudeEnd (天面/底面)", # 1
    "CutSide (切削側面)",     # 2
    "CutEnd (切削底面)",      # 3
    "Fillet (フィレット)",    # 4
    "Chamfer (面取り)",      # 5
    "Hole (穴)",            # 6
    "Other (その他)",        # 7
    "Gear", "Toroidal"
]

# モデル設定
MODEL_PATH = os.environ.get('MODEL_PATH', 'brepnet_best.pth')
NUM_NODE_FEATS = int(os.environ.get('NUM_NODE_FEATS', '9'))
NUM_EDGE_FEATS = int(os.environ.get('NUM_EDGE_FEATS', '3'))
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', '10'))


def predict_step_file(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    print(f"analyzing {filename} ...")

    # 1. データを生成
    data = process_step_to_pyg(filename)
    if data is None:
        print("Failed to process STEP file.")
        return

    if not hasattr(data, 'x') or data.x is None or data.x.size(0) == 0:
        print("No faces/features found in the STEP file.")
        return

    # 2. モデルロード & 推論
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BRepNetModern(NUM_NODE_FEATS, NUM_EDGE_FEATS, NUM_CLASSES).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        return

    model.eval()

    with torch.no_grad():
        data = data.to(device)
        out = model(data)
        probs = torch.exp(out)  # log_softmax -> probability
        preds = out.argmax(dim=1)

    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()

    # 3. 結果表示
    print("\n" + "=" * 40)
    print(f"Result for: {os.path.basename(filename)}")
    print("=" * 40)

    for i, pred_cls in enumerate(preds_np):
        cls_name = CLASS_NAMES[pred_cls] if pred_cls < len(CLASS_NAMES) else f"Class {pred_cls}"
        confidence = probs_np[i][pred_cls] * 100
        marker = " "
        if pred_cls == 6:
            marker = "●"  # Hole
        elif pred_cls == 4:
            marker = "○"  # Fillet
        elif pred_cls == 5:
            marker = "△"  # Chamfer
        print(f"{marker} Face {i:03d}: {cls_name:<20} ({confidence:.1f}%)")

    # サマリー
    unique, counts = np.unique(preds_np, return_counts=True)
    print("-" * 40)
    print("Summary:")
    for cls, count in zip(unique, counts):
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        print(f"  {name}: {count} faces")


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "sample.step"
    predict_step_file(target_file)
