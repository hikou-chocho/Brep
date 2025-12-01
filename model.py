import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINEConv


class BRepNetModern(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes, hidden_dim=128):
        super(BRepNetModern, self).__init__()
        
        self.hidden_dim = hidden_dim

        # 入力層: ノード特徴量を隠れ層の次元へ拡張
        self.node_encoder = Linear(num_node_features, hidden_dim)
        
        # エッジエンコーダー: エッジ特徴量を隠れ層の次元へ拡張
        self.edge_encoder = Linear(num_edge_features, hidden_dim)

        # === Layer 1 ===
        nn1 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
        self.conv1 = GINEConv(nn1)
        self.bn1 = BatchNorm1d(hidden_dim)

        # === Layer 2 ===
        nn2 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
        self.conv2 = GINEConv(nn2)
        self.bn2 = BatchNorm1d(hidden_dim)
        
        # === Layer 3 ===
        nn3 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
        self.conv3 = GINEConv(nn3)
        self.bn3 = BatchNorm1d(hidden_dim)

        # === Layer 4 (Deep化) ===
        nn4 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
        self.conv4 = GINEConv(nn4)
        self.bn4 = BatchNorm1d(hidden_dim)

        # 出力層
        self.classifier = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, 'edge_attr', None)

        # 1. Embedding
        x = self.node_encoder(x)
        if edge_attr is None:
            # 0-pad edge_attr if missing
            edge_attr = torch.zeros((edge_index.size(1), 3), dtype=x.dtype, device=x.device)
        edge_attr = self.edge_encoder(edge_attr)

        # 2. Message Passing (GINEConv)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x, edge_index, edge_attr=edge_attr)
        x = self.bn4(x)
        x = F.relu(x)

        # 3. Output
        out = self.classifier(x)
        return F.log_softmax(out, dim=1)