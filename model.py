import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

class BRepNetLite(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(BRepNetLite, self).__init__()
        
        # --- ネットワーク構造定義 ---
        
        # 1層目: 入力特徴量(8次元) -> 隠れ層(64次元)
        # SAGEConvは近傍の情報を「集約(Aggregate)」するのに適しており、CADデータと相性が良いです
        self.conv1 = SAGEConv(num_node_features, 64)
        
        # 2層目: 隠れ層(64次元) -> 隠れ層(64次元)
        # 深くすることで「隣の隣」の情報まで考慮できるようになります
        self.conv2 = SAGEConv(64, 64)
        
        # 3層目: 隠れ層(64次元) -> 隠れ層(64次元)
        self.conv3 = SAGEConv(64, 64)

        # 出力層: 隠れ層(64次元) -> クラス数(例えば穴、平面など6クラス)
        self.classifier = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # --- 順伝播処理 ---
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x) # 活性化関数
        x = F.dropout(x, p=0.5, training=self.training) # 過学習抑制

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # 全結合層で各ノード(面)ごとのクラス確率を出力
        out = self.classifier(x)
        
        return F.log_softmax(out, dim=1)