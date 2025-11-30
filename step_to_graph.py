import sys
import os
import cadquery as cq
try:
    import torch
    # PyTorch Geometricのデータ構造
    from torch_geometric.data import Data
    HAS_TORCH = True
except Exception:
    torch = None
    Data = None
    HAS_TORCH = False

# === OCP インポート ===
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
# 特徴量抽出用に追加
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                         GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                         GeomAbs_BSplineSurface)
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

def get_face_features(ocp_face):
    """
    1つの面から幾何特徴量(タイプID, 面積)を抽出する
    """
    # 1. 曲面タイプの取得
    adaptor = BRepAdaptor_Surface(ocp_face)
    surf_type_enum = adaptor.GetType()
    
    # Enumを人間が読みやすい文字列にするマッピング（デバッグ用）
    # 実際には学習時にOne-Hotエンコーディング等で使用します
    type_map = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "Bezier",
        GeomAbs_BSplineSurface: "BSpline"
    }
    type_name = type_map.get(surf_type_enum, f"Other({surf_type_enum})")

    # 2. 面積の取得
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(ocp_face, props)
    area = props.Mass()

    return {
        "type_id": int(surf_type_enum),
        "type_name": type_name,
        "area": area
    }

def step_to_brepnet_data(filename):
    """
    STEPを読み込み、グラフ構造(A)と特徴量(X)を返す
    """
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)

    try:
        model = cq.importers.importStep(filename)
        print(f"Success: Loaded {filename}")
    except Exception as e:
        print(f"Error loading STEP file: {e}")
        sys.exit(1)

    shape = model.val().wrapped

    # --- 1. ノード（面）のリスト化と特徴量抽出 ---
    ocp_faces = []
    node_features = [] # これが特徴量行列 X になります
    
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current()) # 明示的にダウンキャスト
        ocp_faces.append(f)
        
        # 特徴量抽出関数を呼び出し
        feats = get_face_features(f)
        node_features.append(feats)
        
        exp.Next()

    print(f"Total Faces (Nodes): {len(ocp_faces)}")

    # --- 2. 隣接関係（エッジ）の構築 ---
    # OCCT の TopTools マップを使って Edge -> [Faces] を構築
    edge_to_faces_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces_map)

    # Face オブジェクトからインデックスへのマップ
    face_to_id = {f: i for i, f in enumerate(ocp_faces)}

    def _get_face_id_by_shape(s, mapping):
        # まず直接キーで探す
        if s in mapping:
            return mapping[s]
        # フォールバック: IsSame を使って線形検索
        for k, v in mapping.items():
            try:
                if s.IsSame(k):
                    return v
            except Exception:
                try:
                    if k.IsSame(s):
                        return v
                except Exception:
                    continue
        return None

    adjacency_set = set()
    for i in range(1, edge_to_faces_map.Extent() + 1):
        faces_list = edge_to_faces_map.FindFromIndex(i)
        if faces_list.Extent() >= 2:
            # faces_list は TopTools_ListOfShape のため First()/Value()/Next() で反復
            ids = []
            try:
                it = faces_list.First()
                while it.More():
                    shp = it.Value()
                    try:
                        shp_face = TopoDS.Face_s(shp)
                    except Exception:
                        shp_face = shp

                    fid = _get_face_id_by_shape(shp_face, face_to_id)
                    if fid is not None:
                        ids.append(fid)
                    it.Next()
            except Exception:
                # 念のため安全に続行
                continue

            # ペア化（自己ペア除外）
            ids = sorted(set(ids))
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    pair = (ids[a], ids[b])
                    adjacency_set.add(pair)

    adjacency_list = sorted(list(adjacency_set))

    # Map ベースで隣接が見つからなかった場合のフォールバック:
    # 各エッジの両端頂点の座標をキーにして、面リストを作る方法
    if len(adjacency_list) == 0:
        print("DEBUG: adjacency_list empty after Map approach, using vertex-key fallback")
        edge_map_v = {}
        for fi, f in enumerate(ocp_faces):
            exp_e = TopExp_Explorer(f, TopAbs_EDGE)
            while exp_e.More():
                e = exp_e.Current()
                # 頂点を取得
                verts = []
                exp_v = TopExp_Explorer(e, TopAbs_VERTEX)
                while exp_v.More():
                    v = exp_v.Current()
                    try:
                        p = BRep_Tool.Pnt(TopoDS.Vertex_s(v))
                        coords = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
                    except Exception:
                        coords = repr(v)
                    verts.append(coords)
                    exp_v.Next()

                if len(verts) >= 2:
                    # ソートして（端点順に依存しない）キー化
                    key = tuple(sorted((verts[0], verts[1])))
                else:
                    key = tuple(verts)

                edge_map_v.setdefault(key, []).append(fi)
                exp_e.Next()

        adjacency_set_v = set()
        for faces in edge_map_v.values():
            if len(faces) >= 2:
                faces = sorted(set(faces))
                for a in range(len(faces)):
                    for b in range(a + 1, len(faces)):
                        adjacency_set_v.add((faces[a], faces[b]))

        adjacency_list = sorted(list(adjacency_set_v))

    return node_features, adjacency_list


# --- PyG 用変換ユーティリティ ---
# 扱う曲面タイプのリスト（順番が重要）
SURFACE_TYPES = [
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
]
NUM_TYPES = len(SURFACE_TYPES)


def get_face_features_tensor(ocp_face):
    """
    面から特徴量ベクトルを作成する
    One-Hot (NUM_TYPES) + 面積 -> 長さ NUM_TYPES+1
    """
    adaptor = BRepAdaptor_Surface(ocp_face)
    surf_type = adaptor.GetType()

    one_hot = [0.0] * NUM_TYPES
    for idx, t in enumerate(SURFACE_TYPES):
        try:
            if surf_type == t:
                one_hot[idx] = 1.0
                break
        except Exception:
            continue

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(ocp_face, props)
    area = props.Mass()

    return one_hot + [area]


def build_adjacency_list(shape, ocp_faces):
    """
    与えられた shape と ocp_faces から隣接リストを構築して返す。
    既存の TopTools マップ + 頂点キーのフォールバック論理を共通化。
    """
    edge_to_faces_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces_map)

    face_to_id = {f: i for i, f in enumerate(ocp_faces)}

    def _get_face_id_by_shape(s, mapping):
        if s in mapping:
            return mapping[s]
        for k, v in mapping.items():
            try:
                if s.IsSame(k):
                    return v
            except Exception:
                try:
                    if k.IsSame(s):
                        return v
                except Exception:
                    continue
        return None

    adjacency_set = set()
    for i in range(1, edge_to_faces_map.Extent() + 1):
        faces_list = edge_to_faces_map.FindFromIndex(i)
        if faces_list.Extent() >= 2:
            # faces_list は TopTools_ListOfShape のため First()/Value()/Next() で反復
            ids = []
            try:
                it = faces_list.First()
                while it.More():
                    shp = it.Value()
                    try:
                        shp_face = TopoDS.Face_s(shp)
                    except Exception:
                        shp_face = shp

                    fid = _get_face_id_by_shape(shp_face, face_to_id)
                    if fid is not None:
                        ids.append(fid)
                    it.Next()
            except Exception:
                continue

            ids = sorted(set(ids))
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    adjacency_set.add((ids[a], ids[b]))

    adjacency_list = sorted(list(adjacency_set))

    # フォールバック: 頂点キーで再構築
    if len(adjacency_list) == 0:
        edge_map_v = {}
        for fi, f in enumerate(ocp_faces):
            exp_e = TopExp_Explorer(f, TopAbs_EDGE)
            while exp_e.More():
                e = exp_e.Current()
                verts = []
                exp_v = TopExp_Explorer(e, TopAbs_VERTEX)
                while exp_v.More():
                    v = exp_v.Current()
                    try:
                        p = BRep_Tool.Pnt(TopoDS.Vertex_s(v))
                        coords = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
                    except Exception:
                        coords = repr(v)
                    verts.append(coords)
                    exp_v.Next()

                if len(verts) >= 2:
                    key = tuple(sorted((verts[0], verts[1])))
                else:
                    key = tuple(verts)

                edge_map_v.setdefault(key, []).append(fi)
                exp_e.Next()

        adjacency_set_v = set()
        for faces in edge_map_v.values():
            if len(faces) >= 2:
                faces = sorted(set(faces))
                for a in range(len(faces)):
                    for b in range(a + 1, len(faces)):
                        adjacency_set_v.add((faces[a], faces[b]))

        adjacency_list = sorted(list(adjacency_set_v))

    return adjacency_list


def process_step_to_pyg(filename):
    """
    STEP を読み込み、PyG の Data を作成して返す
    """
    if not os.path.exists(filename):
        return None

    try:
        model = cq.importers.importStep(filename)
    except Exception:
        return None

    shape = model.val().wrapped

    # ノード列挙と特徴量抽出
    ocp_faces = []
    x_data = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        ocp_faces.append(f)
        vec = get_face_features_tensor(f)
        x_data.append(vec)
        exp.Next()

    if len(x_data) == 0:
        return None

    x = torch.tensor(x_data, dtype=torch.float)

    # 隣接リスト構築（既存の堅牢なロジックを使う）
    adjacency_list = build_adjacency_list(shape, ocp_faces)

    # edge_index を作成（無向グラフ -> 双方向で保存）
    src_list = []
    dst_list = []
    for a, b in adjacency_list:
        src_list.append(a)
        dst_list.append(b)
        src_list.append(b)
        dst_list.append(a)

    if len(src_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data

if __name__ == "__main__":
    input_filename = sys.argv[1] if len(sys.argv) > 1 else "sample.step"
    output_filename = "processed_data.pt"

    print(f"Processing {input_filename} ...")

    if not HAS_TORCH:
        print("Warning: torch or torch_geometric not installed. Falling back to diagnostic output.")
        features, adj_list = step_to_brepnet_data(input_filename)

        print("\n=== 1. Node Features (Geometry) ===")
        for i, ft in enumerate(features[:10]):
            print(f"Face {i}: Type={ft['type_name']:<10} Area={ft['area']:.2f}")
        if len(features) > 10:
            print("...")

        print("\n=== 2. Adjacency Graph (Topology) ===")
        print(f"Total Connections: {len(adj_list)}")
        for pair in adj_list[:10]:
            print(f"Face {pair[0]} <--> Face {pair[1]}")
        if len(adj_list) > 10:
            print("...")

        print("\nInstall torch and torch_geometric to export a PyG Data object.")
        sys.exit(0)

    data = process_step_to_pyg(input_filename)

    if data:
        print("\n=== PyTorch Geometric Data Object ===")
        print(data)
        print("-" * 30)
        print(f"x (Features): \n{data.x.shape}  <- [Nodes, {NUM_TYPES + 1} dims ({NUM_TYPES} Type + 1 Area)]")
        print(f"edge_index: \n{data.edge_index.shape}  <- [2, Edges]")
        
        # 保存
        try:
            torch.save(data, output_filename)
            print(f"\nSaved to: {output_filename}")
        except Exception as e:
            print(f"Failed to save data: {e}")

        print("Ready for training/inference!")
    else:
        print("Failed to process file.")