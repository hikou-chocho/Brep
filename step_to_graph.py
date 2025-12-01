import sys
import os
import math
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
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX, TopAbs_WIRE, TopAbs_REVERSED
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
# 特徴量抽出用に追加
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.CSLib import CSLib
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.gp import gp_Vec, gp_Pnt
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
    [v2] 面からリッチな特徴量ベクトルを作成する
    構成: [Type(7), LogArea(1), NumWires(1), NumEdges(1)] -> 合計 NUM_TYPES + 3 次元
    """
    # 1. 曲面タイプ (One-Hot)
    adaptor = BRepAdaptor_Surface(ocp_face)
    surf_type = adaptor.GetType()
    
    one_hot = [0.0] * NUM_TYPES
    if surf_type in SURFACE_TYPES:
        idx = SURFACE_TYPES.index(surf_type)
        one_hot[idx] = 1.0
    
    # 2. 面積 (対数正規化)
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(ocp_face, props)
    raw_area = props.Mass()
    
    # log(area + 1) で0除算を防ぎつつ正規化
    # さらに適当な係数(例: 0.1)を掛けて 0~1 近辺に寄せると学習が安定します
    norm_area = math.log(raw_area + 1.0) * 0.1

    # 3. ワイヤー数 (穴あき情報のヒント)
    num_wires = 0
    try:
        w_exp = TopExp_Explorer(ocp_face, TopAbs_WIRE)
        while w_exp.More():
            num_wires += 1
            w_exp.Next()
    except Exception:
        num_wires = 0

    # 4. エッジ数 (複雑さのヒント)
    num_edges = 0
    try:
        e_exp = TopExp_Explorer(ocp_face, TopAbs_EDGE)
        while e_exp.More():
            num_edges += 1
            e_exp.Next()
    except Exception:
        num_edges = 0

    # 正規化: エッジ数やワイヤー数も大きくなりすぎないようにスケーリング
    norm_wires = num_wires * 0.2
    norm_edges = num_edges * 0.05

    # リスト結合: [Type(7)] + [Area, Wires, Edges]
    feature_vec = one_hot + [norm_area, norm_wires, norm_edges]
    return feature_vec


def get_convexity(edge, face1, face2):
    """
    エッジに対して凸(1)、凹(-1)、滑らか/不明(0) を返す。
    """
    try:
        # エッジ中点と接線
        ca = BRepAdaptor_Curve(edge)
        u_min = ca.FirstParameter()
        u_max = ca.LastParameter()
        u = (u_min + u_max) / 2.0
        p_mid = ca.Value(u)
        tan = gp_Vec()
        ca.D1(u, p_mid, tan)
        # エッジ向きを考慮
        try:
            if edge.Orientation() == TopAbs_REVERSED:
                tan.Reverse()
        except Exception:
            pass

        # 面上の法線取得 (ShapeAnalysis を使ってUVを取得)
        def _normal_at(face, point):
            try:
                s_adapt = BRepAdaptor_Surface(face)
                sas = ShapeAnalysis_Surface(s_adapt.Surface().Surface())
                uv = sas.ValueOfUV(point, 1e-6)
                # D1 を使って接線を取り、外積で法線を得る
                ptmp = gp_Pnt()
                du = gp_Vec()
                dv = gp_Vec()
                s_adapt.D1(uv.X(), uv.Y(), ptmp, du, dv)
                # 法線 = du x dv
                n = du.Crossed(dv)
                if face.Orientation() == TopAbs_REVERSED:
                    n.Reverse()
                return n
            except Exception:
                return None

        n1 = _normal_at(face1, p_mid)
        n2 = _normal_at(face2, p_mid)
        if n1 is None or n2 is None:
            return 0.0

        # 外積と接線の内積で凸凹判定
        cp = n1.Crossed(n2)
        dot = cp.Dot(tan)
        thr = 1e-4
        if dot > thr:
            return 1.0
        elif dot < -thr:
            return -1.0
        else:
            return 0.0
    except Exception:
        return 0.0


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

    # 1) Faces とそのタイプを先に列挙
    ocp_faces = []
    face_types = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        ocp_faces.append(f)

        # タイプ判定（未知の場合は NUM_TYPES を割り当てる）
        try:
            adaptor = BRepAdaptor_Surface(f)
            s_type = adaptor.GetType()
            if s_type in SURFACE_TYPES:
                t_idx = SURFACE_TYPES.index(s_type)
            else:
                t_idx = NUM_TYPES
        except Exception:
            t_idx = NUM_TYPES

        face_types.append(t_idx)
        exp.Next()

    if len(ocp_faces) == 0:
        return None

    # 2) 隣接関係を構築
    adjacency_list = build_adjacency_list(shape, ocp_faces)

    # edge_map を構築して Edge オブジェクトと関連面インデックスを保持
    edge_map = {}
    for fi, f in enumerate(ocp_faces):
        exp_e = TopExp_Explorer(f, TopAbs_EDGE)
        while exp_e.More():
            e = exp_e.Current()
            try:
                key = e.HashCode(2147483647)
            except Exception:
                key = repr(e)

            if key not in edge_map:
                edge_map[key] = [e, []]
            edge_map[key][1].append(fi)
            exp_e.Next()

    # 3) 隣接タイプのカウントを計算
    # 各面ごとに (NUM_TYPES + 1) 長のカウンタを持つ (未知タイプ用のスロット含む)
    neighbor_counts = [[0] * (NUM_TYPES + 1) for _ in range(len(ocp_faces))]
    for a, b in adjacency_list:
        # 双方向でカウント
        t_b = face_types[b] if b < len(face_types) else NUM_TYPES
        t_a = face_types[a] if a < len(face_types) else NUM_TYPES
        neighbor_counts[a][t_b] += 1
        neighbor_counts[b][t_a] += 1

    # 4) 特徴量ベクトルを作成 (v2 のベース + 隣接タイプカウントを連結)
    x_data = []
    # convexity counts per face: [convex_count, concave_count, flat_count]
    convexity_counts = [[0, 0, 0] for _ in range(len(ocp_faces))]

    # 計算: edge_map を巡回し、2面共有エッジのみ判定
    for key, (edge_obj, faces_idx) in edge_map.items():
        if len(faces_idx) >= 2:
            # 通常は2面だが、非多様体のため複数ある場合は全ペア処理
            # 全てのペアについて convexity を評価してカウント
            for i in range(len(faces_idx)):
                for j in range(i + 1, len(faces_idx)):
                    fi = faces_idx[i]
                    fj = faces_idx[j]
                    try:
                        cv = get_convexity(edge_obj, ocp_faces[fi], ocp_faces[fj])
                    except Exception:
                        cv = 0.0

                    if cv > 0:
                        convexity_counts[fi][0] += 1
                        convexity_counts[fj][0] += 1
                    elif cv < 0:
                        convexity_counts[fi][1] += 1
                        convexity_counts[fj][1] += 1
                    else:
                        convexity_counts[fi][2] += 1
                        convexity_counts[fj][2] += 1

    for i, f in enumerate(ocp_faces):
        base_vec = get_face_features_tensor(f)
        # 正規化係数: 値が大きくなり過ぎないようスケーリング
        neighbor_vec = [float(c) * 0.2 for c in neighbor_counts[i]]
        # convexity 正規化（小さめの係数）
        cvx = convexity_counts[i]
        cvx_vec = [float(cvx[0]) * 0.1, float(cvx[1]) * 0.1, float(cvx[2]) * 0.05]
        full_vec = base_vec + neighbor_vec + cvx_vec
        x_data.append(full_vec)

    x = torch.tensor(x_data, dtype=torch.float)

    # 5) edge_index を作成（無向グラフ -> 双方向で保存）
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
        print(f"x (Features): \n{data.x.shape}  <- [Nodes, {NUM_TYPES + 3} dims ({NUM_TYPES} Type + 'LogArea+Wires+Edges)])")
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