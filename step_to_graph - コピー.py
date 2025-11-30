import sys
import os
import cadquery as cq

# === 修正箇所: OCPの正しいクラス名をインポート ===
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS import TopoDS  # ここを修正 (topods -> TopoDS)

def get_face_adjacency_with_cq(filename):
    """
    CadQueryを使用してSTEPを読み込み、隣接グラフを構築する
    """
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)

    # 1. CadQueryでSTEPを読み込む
    try:
        model = cq.importers.importStep(filename)
        print(f"Success: Loaded {filename}")
    except Exception as e:
        print(f"Error loading STEP file: {e}")
        sys.exit(1)

    # 2. CadQueryオブジェクトから生のOCCT形状(TopoDS_Shape)を取得
    shape = model.val().wrapped

    # 3. 全ての面(Face)を取得し、インデックスを振る
    
    # エッジごとに面を集める方法: 各面を走査して、その面に含まれるエッジをキーに面インデックスを格納する
    ocp_faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        ocp_faces.append(exp.Current())
        exp.Next()

    face_to_id = {f: i for i, f in enumerate(ocp_faces)}

    print(f"Total Faces detected: {len(ocp_faces)}")

    edge_map = {}
    for fi, f in enumerate(ocp_faces):
        exp_e = TopExp_Explorer(f, TopAbs_EDGE)
        while exp_e.More():
            e = exp_e.Current()
            # エッジを一意に識別するため、内部のアドレスやreprをキーに使う
            try:
                key = e._from_address
            except Exception:
                key = getattr(e, '_address', repr(e))

            edge_map.setdefault(key, []).append(fi)
            exp_e.Next()

    adjacency_set = set()
    for faces in edge_map.values():
        if len(faces) >= 2:
            for a in range(len(faces)):
                for b in range(a + 1, len(faces)):
                    if faces[a] == faces[b]:
                        continue
                    pair = tuple(sorted((faces[a], faces[b])))
                    adjacency_set.add(pair)

    return sorted(list(adjacency_set))

if __name__ == "__main__":
    input_filename = sys.argv[1] if len(sys.argv) > 1 else "sample.step"
    
    adj_list = get_face_adjacency_with_cq(input_filename)

    print("-" * 30)
    print("Adjacency Graph (Node=Face ID, Edge=Connection):")
    print(f"Number of connections: {len(adj_list)}")
    print("-" * 30)
    
    for pair in adj_list[:20]:
        print(f"Face {pair[0]} <--> Face {pair[1]}")
    
    if len(adj_list) > 20:
        print("... (truncated)")