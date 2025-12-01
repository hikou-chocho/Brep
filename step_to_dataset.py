import sys
import os
import math
import torch
import cadquery as cq
from torch_geometric.data import Data

# OCP Imports
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                         GeomAbs_Sphere, GeomAbs_Torus)
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Vec, gp_Pnt
from OCP.CSLib import CSLib
from OCP.ShapeAnalysis import ShapeAnalysis_Surface

SURFACE_TYPES = [GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus]
NUM_TYPES = len(SURFACE_TYPES)


def get_face_basic_features(ocp_face):
    """基本特徴量 v6: [Type(5), LogArea, NumWires, NumEdges, IsClosed(1)]"""
    adaptor = BRepAdaptor_Surface(ocp_face)
    try:
        s_type = adaptor.GetType()
    except Exception:
        s_type = None
    one_hot = [0.0] * NUM_TYPES
    if s_type in SURFACE_TYPES:
        one_hot[SURFACE_TYPES.index(s_type)] = 1.0

    props = GProp_GProps()
    try:
        BRepGProp.SurfaceProperties_s(ocp_face, props)
        area = props.Mass()
        # Guard: some bindings may return small negative values due to numerical noise
        if area is None:
            area = 0.0
        try:
            # coerce to float
            area = float(area)
        except Exception:
            area = 0.0
        if area < 0.0:
            area = 0.0
    except Exception:
        area = 0.0
    safe_area = max(area, 0.0)
    norm_area = math.log(safe_area + 1.0) * 0.1

    # Counts using Explorer for robustness
    n_wires = 0
    try:
        exp_w = TopExp_Explorer(ocp_face, TopAbs_WIRE)
        while exp_w.More():
            n_wires += 1
            exp_w.Next()
    except Exception:
        n_wires = 0

    n_edges = 0
    try:
        exp_e = TopExp_Explorer(ocp_face, TopAbs_EDGE)
        while exp_e.More():
            n_edges += 1
            exp_e.Next()
    except Exception:
        n_edges = 0

    # IsClosed: 円筒や球、トーラスなど UまたはV方向に閉じている面を検出
    try:
        is_closed_flag = 1.0 if (adaptor.IsUClosed() or adaptor.IsVClosed()) else 0.0
    except Exception:
        is_closed_flag = 0.0

    return one_hot + [norm_area, n_wires * 0.2, n_edges * 0.05, is_closed_flag]


def get_convexity(edge, f1, f2, debug=False):
    """凸(1), 凹(-1), その他(0)"""
    try:
        try:
            eedge = TopoDS.Edge_s(edge)
        except Exception:
            eedge = edge
        ca = BRepAdaptor_Curve(eedge)
        u_mid = (ca.FirstParameter() + ca.LastParameter()) / 2.0
        p = ca.Value(u_mid)
        tan = gp_Vec()
        ca.D1(u_mid, p, tan)
        try:
            if eedge.Orientation() == TopAbs_REVERSED:
                tan.Reverse()
        except Exception:
            pass

        def _get_n(face, pt):
            try:
                surf = BRepAdaptor_Surface(face)
                sa = ShapeAnalysis_Surface(surf.Surface().Surface())
                uv = sa.ValueOfUV(pt, 1e-3)
                ptmp = gp_Pnt()
                du = gp_Vec()
                dv = gp_Vec()
                surf.D1(uv.X(), uv.Y(), ptmp, du, dv)
                n = du.Crossed(dv)
                if face.Orientation() == TopAbs_REVERSED:
                    n.Reverse()
                return n
            except Exception:
                return None

        n1 = _get_n(f1, p)
        n2 = _get_n(f2, p)
        if n1 is None or n2 is None:
            if debug:
                print("get_convexity: failed to get normals (n1 or n2 is None)")
            return 0.0

        cp = n1.Crossed(n2)
        dot = cp.Dot(tan)

        if debug:
            print(f"get_convexity: dot={dot}")
        if dot > 1e-3:
            return 1.0
        elif dot < -1e-3:
            return -1.0
        return 0.0
    except Exception:
        if debug:
            import traceback
            print("get_convexity: exception:\n", traceback.format_exc())
        return 0.0


def process_step_to_pyg(filename, debug=False):
    if not os.path.exists(filename):
        return None
    try:
        model = cq.importers.importStep(filename)
        shape = model.val().wrapped
    except Exception:
        return None

    # 1. Face list & basic features
    ocp_faces = []
    face_feats = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        ocp_faces.append(f)
        face_feats.append(get_face_basic_features(f))
        exp.Next()

    if len(ocp_faces) == 0:
        return None

    # 2. Build adjacency by scanning shared edges between face pairs
    adj_src = []
    adj_dst = []
    edge_attr_list = []  # 新設: エッジごとの属性 (one-hot: [convex, concave, smooth])

    total_convex = 0
    total_concave = 0
    candidate_edges = 0
    matched_pairs = 0

    for i_a, fa in enumerate(ocp_faces):
        exp_e = TopExp_Explorer(fa, TopAbs_EDGE)
        while exp_e.More():
            ea = exp_e.Current()
            # compare with faces having larger index to avoid duplication
            for i_b in range(i_a + 1, len(ocp_faces)):
                fb = ocp_faces[i_b]
                # check if ea is also edge of fb
                exp_e2 = TopExp_Explorer(fb, TopAbs_EDGE)
                found = False
                while exp_e2.More():
                    eb = exp_e2.Current()
                    try:
                        if ea.IsSame(eb):
                            found = True
                            break
                    except Exception:
                        try:
                            if eb.IsSame(ea):
                                found = True
                                break
                        except Exception:
                            pass
                    exp_e2.Next()

                if found:
                    # adjacency (bidirectional)
                    adj_src.extend([i_a, i_b])
                    adj_dst.extend([i_b, i_a])
                    candidate_edges += 1
                    # compute convexity for this shared edge
                    cvx = get_convexity(ea, fa, fb, debug=debug)
                    if debug and cvx != 0.0:
                        print(f"Edge between faces {i_a}-{i_b}: convexity={cvx}")

                    # Edge attribute: one-hot [Convex, Concave, Smooth]
                    attr = [0.0, 0.0, 0.0]
                    if cvx > 0:
                        attr[0] = 1.0
                        total_convex += 1
                    elif cvx < 0:
                        attr[1] = 1.0
                        total_concave += 1
                    else:
                        attr[2] = 1.0

                    # add attribute for both directions
                    edge_attr_list.extend([attr, attr])
                    matched_pairs += 1
                # next candidate fb
            exp_e.Next()

    if debug:
        print(f"Debug {filename}: Convex={total_convex}, Concave={total_concave}")
        print(f"Candidate edges(found by shared-edge scan): {candidate_edges}, Matched pairs: {matched_pairs}")

    # 3. Combine features
    # Node features: 基本特徴量のみ（凸凹のカウントは廃止し、エッジ属性へ移行）
    final_x = face_feats
    x = torch.tensor(final_x, dtype=torch.float)
    if len(adj_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([adj_src, adj_dst], dtype=torch.long)

    # edge_attr を作成（存在しない場合は空テンソル）
    if len(edge_attr_list) == 0:
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

