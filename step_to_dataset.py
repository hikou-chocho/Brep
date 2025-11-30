from step_to_graph import process_step_to_pyg as _process_step_to_pyg

# Backwards-compatibility wrapper
def process_step_to_pyg(filename, seg_path=None):
    # step_to_graph provides the implementation; this wrapper keeps the same API
    return _process_step_to_pyg(filename)
