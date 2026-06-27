from .homogeneous import (
    build_passing_network, build_all_networks_for_match,
    compute_network_stats, save_network, load_network, network_to_edge_list,
)
from .temporal import (
    get_possession_units, build_score_timeline,
    get_score_state_at_minute, get_period_label, build_player_team_map,
)
from .spatial import normalize_coords, zone_index, zone_onehot
from ..data.schema import get_zone
from .node_features import build_all_node_features, NODE_FEATURE_DIM
from .edge_features import build_pass_edge_feature, PASS_EDGE_DIM, ADV_EDGE_DIM, TURN_EDGE_DIM

# heterogeneous.py requires torch / torch_geometric.
# Import it only when actually needed (Task 3/9) to avoid crashing on machines
# where PyTorch is not installed.
def _import_heterogeneous():
    from .heterogeneous import build_heterogeneous_graph, save_hetero_graph, load_hetero_graph
    return build_heterogeneous_graph, save_hetero_graph, load_hetero_graph