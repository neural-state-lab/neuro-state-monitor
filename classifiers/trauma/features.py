"""PTSD-specific feature extraction.

Extracts features from fMRI connectivity matrices and EEG
for PTSD classification and reconsolidation window detection.

fMRI features:
- Resting-state connectivity matrix (upper triangle)
- Network-level connectivity (DMN, salience, executive)
- Graph theory metrics (efficiency, modularity)

EEG features (reconsolidation window):
- Theta power at retrieval (hippocampal engagement)
- P300 amplitude (memory retrieval strength)
- Alpha desynchronization (reactivation marker)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Default Mode Network ROIs (Schaefer atlas indices, approximate)
DMN_ROIS = list(range(40, 55))  # medial prefrontal, PCC, angular gyrus
SALIENCE_ROIS = list(range(55, 65))  # anterior insula, dACC
EXECUTIVE_ROIS = list(range(0, 20))  # dlPFC, posterior parietal


@dataclass
class TraumaFeatureConfig:
    """Configuration for PTSD feature extraction."""

    # fMRI connectivity features
    use_full_connectivity: bool = True
    use_network_connectivity: bool = True
    use_graph_metrics: bool = True

    # EEG reconsolidation features
    theta_range: tuple[float, float] = (4.0, 8.0)
    alpha_range: tuple[float, float] = (8.0, 13.0)
    p300_tmin: float = 0.25
    p300_tmax: float = 0.5


def extract_connectivity_features(
    connectivity_matrices: np.ndarray,
    config: Optional[TraumaFeatureConfig] = None,
) -> np.ndarray:
    """Extract features from fMRI connectivity matrices.

    Args:
        connectivity_matrices: Shape (n_subjects, n_rois, n_rois).

    Returns:
        Feature array of shape (n_subjects, n_features).
    """
    if config is None:
        config = TraumaFeatureConfig()

    feature_arrays: list[np.ndarray] = []

    if config.use_full_connectivity:
        # Upper triangle of connectivity matrix (flattened)
        n_rois = connectivity_matrices.shape[1]
        upper_idx = np.triu_indices(n_rois, k=1)
        flat_conn = np.array([m[upper_idx] for m in connectivity_matrices])
        feature_arrays.append(flat_conn)

    if config.use_network_connectivity:
        # Mean connectivity within and between major networks
        network_features = _compute_network_connectivity(connectivity_matrices)
        feature_arrays.append(network_features)

    if config.use_graph_metrics:
        # Graph theory metrics per subject
        graph_features = _compute_graph_metrics(connectivity_matrices)
        feature_arrays.append(graph_features)

    features = np.concatenate(feature_arrays, axis=1)

    logger.info(
        "extracted_trauma_features",
        shape=features.shape,
        n_subjects=features.shape[0],
    )
    return features


def _compute_network_connectivity(
    matrices: np.ndarray,
) -> np.ndarray:
    """Compute within- and between-network connectivity.

    Returns:
        Array of shape (n_subjects, n_network_features).
    """
    networks = {
        "dmn": DMN_ROIS,
        "salience": SALIENCE_ROIS,
        "executive": EXECUTIVE_ROIS,
    }

    features = []
    for matrix in matrices:
        subj_features = []

        # Within-network connectivity
        for name, rois in networks.items():
            valid_rois = [r for r in rois if r < matrix.shape[0]]
            if len(valid_rois) >= 2:
                submatrix = matrix[np.ix_(valid_rois, valid_rois)]
                upper = submatrix[np.triu_indices(len(valid_rois), k=1)]
                subj_features.append(np.mean(upper))
            else:
                subj_features.append(0.0)

        # Between-network connectivity
        network_names = list(networks.keys())
        for i in range(len(network_names)):
            for j in range(i + 1, len(network_names)):
                rois_i = [r for r in networks[network_names[i]] if r < matrix.shape[0]]
                rois_j = [r for r in networks[network_names[j]] if r < matrix.shape[0]]
                if rois_i and rois_j:
                    between = matrix[np.ix_(rois_i, rois_j)]
                    subj_features.append(np.mean(between))
                else:
                    subj_features.append(0.0)

        features.append(subj_features)

    return np.array(features)


def _compute_graph_metrics(
    matrices: np.ndarray,
) -> np.ndarray:
    """Compute graph theory metrics from connectivity matrices.

    Returns:
        Array of shape (n_subjects, n_graph_features).
    """
    features = []

    for matrix in matrices:
        # Threshold to binary adjacency (top 20% connections)
        threshold = np.percentile(np.abs(matrix[matrix != 0]), 80)
        binary = (np.abs(matrix) > threshold).astype(float)
        np.fill_diagonal(binary, 0)

        # Degree: number of connections per node
        degrees = binary.sum(axis=1)
        mean_degree = np.mean(degrees)

        # Clustering coefficient (local)
        clustering = _local_clustering(binary)

        # Global efficiency (inverse path length)
        efficiency = _global_efficiency(binary)

        features.append([mean_degree, clustering, efficiency])

    return np.array(features)


def _local_clustering(adj: np.ndarray) -> float:
    """Compute mean local clustering coefficient."""
    n = adj.shape[0]
    cc_values = []

    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue

        # Count triangles through node i
        subgraph = adj[np.ix_(neighbors, neighbors)]
        triangles = np.sum(subgraph) / 2
        possible = k * (k - 1) / 2
        cc_values.append(triangles / possible if possible > 0 else 0)

    return float(np.mean(cc_values)) if cc_values else 0.0


def _global_efficiency(adj: np.ndarray) -> float:
    """Compute global efficiency (mean inverse shortest path)."""
    n = adj.shape[0]
    if n < 2:
        return 0.0

    # BFS shortest paths
    from collections import deque

    total_inv = 0.0
    for source in range(n):
        dist = np.full(n, np.inf)
        dist[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for neighbor in np.where(adj[node] > 0)[0]:
                if dist[neighbor] == np.inf:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        for target in range(n):
            if target != source and dist[target] != np.inf:
                total_inv += 1.0 / dist[target]

    return total_inv / (n * (n - 1))
