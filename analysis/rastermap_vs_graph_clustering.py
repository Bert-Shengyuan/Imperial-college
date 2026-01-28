"""
Rastermap vs Graph-Based Clustering Comparison Analysis

This script:
1. Loads firing rate data (fam1, nov, fam2/fam1r2) from wild-type mice
2. Applies Rastermap clustering to identify neural clusters
3. Compares Rastermap clusters with graph-based (PyGenStability) clusters
4. Uses Rastermap cluster indices to mask the original graph
5. Runs the same graph analyses from figure_graph_sub2.0.py on the masked subgraphs

Author: Auto-generated
Date: 2024
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
import pandas as pd
from pathlib import Path
from collections import deque
import math
import random
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score, homogeneity_completeness_v_measure
from scipy.stats import spearmanr, wilcoxon, ttest_1samp
from scipy.cluster.hierarchy import linkage, leaves_list
import warnings

try:
    from rastermap import Rastermap
except ImportError:
    import subprocess

    subprocess.check_call(['pip', 'install', 'rastermap'])
    from rastermap import Rastermap

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Update these paths to match your data location
# =============================================================================

# Base data directory (update this path for your system)
# BASE_DATA_DIR = Path('/Users/sonmjack/Downloads')
# Alternative path format:
BASE_DATA_DIR = Path('/Users/shengyuancai/Downloads/Imperial paper/Data')

# Data paths for wild-type young animals (age2)
DATA_PATHS = {
    'fam1': {
        'epsp': BASE_DATA_DIR / 'age2 result_fam1' / 'fam1_all_EPSP_young.pkl',
        'markov': BASE_DATA_DIR / 'age2 result_fam1' / 'fam1_Signal_Markov_{idx}.pkl',
        'spike': BASE_DATA_DIR / 'age2 result_fam1' / 'fam1_S_spike.pkl',
        'df_f': BASE_DATA_DIR / 'age2 result_fam1' / 'fam1_S_df_f.pkl',
        'tuning': BASE_DATA_DIR / 'age2 result_fam1' / 'fam1_S_tuning curve.pkl',
    },
    'nov': {
        'epsp': BASE_DATA_DIR / 'age2 result_nov' / 'nov_all_EPSP_young.pkl',
        'markov': BASE_DATA_DIR / 'age2 result_nov' / 'nov_Signal_Markov_{idx}.pkl',
        'spike': BASE_DATA_DIR / 'age2 result_nov' / 'nov_S_spike.pkl',
        'df_f': BASE_DATA_DIR / 'age2 result_nov' / 'nov_S_df_f.pkl',
        'tuning': BASE_DATA_DIR / 'age2 result_nov' / 'nov_S_tuning curve.pkl',
    },
    'fam2': {  # fam1r2 is referred to as fam2
        'epsp': BASE_DATA_DIR / 'age2 result_fam1r2' / 'fam1r2_all_EPSP_young.pkl',
        'markov': BASE_DATA_DIR / 'age2 result_fam1r2' / 'fam1r2_Signal_Markov_{idx}.pkl',
        'spike': BASE_DATA_DIR / 'age2 result_fam1r2' / 'fam1r2_S_spike.pkl',
        'df_f': BASE_DATA_DIR / 'age2 result_fam1r2' / 'fam1r2_S_df_f.pkl',
        'tuning': BASE_DATA_DIR / 'age2 result_fam1r2' / 'fam1r2_S_tuning curve.pkl',
    }
}

# Trigger file for identifying wild-type animals (gene code 119)
#TRIGGER_FILE = BASE_DATA_DIR / 'simon_paper' / 'shengyuan_trigger_fam1.npy'
# Alternative:
TRIGGER_FILE = BASE_DATA_DIR / 'Raw data' / 'shengyuan_trigger_fam1.npy'

# Output directory for results
OUTPUT_DIR = Path(BASE_DATA_DIR/'rastermap_analysis_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# Wild-type gene code
WT_GENE_CODE = 119

# =============================================================================
# UTILITY FUNCTIONS (from figure_graph_sub2.0.py)
# =============================================================================

def Connector(Q):
    """Ensure graph connectivity by adding minimal edges to isolated nodes."""
    D = nx.to_networkx_graph(Q, create_using=nx.DiGraph())
    Isolate_list = list(nx.isolates(D))
    if len(Isolate_list) > 0:
        for i in Isolate_list:
            if i == 0:
                Q[i + 1, i] = 0.0001
            else:
                Q[i - 1, i] = 0.0001
    del D
    Q = nx.to_networkx_graph(Q, create_using=nx.DiGraph())
    return Q


def normal(A):
    """Min-max normalize matrix A, setting diagonal to zero."""
    np.fill_diagonal(A, 0)
    min_val = np.min(A)
    max_val = np.max(A)
    if max_val - min_val > 0:
        A = (A - min_val) / (max_val - min_val)
    return A


def sparse(A, k=10):
    """Apply KNN-based sparsification to adjacency matrix A."""
    N = A.shape[0]
    np.fill_diagonal(A, 0)
    A = normal(A)

    # Row-wise top-k
    B1 = np.zeros((N, N))
    for i in range(N):
        W = sorted(A[i, :], reverse=True)
        threshold = W[min(k, N-1)]
        B1[i, :] = np.where(A[i, :] > threshold, 1, 0)

    # Column-wise top-k
    C1 = np.zeros((N, N))
    for i in range(N):
        W = sorted(A[:, i], reverse=True)
        threshold = W[min(k, N-1)]
        C1[:, i] = np.where(A[:, i] > threshold, 1, 0)

    Q1 = B1 + C1
    Q2 = np.where(Q1 > 0.9, 1, 0)
    Q = np.multiply(Q2, A)

    # Handle isolated rows
    for i in range(Q.shape[0]):
        if np.all(Q[i] == 0):
            random_index = np.random.randint(0, Q.shape[1])
            Q[i, random_index] = 0.001

    Q = Connector(Q)
    return Q


def find_shortest_path(graph, start, end):
    """Find shortest path using BFS."""
    n = len(graph)
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        if node in visited:
            continue
        visited.add(node)

        for neighbor in range(n):
            if graph[node][neighbor] == 1 and neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None


def calculate_average_path_length(graph):
    """Calculate average path length and local efficiency."""
    n = len(graph)
    dist = np.where(graph == 1, 1, np.inf)
    np.fill_diagonal(dist, 0)

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    finite_paths = dist[np.isfinite(dist)]
    if len(finite_paths) > 0:
        average_path_length = np.sum(finite_paths) / len(finite_paths)
        local_efficiency = 1 / average_path_length if average_path_length > 0 else 0
        return average_path_length, local_efficiency
    else:
        return 0, 0


def find_reciprocal(G):
    """Find proportion of reciprocal edges."""
    reciprocal = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
    n_nodes = len(G.nodes)
    if n_nodes < 2:
        return 0
    permutations_count = n_nodes * (n_nodes - 1)
    reciprocal_len = len(reciprocal) / permutations_count
    return reciprocal_len


def find_divergent(G):
    """Find proportion of divergent nodes (out-degree >= 2)."""
    if len(G.nodes) == 0:
        return 0
    divergent = [n for n in G.nodes() if G.out_degree(n) >= 2]
    return len(divergent) / len(G.nodes)


def find_convergent(G):
    """Find proportion of convergent nodes (in-degree >= 2)."""
    if len(G.nodes) == 0:
        return 0
    convergent = [n for n in G.nodes() if G.in_degree(n) >= 2]
    return len(convergent) / len(G.nodes)


def find_chain(G):
    """Find proportion of chain motifs."""
    n_nodes = len(G.nodes)
    if n_nodes < 3:
        return 0
    chain = [(u, v, w) for u in G.nodes() for v in G.successors(u) for w in G.successors(v) if u != w]
    permutations_count = n_nodes * (n_nodes - 1) * (n_nodes - 2)
    chain_len = len(chain) / permutations_count if permutations_count > 0 else 0
    return chain_len


def build_graph(g, Community):
    """
    Build within-community and between-community subgraphs and compute metrics.

    Returns metrics for both within-community (W) and between-community (B) subgraphs.
    """
    t_p_G = g.copy()
    N = t_p_G.shape[0]

    # Sparsify and convert to networkx graph
    t_p_G = sparse(t_p_G)

    if isinstance(t_p_G, np.ndarray):
        D = nx.to_networkx_graph(t_p_G, create_using=nx.DiGraph())
    else:
        D = t_p_G.copy()

    B = D.copy()
    Between = D.copy()
    Within = D.copy()

    Community = Community.tolist() if hasattr(Community, 'tolist') else list(Community)

    # Set node community attributes
    for node in B.nodes:
        B.nodes[node]["community"] = Community[node]

    # Create community subgraphs
    for x in range(max(Community) + 1):
        selected_nodes = [n for n, v in B.nodes(data=True) if v["community"] == x]
        globals()[f'G{x}'] = B.subgraph(selected_nodes)

    # Remove within-community edges from Between graph
    for x in range(max(Community) + 1):
        List_edge = list(globals()[f'G{x}'].edges())
        Between.remove_edges_from(List_edge)

    # Remove between-community edges from Within graph
    List_edge1 = list(Between.edges())
    Within.remove_edges_from(List_edge1)

    # Calculate metrics for Within-community graph
    Wi = nx.to_numpy_array(Within)
    Be = nx.to_numpy_array(Between)

    local_cost_W = np.sum(Wi)
    local_cost_B = np.sum(Be)

    # Clustering coefficient (symmetrize first)
    upper_triangular = np.triu(Wi)
    lower_triangular = np.tril(Wi)
    symmetric_upper = upper_triangular + upper_triangular.T - np.diag(np.diag(upper_triangular))
    symmetric_lower = lower_triangular + lower_triangular.T - np.diag(np.diag(lower_triangular))

    G1 = nx.Graph(symmetric_upper)
    avg_clustering1 = nx.average_clustering(G1) if len(G1.nodes) > 0 else 0
    G2 = nx.Graph(symmetric_lower)
    avg_clustering2 = nx.average_clustering(G2) if len(G2.nodes) > 0 else 0
    avg_clustering_W = (avg_clustering1 + avg_clustering2) / 2

    # Binarize and calculate path metrics
    Wi_binary = np.where(Wi >= 0.05, 1, 0)
    avg_path_length_W, local_efficiency_W = calculate_average_path_length(Wi_binary)

    # Motif analysis
    G_all = nx.DiGraph(Wi_binary)
    chain_W = find_chain(G_all)
    convergent_W = find_convergent(G_all)
    divergent_W = find_divergent(G_all)
    reciprocal_W = find_reciprocal(G_all)

    # Between-community metrics
    upper_triangular = np.triu(Be)
    lower_triangular = np.tril(Be)
    symmetric_upper = upper_triangular + upper_triangular.T - np.diag(np.diag(upper_triangular))
    symmetric_lower = lower_triangular + lower_triangular.T - np.diag(np.diag(lower_triangular))

    G1 = nx.Graph(symmetric_upper)
    avg_clustering1 = nx.average_clustering(G1) if len(G1.nodes) > 0 else 0
    G2 = nx.Graph(symmetric_lower)
    avg_clustering2 = nx.average_clustering(G2) if len(G2.nodes) > 0 else 0
    avg_clustering_B = (avg_clustering1 + avg_clustering2) / 2

    Be_binary = np.where(Be >= 0.05, 1, 0)
    avg_path_length_B, local_efficiency_B = calculate_average_path_length(Be_binary)

    G_all = nx.DiGraph(Be_binary)
    chain_B = find_chain(G_all)
    convergent_B = find_convergent(G_all)
    divergent_B = find_divergent(G_all)
    reciprocal_B = find_reciprocal(G_all)

    return {
        'within': {
            'clustering': avg_clustering_W,
            'path_length': avg_path_length_W,
            'chain': chain_W,
            'convergent': convergent_W,
            'divergent': divergent_W,
            'reciprocal': reciprocal_W,
            'efficiency': local_efficiency_W,
            'cost': local_cost_W
        },
        'between': {
            'clustering': avg_clustering_B,
            'path_length': avg_path_length_B,
            'chain': chain_B,
            'convergent': convergent_B,
            'divergent': divergent_B,
            'reciprocal': reciprocal_B,
            'efficiency': local_efficiency_B,
            'cost': local_cost_B
        }
    }


# =============================================================================
# RASTERMAP CLUSTERING FUNCTIONS
# =============================================================================

def apply_rastermap(firing_rates, n_clusters=None, n_components=1):
    """
    Apply Rastermap to firing rate matrix to identify neural clusters.

    Parameters:
    -----------
    firing_rates : np.ndarray
        Firing rate matrix (neurons x time) or (neurons x neurons) for connectivity
    n_clusters : int, optional
        Number of clusters to identify. If None, determined automatically.
    n_components : int
        Number of components for embedding

    Returns:
    --------
    cluster_labels : np.ndarray
        Cluster assignment for each neuron
    isort : np.ndarray
        Sorting indices from rastermap
    embedding : np.ndarray
        Low-dimensional embedding
    """
    try:
        from rastermap import Rastermap
    except ImportError:
        print("Rastermap not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'rastermap'])
        from rastermap import Rastermap

    # Initialize Rastermap
    model = Rastermap(
        n_components=n_components,
        n_X=40,  # number of smoothing bins in x
        n_Y=40,  # number of smoothing bins in y
        nbin=50,  # number of bins for clustering
        alpha=1.0,
        K=1.0,
        init='pca'
    )

    # Fit the model
    if firing_rates.ndim == 1:
        firing_rates = firing_rates.reshape(-1, 1)

    # Rastermap expects (neurons x time) data
    embedding = model.fit_transform(firing_rates)
    isort = model.isort

    # Extract clusters from the sorted indices
    n_neurons = len(isort)

    if n_clusters is None:
        # Estimate number of clusters based on data
        n_clusters = max(2, min(10, n_neurons // 20))

    # Assign clusters based on sorted position
    cluster_labels = np.zeros(n_neurons, dtype=int)
    cluster_size = n_neurons // n_clusters
    for i in range(n_clusters):
        start_idx = i * cluster_size
        end_idx = (i + 1) * cluster_size if i < n_clusters - 1 else n_neurons
        neurons_in_cluster = isort[start_idx:end_idx]
        cluster_labels[neurons_in_cluster] = i

    return cluster_labels, isort, embedding


def rastermap_clustering_from_correlation(connectivity_matrix, n_clusters=None):
    """
    Apply Rastermap clustering using the connectivity/correlation matrix.

    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Neuron x Neuron connectivity or correlation matrix
    n_clusters : int, optional
        Number of clusters

    Returns:
    --------
    cluster_labels : np.ndarray
        Cluster assignment for each neuron
    isort : np.ndarray
        Rastermap sorting indices
    """

    # Use the connectivity matrix as features
    # Each neuron's row represents its connectivity profile
    model = Rastermap(n_PCs=n_clusters,
                      locality=0,
                      grid_upsample=5) # 0.5 1 
    # Fit using connectivity profiles
    embedding = model.fit(connectivity_matrix)
    isort = model.isort

    n_neurons = len(isort)
    if n_clusters is None:
        n_clusters = max(2, min(10, n_neurons // 20))

    # Assign clusters based on sorted position
    cluster_labels = np.zeros(n_neurons, dtype=int)
    cluster_size = n_neurons // n_clusters
    for i in range(n_clusters):
        start_idx = i * cluster_size
        end_idx = (i + 1) * cluster_size if i < n_clusters - 1 else n_neurons
        neurons_in_cluster = isort[start_idx:end_idx]
        cluster_labels[neurons_in_cluster] = i

    return cluster_labels, isort, embedding


# =============================================================================
# CLUSTERING COMPARISON FUNCTIONS
# =============================================================================

def compare_clusterings(labels1, labels2, name1='Graph-based', name2='Rastermap'):
    """
    Compare two clustering results using multiple metrics.

    Parameters:
    -----------
    labels1, labels2 : np.ndarray
        Cluster labels from two methods
    name1, name2 : str
        Names for reporting

    Returns:
    --------
    metrics : dict
        Dictionary containing all comparison metrics
    """
    # Ensure same length
    assert len(labels1) == len(labels2), "Label arrays must have same length"

    # Adjusted Rand Index
    ari = adjusted_rand_score(labels1, labels2)

    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(labels1, labels2)

    # Fowlkes-Mallows Index
    fmi = fowlkes_mallows_score(labels1, labels2)

    # Homogeneity, Completeness, V-measure
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels1, labels2)

    # Number of clusters
    n_clusters_1 = len(np.unique(labels1))
    n_clusters_2 = len(np.unique(labels2))

    metrics = {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'fowlkes_mallows_index': fmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        f'n_clusters_{name1}': n_clusters_1,
        f'n_clusters_{name2}': n_clusters_2
    }

    return metrics


def create_confusion_matrix(labels1, labels2, name1='Graph-based', name2='Rastermap'):
    """Create a confusion matrix between two clustering results."""
    unique1 = np.unique(labels1)
    unique2 = np.unique(labels2)

    confusion = np.zeros((len(unique1), len(unique2)))

    for i, c1 in enumerate(unique1):
        for j, c2 in enumerate(unique2):
            confusion[i, j] = np.sum((labels1 == c1) & (labels2 == c2))

    return confusion, unique1, unique2


# =============================================================================
# ORDER CORRELATION AND VISUALIZATION FUNCTIONS
# =============================================================================

def get_graph_markov_order(graph_labels):
    """
    Get neuron ordering based on graph-based Markov clustering.

    Neurons are sorted first by cluster label, then by their original index
    within each cluster. This provides a canonical ordering that groups
    neurons by their community membership.

    Parameters:
    -----------
    graph_labels : np.ndarray
        Cluster labels from graph-based Markov clustering

    Returns:
    --------
    order : np.ndarray
        Indices that sort neurons by cluster membership
    """
    n_neurons = len(graph_labels)
    # Sort neurons by cluster label, maintaining original order within clusters
    order = np.lexsort((np.arange(n_neurons), graph_labels))
    return order


def calculate_order_correlation(rastermap_isort, graph_labels):
    """
    Calculate correlation between rastermap order and graph-based Markov order.

    Uses Spearman rank correlation to compare the orderings from both methods.

    Parameters:
    -----------
    rastermap_isort : np.ndarray
        Sorting indices from rastermap (neuron indices in sorted order)
    graph_labels : np.ndarray
        Cluster labels from graph-based Markov clustering

    Returns:
    --------
    correlation_results : dict
        Dictionary containing correlation metrics:
        - spearman_rho: Spearman correlation coefficient
        - spearman_pvalue: p-value for the correlation
        - rank_rastermap: Rank positions from rastermap ordering
        - rank_graph: Rank positions from graph-based ordering
    """
    n_neurons = len(rastermap_isort)

    # Get rastermap ranks (position in sorted order for each neuron)
    rank_rastermap = np.zeros(n_neurons, dtype=int)
    rank_rastermap[rastermap_isort] = np.arange(n_neurons)

    # Get graph-based Markov ordering
    graph_order = get_graph_markov_order(graph_labels)
    rank_graph = np.zeros(n_neurons, dtype=int)
    rank_graph[graph_order] = np.arange(n_neurons)

    # Calculate Spearman correlation
    rho, pvalue = spearmanr(rank_rastermap, rank_graph)

    return {
        'spearman_rho': rho,
        'spearman_pvalue': pvalue,
        'rank_rastermap': rank_rastermap,
        'rank_graph': rank_graph,
        'rastermap_order': rastermap_isort,
        'graph_order': graph_order
    }


def visualize_firing_rates_dual_ordering(firing_rate_matrix, rastermap_isort, graph_labels,
                                         session_id, env_name, output_dir=OUTPUT_DIR,
                                         figsize=(14, 6)):
    """
    Visualize firing rates under rastermap order and graph-based Markov order side by side.

    Parameters:
    -----------
    firing_rate_matrix : np.ndarray
        Firing rate matrix (neurons x time) or connectivity matrix (neurons x neurons)
    rastermap_isort : np.ndarray
        Sorting indices from rastermap
    graph_labels : np.ndarray
        Cluster labels from graph-based Markov clustering
    session_id : int
        Session identifier
    env_name : str
        Environment name
    output_dir : Path
        Output directory for saving figures
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig_path : Path
        Path to the saved figure
    """
    output_dir = Path(output_dir)

    # Get graph-based ordering
    graph_order = get_graph_markov_order(graph_labels)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Prepare firing rate data
    data = firing_rate_matrix.copy()
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Normalize data to [0, 1] range for raster-style visualization
    # Use percentile-based normalization to handle outliers
    vmin_pct, vmax_pct = np.percentile(data, [1, 99])
    if vmax_pct > vmin_pct:
        data_normalized = (data - vmin_pct) / (vmax_pct - vmin_pct)
    else:
        data_normalized = data - data.min()
        if data_normalized.max() > 0:
            data_normalized = data_normalized / data_normalized.max()

    # Clip to [0, 1]
    data_normalized = np.clip(data_normalized, 0, 1)

    # Left panel: Rastermap ordering
    data_rastermap = data_normalized[rastermap_isort, :]
    im1 = axes[0].imshow(data_rastermap, aspect='auto', cmap='gray_r',
                         vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title(f'Rastermap Order\n{env_name} - Session {session_id}', fontsize=12)
    axes[0].set_xlabel('Time / Features', fontsize=10)
    axes[0].set_ylabel('Neurons (sorted)', fontsize=10)

    # Right panel: Graph-based Markov ordering
    data_graph = data_normalized[graph_order, :]
    im2 = axes[1].imshow(data_graph, aspect='auto', cmap='gray_r',
                         vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title(f'Graph-based Markov Order\n{env_name} - Session {session_id}', fontsize=12)
    axes[1].set_xlabel('Time / Features', fontsize=10)
    axes[1].set_ylabel('Neurons (sorted)', fontsize=10)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f'firing_rates_dual_order_{env_name}_session{session_id}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return fig_path


def visualize_all_sessions_firing_rates(all_results, firing_rates_dict, output_dir=OUTPUT_DIR):
    """
    Generate firing rate visualizations for all sessions across environments.

    Parameters:
    -----------
    all_results : dict
        Results dictionary from run_full_analysis
    firing_rates_dict : dict
        Dictionary mapping environment names to firing rate data
    output_dir : Path
        Output directory

    Returns:
    --------
    figure_paths : list
        List of paths to generated figures
    """
    output_dir = Path(output_dir)
    figure_paths = []

    for env, sessions in all_results.items():
        if not sessions:
            continue

        firing_rates = firing_rates_dict.get(env)
        if firing_rates is None:
            print(f"Warning: No firing rate data for {env}")
            continue

        for session in sessions:
            session_id = session['session_id']

            if session_id >= len(firing_rates):
                print(f"Warning: Session {session_id} out of range for {env}")
                continue

            firing_rate_matrix = firing_rates[session_id]
            rastermap_isort = session['rastermap_isort']
            graph_labels = session['graph_labels']

            fig_path = visualize_firing_rates_dual_ordering(
                firing_rate_matrix,
                rastermap_isort,
                graph_labels,
                session_id,
                env,
                output_dir
            )
            figure_paths.append(fig_path)
            print(f"  Saved: {fig_path}")

    return figure_paths


def compute_order_correlations_all_sessions(all_results):
    """
    Compute order correlations between rastermap and graph-based Markov
    clustering for all sessions.

    Parameters:
    -----------
    all_results : dict
        Results dictionary from run_full_analysis

    Returns:
    --------
    correlation_data : list
        List of dictionaries containing correlation data for each session
    """
    correlation_data = []

    for env, sessions in all_results.items():
        for session in sessions:
            rastermap_isort = session['rastermap_isort']
            graph_labels = session['graph_labels']

            corr_results = calculate_order_correlation(rastermap_isort, graph_labels)

            correlation_data.append({
                'environment': env,
                'session_id': session['session_id'],
                'n_neurons': session['n_neurons'],
                'spearman_rho': corr_results['spearman_rho'],
                'spearman_pvalue': corr_results['spearman_pvalue'],
                'n_clusters_graph': len(np.unique(graph_labels)),
                'n_clusters_rastermap': len(np.unique(session['rastermap_labels']))
            })

    return correlation_data


def statistical_analysis_order_correlation(correlation_data, output_dir=OUTPUT_DIR):
    """
    Perform statistical analysis on order correlations across sessions.

    Tests whether the correlation is significantly different from zero
    across all sessions.

    Parameters:
    -----------
    correlation_data : list
        List of correlation dictionaries from compute_order_correlations_all_sessions
    output_dir : Path
        Output directory for saving results

    Returns:
    --------
    stats_results : dict
        Dictionary containing statistical results
    """
    output_dir = Path(output_dir)

    if not correlation_data:
        print("No correlation data available for statistical analysis")
        return None

    df = pd.DataFrame(correlation_data)

    # Overall statistics
    rho_values = df['spearman_rho'].values

    stats_results = {
        'n_sessions': len(rho_values),
        'mean_rho': np.mean(rho_values),
        'std_rho': np.std(rho_values),
        'median_rho': np.median(rho_values),
        'min_rho': np.min(rho_values),
        'max_rho': np.max(rho_values)
    }

    # One-sample t-test: is mean correlation different from zero?
    if len(rho_values) >= 2:
        t_stat, t_pvalue = ttest_1samp(rho_values, 0)
        stats_results['ttest_statistic'] = t_stat
        stats_results['ttest_pvalue'] = t_pvalue

    # Wilcoxon signed-rank test (non-parametric alternative)
    if len(rho_values) >= 5:
        try:
            w_stat, w_pvalue = wilcoxon(rho_values)
            stats_results['wilcoxon_statistic'] = w_stat
            stats_results['wilcoxon_pvalue'] = w_pvalue
        except ValueError:
            # Wilcoxon test requires non-zero differences
            stats_results['wilcoxon_statistic'] = np.nan
            stats_results['wilcoxon_pvalue'] = np.nan

    # Per-environment statistics
    env_stats = {}
    for env in df['environment'].unique():
        env_df = df[df['environment'] == env]
        env_rho = env_df['spearman_rho'].values
        env_stats[env] = {
            'n_sessions': len(env_rho),
            'mean_rho': np.mean(env_rho),
            'std_rho': np.std(env_rho),
            'median_rho': np.median(env_rho)
        }
    stats_results['per_environment'] = env_stats

    return stats_results, df


def plot_order_correlation_results(correlation_data, stats_results, output_dir=OUTPUT_DIR):
    """
    Generate plots for order correlation analysis.

    Parameters:
    -----------
    correlation_data : list
        List of correlation dictionaries
    stats_results : dict
        Statistical results from statistical_analysis_order_correlation
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)

    if not correlation_data:
        print("No correlation data to plot")
        return

    df = pd.DataFrame(correlation_data)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Box plot of Spearman rho by environment
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='environment', y='spearman_rho', ax=ax1, palette='Set2')
    sns.swarmplot(data=df, x='environment', y='spearman_rho', ax=ax1,
                  color='black', alpha=0.6, size=8)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, label='No correlation')
    ax1.set_xlabel('Environment', fontsize=11)
    ax1.set_ylabel('Spearman ρ', fontsize=11)
    ax1.set_title('Order Correlation: Rastermap vs Graph-based Markov', fontsize=12)
    ax1.legend(loc='upper right')

    # Add significance annotation
    if 'ttest_pvalue' in stats_results:
        pval = stats_results['ttest_pvalue']
        sig_text = f'p = {pval:.4f}' if pval >= 0.0001 else 'p < 0.0001'
        if pval < 0.05:
            sig_text += ' *'
        if pval < 0.01:
            sig_text += '*'
        if pval < 0.001:
            sig_text += '*'
        ax1.text(0.02, 0.98, f't-test vs 0: {sig_text}', transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Histogram of correlation values
    ax2 = axes[0, 1]
    ax2.hist(df['spearman_rho'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=np.mean(df['spearman_rho']), color='red', linestyle='-',
                linewidth=2, label=f'Mean = {np.mean(df["spearman_rho"]):.3f}')
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Zero')
    ax2.set_xlabel('Spearman ρ', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of Order Correlations', fontsize=12)
    ax2.legend(loc='upper right')

    # Plot 3: Correlation vs number of neurons
    ax3 = axes[1, 0]
    colors = {'fam1': 'blue', 'nov': 'orange', 'fam2': 'green'}
    for env in df['environment'].unique():
        env_df = df[df['environment'] == env]
        ax3.scatter(env_df['n_neurons'], env_df['spearman_rho'],
                   label=env, s=80, alpha=0.7, c=colors.get(env, 'gray'))
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Number of Neurons', fontsize=11)
    ax3.set_ylabel('Spearman ρ', fontsize=11)
    ax3.set_title('Order Correlation vs Network Size', fontsize=12)
    ax3.legend(title='Environment')

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary table
    table_data = [
        ['Metric', 'Value'],
        ['N Sessions', f'{stats_results["n_sessions"]}'],
        ['Mean ρ', f'{stats_results["mean_rho"]:.4f}'],
        ['Std ρ', f'{stats_results["std_rho"]:.4f}'],
        ['Median ρ', f'{stats_results["median_rho"]:.4f}'],
        ['Range', f'[{stats_results["min_rho"]:.4f}, {stats_results["max_rho"]:.4f}]'],
    ]

    if 'ttest_pvalue' in stats_results:
        table_data.append(['t-test p-value', f'{stats_results["ttest_pvalue"]:.4e}'])
    if 'wilcoxon_pvalue' in stats_results and not np.isnan(stats_results['wilcoxon_pvalue']):
        table_data.append(['Wilcoxon p-value', f'{stats_results["wilcoxon_pvalue"]:.4e}'])

    table = ax4.table(cellText=table_data, loc='center', cellLoc='left',
                      colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('Statistical Summary', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'order_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Order correlation plot saved to {output_dir / 'order_correlation_analysis.png'}")

    # Save detailed results to CSV
    df.to_csv(output_dir / 'order_correlation_results.csv', index=False)
    print(f"Detailed results saved to {output_dir / 'order_correlation_results.csv'}")


def print_statistical_summary(stats_results):
    """Print a formatted statistical summary to console."""
    print("\n" + "=" * 70)
    print("ORDER CORRELATION STATISTICAL SUMMARY")
    print("Rastermap Order vs Graph-based Markov Order")
    print("=" * 70)

    print(f"\nOverall Statistics (N = {stats_results['n_sessions']} sessions):")
    print(f"  Mean Spearman ρ:   {stats_results['mean_rho']:.4f} ± {stats_results['std_rho']:.4f}")
    print(f"  Median Spearman ρ: {stats_results['median_rho']:.4f}")
    print(f"  Range:             [{stats_results['min_rho']:.4f}, {stats_results['max_rho']:.4f}]")

    print("\nStatistical Tests (H0: ρ = 0):")
    if 'ttest_pvalue' in stats_results:
        pval = stats_results['ttest_pvalue']
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
        print(f"  One-sample t-test: t = {stats_results['ttest_statistic']:.3f}, p = {pval:.4e} {sig}")

    if 'wilcoxon_pvalue' in stats_results and not np.isnan(stats_results['wilcoxon_pvalue']):
        pval = stats_results['wilcoxon_pvalue']
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
        print(f"  Wilcoxon signed-rank: W = {stats_results['wilcoxon_statistic']:.1f}, p = {pval:.4e} {sig}")

    print("\nPer-Environment Statistics:")
    for env, env_stats in stats_results['per_environment'].items():
        print(f"  {env.upper()}:")
        print(f"    N = {env_stats['n_sessions']}, Mean ρ = {env_stats['mean_rho']:.4f} ± {env_stats['std_rho']:.4f}")

    print("\n" + "=" * 70)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_firing_rates(env):
    """
    Load firing rate (EPSP) data for a given environment.

    Parameters:
    -----------
    env : str
        Environment name ('fam1', 'nov', or 'fam2')

    Returns:
    --------
    firing_rates_list : list
        List of firing rate matrices for each session
    """
    path = DATA_PATHS[env]['spike']
    if not path.exists():
        print(f"Warning: {path} not found")
        return None

    with open(path, 'rb') as f:
        firing_rates = pickle.load(f)

    return firing_rates

def load_epsp_results(env):
    """
    Load firing rate (EPSP) data for a given environment.

    Parameters:
    -----------
    env : str
        Environment name ('fam1', 'nov', or 'fam2')

    Returns:
    --------
    firing_rates_list : list
        List of firing rate matrices for each session
    """
    path = DATA_PATHS[env]['epsp']
    if not path.exists():
        print(f"Warning: {path} not found")
        return None

    with open(path, 'rb') as f:
        epsp_results = pickle.load(f)

    return epsp_results

def load_graph_clustering(env, session_idx):
    """
    Load graph-based (Markov) clustering results.

    Parameters:
    -----------
    env : str
        Environment name
    session_idx : int
        Session index for wild-type animal

    Returns:
    --------
    all_results : dict
        Clustering results containing 'community_id' and 'selected_partitions'
    """
    path_template = str(DATA_PATHS[env]['markov'])
    path = Path(path_template.format(idx=session_idx))

    if not path.exists():
        print(f"Warning: {path} not found")
        return None

    with open(path, 'rb') as f:
        all_results = pickle.load(f)

    return all_results


def get_wildtype_indices(trigger_file=TRIGGER_FILE):
    """
    Get indices of wild-type animals from trigger file.

    Returns:
    --------
    wt_indices : list
        List of (global_index, wt_counter) tuples for wild-type animals
    gene_list : list
        List of gene codes for each session
    """
    if not Path(trigger_file).exists():
        print(f"Warning: Trigger file {trigger_file} not found")
        return [], []

    mat_trigger = np.load(trigger_file)

    gene_list_young = []
    wt_indices = []
    wt_counter = 0

    # Young animals: indices 10-46 (step 2), excluding 18
    for i in range(10, 46, 2):
        if i == len(mat_trigger):
            break
        if i == 18:
            continue

        gene_code = mat_trigger[i, 1]
        gene_list_young.append(gene_code)

        session_idx = len(gene_list_young) - 1

        if gene_code == WT_GENE_CODE:
            wt_indices.append((session_idx, wt_counter))
            wt_counter += 1

    return wt_indices, gene_list_young


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_single_session(firing_rate_matrix, graph_clustering, EPSP_matrix, session_id, env_name,
                          target_scale=133, output_dir=OUTPUT_DIR):
    """
    Perform full analysis on a single session.

    Parameters:
    -----------
    firing_rate_matrix : np.ndarray
        Connectivity/firing rate matrix (neurons x neurons)
    graph_clustering : dict
        Graph-based clustering results
    session_id : int
        Session identifier
    env_name : str
        Environment name
    target_scale : int
        Target Markov scale for graph clustering
    output_dir : Path
        Output directory for results

    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    results = {
        'session_id': session_id,
        'environment': env_name,
        'n_neurons': firing_rate_matrix.shape[0]
    }

    # Get graph-based clustering at target scale
    selected_partitions = graph_clustering.get('selected_partitions', [])
    if len(selected_partitions) > 0:
        closest_scale = min(selected_partitions, key=lambda x: abs(x - target_scale))
    else:
        closest_scale = target_scale if target_scale in graph_clustering['community_id'] else 199

    graph_labels = graph_clustering['community_id'][closest_scale]
    graph_labels = np.array(graph_labels)

    # Determine number of clusters from graph method
    n_graph_clusters = len(np.unique(graph_labels))

    # Apply Rastermap clustering
    rastermap_labels, isort, embedding = rastermap_clustering_from_correlation(
        firing_rate_matrix,
        n_clusters=n_graph_clusters  # Match number of clusters
    )

    results['graph_labels'] = graph_labels
    results['rastermap_labels'] = rastermap_labels
    results['rastermap_isort'] = isort
    results['rastermap_embedding'] = embedding

    # Compare clusterings
    comparison_metrics = compare_clusterings(graph_labels, rastermap_labels)
    results['comparison_metrics'] = comparison_metrics

    # Create confusion matrix
    confusion, unique1, unique2 = create_confusion_matrix(graph_labels, rastermap_labels)
    results['confusion_matrix'] = confusion
    results['graph_clusters'] = unique1
    results['rastermap_clusters'] = unique2

    # Run graph analyses for both clustering methods
    print(f"  Running graph analysis with graph-based clusters...")
    graph_metrics_graph = build_graph(EPSP_matrix.copy(), graph_labels)
    results['graph_metrics_graph_based'] = graph_metrics_graph

    print(f"  Running graph analysis with rastermap clusters...")
    graph_metrics_rastermap = build_graph(EPSP_matrix.copy(), rastermap_labels)
    results['graph_metrics_rastermap'] = graph_metrics_rastermap

    return results


def run_full_analysis(envs=['fam1', 'nov', 'fam2'], output_dir=OUTPUT_DIR):
    """
    Run complete analysis across all environments.

    Parameters:
    -----------
    envs : list
        List of environment names to analyze
    output_dir : Path
        Output directory for results

    Returns:
    --------
    all_results : dict
        Dictionary containing results for all environments and sessions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get wild-type indices
    wt_indices, gene_list = get_wildtype_indices()
    print(f"Found {len(wt_indices)} wild-type sessions")

    all_results = {env: [] for env in envs}

    for env in envs:
        print(f"\n{'='*60}")
        print(f"Processing environment: {env}")
        print(f"{'='*60}")

        # Load firing rates
        firing_rates = load_firing_rates(env)
        EPSP_results = load_epsp_results(env)
        if firing_rates is None:
            print(f"Could not load firing rates for {env}")
            continue

        # Process each wild-type session
        for global_idx, wt_idx in wt_indices:
            print(f"\n  Session {global_idx} (WT index {wt_idx})")

            # Get firing rate matrix for this session
            if global_idx >= len(firing_rates):
                print(f"    Skipping: index {global_idx} out of range")
                continue

            firing_rate_matrix = firing_rates[global_idx]
            EPSP_matrix = EPSP_results[global_idx]
            # Load graph clustering
            graph_clustering = load_graph_clustering(env, wt_idx)
            if graph_clustering is None:
                print(f"    Skipping: no graph clustering found")
                continue

            # Run analysis
            try:
                session_results = analyze_single_session(
                    firing_rate_matrix,
                    graph_clustering,
                    EPSP_matrix,
                    session_id=global_idx,
                    env_name=env,
                    output_dir=output_dir
                )
                all_results[env].append(session_results)

                # Print summary
                metrics = session_results['comparison_metrics']
                print(f"    ARI: {metrics['adjusted_rand_index']:.3f}")
                print(f"    NMI: {metrics['normalized_mutual_info']:.3f}")
                print(f"    FMI: {metrics['fowlkes_mallows_index']:.3f}")

            except Exception as e:
                print(f"    Error processing session: {e}")
                continue

    return all_results


def plot_comparison_results(all_results, output_dir=OUTPUT_DIR):
    """Generate comparison plots for all results."""
    output_dir = Path(output_dir)

    # Collect metrics across all environments
    metrics_data = []

    for env, sessions in all_results.items():
        for session in sessions:
            metrics = session['comparison_metrics']
            metrics_data.append({
                'Environment': env,
                'Session': session['session_id'],
                'ARI': metrics['adjusted_rand_index'],
                'NMI': metrics['normalized_mutual_info'],
                'FMI': metrics['fowlkes_mallows_index'],
                'V-measure': metrics['v_measure']
            })

    if not metrics_data:
        print("No data to plot")
        return

    df = pd.DataFrame(metrics_data)

    # Plot 1: Comparison metrics by environment
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics_to_plot = ['ARI', 'NMI', 'FMI', 'V-measure']
    for ax, metric in zip(axes.flatten(), metrics_to_plot):
        sns.boxplot(data=df, x='Environment', y=metric, ax=ax)
        sns.swarmplot(data=df, x='Environment', y=metric, ax=ax, color='black', alpha=0.5)
        ax.set_title(f'{metric} by Environment')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_comparison_metrics.png', dpi=150)
    plt.close()

    # Plot 2: Graph metrics comparison (within-community)
    graph_metrics_data = []
    for env, sessions in all_results.items():
        for session in sessions:
            for method, prefix in [('graph_metrics_graph_based', 'Graph'),
                                   ('graph_metrics_rastermap', 'Rastermap')]:
                if method in session:
                    metrics = session[method]['within']
                    graph_metrics_data.append({
                        'Environment': env,
                        'Method': prefix,
                        'Clustering': metrics['clustering'],
                        'Reciprocal': metrics['reciprocal'],
                        'Divergent': metrics['divergent'],
                        'Convergent': metrics['convergent']
                    })

    if graph_metrics_data:
        df_graph = pd.DataFrame(graph_metrics_data)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics_to_plot = ['Clustering', 'Reciprocal', 'Divergent', 'Convergent']
        for ax, metric in zip(axes.flatten(), metrics_to_plot):
            sns.boxplot(data=df_graph, x='Environment', y=metric, hue='Method', ax=ax)
            ax.set_title(f'Within-Community {metric}')
            ax.legend(title='Clustering Method')

        plt.tight_layout()
        plt.savefig(output_dir / 'graph_metrics_comparison.png', dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


def save_results(all_results, output_dir=OUTPUT_DIR):
    """Save all results to pickle file."""
    output_dir = Path(output_dir)
    output_path = output_dir / 'rastermap_vs_graph_results.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Results saved to {output_path}")

#%%
def load_existing_results(output_dir=OUTPUT_DIR):
    """
    Load existing analysis results from pickle file.

    Parameters:
    -----------
    output_dir : Path
        Directory containing the results file

    Returns:
    --------
    all_results : dict or None
        Loaded results dictionary, or None if file not found
    """
    output_dir = Path(output_dir)
    results_path = output_dir / 'rastermap_vs_graph_results.pkl'

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return None

    with open(results_path, 'rb') as f:
        all_results = pickle.load(f)

    print(f"Loaded results from {results_path}")
    return all_results

#%%
def run_visualization_and_correlation_analysis(all_results=None, output_dir=OUTPUT_DIR,
                                                load_from_file=True):
    """
    Run visualization and order correlation analysis on existing results.

    This function can be used independently when results already exist,
    without re-running the full clustering analysis.

    Parameters:
    -----------
    all_results : dict, optional
        Pre-loaded results dictionary. If None and load_from_file=True,
        will attempt to load from disk.
    output_dir : Path
        Output directory for figures and results
    load_from_file : bool
        Whether to load results from file if all_results is None

    Returns:
    --------
    correlation_data : list
        List of correlation results for each session
    stats_results : dict
        Statistical analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load results if not provided
    if all_results is None and load_from_file:
        all_results = load_existing_results(output_dir)
        if all_results is None:
            print("Cannot proceed without results. Run full analysis first.")
            return None, None

    print("\n" + "=" * 70)
    print("FIRING RATE VISUALIZATION AND ORDER CORRELATION ANALYSIS")
    print("=" * 70)

    # Load firing rate data for visualizations
    print("\nLoading firing rate data for visualizations...")
    firing_rates_dict = {}
    for env in ['fam1', 'nov', 'fam2']:
        firing_rates = load_firing_rates(env)
        if firing_rates is not None:
            firing_rates_dict[env] = firing_rates
            print(f"  {env}: Loaded {len(firing_rates)} sessions")

    # Generate firing rate visualizations
    if firing_rates_dict:
        print("\nGenerating firing rate visualizations (dual ordering)...")
        figure_paths = visualize_all_sessions_firing_rates(
            all_results, firing_rates_dict, output_dir
        )
        print(f"Generated {len(figure_paths)} visualization figures")
    else:
        print("Warning: No firing rate data available for visualization")

    # Compute order correlations
    print("\nComputing order correlations...")
    correlation_data = compute_order_correlations_all_sessions(all_results)
    print(f"Computed correlations for {len(correlation_data)} sessions")

    # Statistical analysis
    if correlation_data:
        print("\nPerforming statistical analysis...")
        stats_results, corr_df = statistical_analysis_order_correlation(
            correlation_data, output_dir
        )

        # Print summary
        print_statistical_summary(stats_results)

        # Generate plots
        print("\nGenerating correlation analysis plots...")
        plot_order_correlation_results(correlation_data, stats_results, output_dir)

        return correlation_data, stats_results
    else:
        print("No correlation data to analyze")
        return None, None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
#%%
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Rastermap vs Graph-Based Clustering Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis (clustering + visualization + correlation)
  python rastermap_vs_graph_clustering.py

  # Only run visualization and correlation analysis on existing results
  python rastermap_vs_graph_clustering.py --visualization-only

  # Specify custom output directory
  python rastermap_vs_graph_clustering.py --output-dir /path/to/output
        """
    )
    parser.add_argument('--visualization-only', action='store_true',default=True,
                        help='Only run visualization and correlation analysis (requires existing results)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for results')
    parser.add_argument('--skip-full-analysis', action='store_true',
                        help='Skip full clustering analysis, only run visualization')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(exist_ok=True)

    print("Rastermap vs Graph-Based Clustering Analysis")
    print("=" * 70)

    # Check if data paths exist
    print("\nChecking data paths...")
    for env, paths in DATA_PATHS.items():
        epsp_path = paths['epsp']
        if epsp_path.exists():
            print(f"  {env}: Found EPSP data")
        else:
            print(f"  {env}: EPSP data NOT found at {epsp_path}")

    if TRIGGER_FILE.exists():
        print(f"  Trigger file: Found")
    else:
        print(f"  Trigger file: NOT found at {TRIGGER_FILE}")

    if args.visualization_only or args.skip_full_analysis:
        # Only run visualization and correlation analysis
        print("\n[MODE: Visualization and Correlation Analysis Only]")
        correlation_data, stats_results = run_visualization_and_correlation_analysis(
            output_dir=OUTPUT_DIR,
            load_from_file=True
        )
    else:
        # Run full analysis
        print("\nRunning full clustering analysis...")
        all_results = run_full_analysis()

        # Generate comparison plots
        print("\nGenerating clustering comparison plots...")
        plot_comparison_results(all_results)

        # Save results
        save_results(all_results)

        # Run visualization and correlation analysis
        print("\nRunning visualization and order correlation analysis...")
        correlation_data, stats_results = run_visualization_and_correlation_analysis(
            all_results=all_results,
            output_dir=OUTPUT_DIR,
            load_from_file=False
        )

        # Print clustering summary
        print("\n" + "=" * 70)
        print("CLUSTERING COMPARISON SUMMARY")
        print("=" * 70)

        for env, sessions in all_results.items():
            if sessions:
                ari_values = [s['comparison_metrics']['adjusted_rand_index'] for s in sessions]
                nmi_values = [s['comparison_metrics']['normalized_mutual_info'] for s in sessions]
                print(f"\n{env.upper()}:")
                print(f"  Sessions analyzed: {len(sessions)}")
                print(f"  Mean ARI: {np.mean(ari_values):.3f} (+/- {np.std(ari_values):.3f})")
                print(f"  Mean NMI: {np.mean(nmi_values):.3f} (+/- {np.std(nmi_values):.3f})")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

