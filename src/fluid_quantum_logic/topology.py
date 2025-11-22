"""
Domain-Agnostic Binding Topology.

Defines abstract neighbor relationships for quantum qubits across different
modalities (Vision, Audio, Language). Used to generate connectivity graphs
for binding primitives.
"""

import numpy as np
from typing import List, Tuple, Dict

class BindingTopology:
    """
    Base class for defining qubit connectivity.

    Attributes:
        adjacency (np.ndarray): N×N binary matrix where [i,j]=1 implies connection.
        n_qubits (int): Total number of qubits in the topology.
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        self.adjacency = adjacency_matrix
        self.n_qubits = adjacency_matrix.shape[0]

    def get_binding_pairs(self) -> List[Tuple[int, int]]:
        """Returns list of (control, target) pairs based on adjacency."""
        pairs = []
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                if self.adjacency[i, j] == 1:
                    pairs.append((i, j))
        return pairs

    def get_binding_groups(self, group_fn) -> Dict[str, List[int]]:
        """Partitions qubits into named groups based on a mapping function."""
        groups = {}
        for i in range(self.n_qubits):
            group = group_fn(i)
            if group not in groups:
                groups[group] = []
            groups[group].append(i)
        return groups


class VisionTopology(BindingTopology):
    """
    2D spatial grid topology for visual inputs.
    Connects pixels to their horizontal and vertical neighbors.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        n_qubits = rows * cols
        adjacency = np.zeros((n_qubits, n_qubits))

        # Horizontal adjacency
        for r in range(rows):
            for c in range(cols - 1):
                i = r * cols + c
                j = r * cols + (c + 1)
                adjacency[i, j] = 1
                adjacency[j, i] = 1

        # Vertical adjacency
        for c in range(cols):
            for r in range(rows - 1):
                i = r * cols + c
                j = (r + 1) * cols + c
                adjacency[i, j] = 1
                adjacency[j, i] = 1

        super().__init__(adjacency)

    def get_row_indices(self, row: int) -> List[int]:
        """Returns qubit indices for a specific row."""
        return list(range(row * self.cols, (row + 1) * self.cols))

    def get_col_indices(self, col: int) -> List[int]:
        """Returns qubit indices for a specific column."""
        return list(range(col, self.n_qubits, self.cols))


class AudioTopology(BindingTopology):
    """
    1D temporal sequence topology for audio inputs.
    Connects time steps t to t-lag neighbors.
    """

    def __init__(self, n_timesteps: int, window_size: int = 2):
        """
        Args:
            n_timesteps: Number of time steps in sequence.
            window_size: Temporal window radius (1 = adjacent only, 2 = ±2 steps).
        """
        self.n_timesteps = n_timesteps
        self.window_size = window_size
        adjacency = np.zeros((n_timesteps, n_timesteps))

        for t in range(n_timesteps):
            for offset in range(1, window_size + 1):
                if t - offset >= 0:
                    adjacency[t, t - offset] = 1

        super().__init__(adjacency)

    def get_temporal_pairs(self, lag: int = 1) -> List[Tuple[int, int]]:
        """
        Get pairs of (t, t-lag) for change detection.

        Args:
            lag: Time offset (1 = adjacent, 2 = skip one, etc.)

        Returns:
            List of (current_timestep, past_timestep) pairs.
        """
        pairs = []
        for t in range(lag, self.n_timesteps):
            pairs.append((t, t - lag))
        return pairs


class LanguageTopology(BindingTopology):
    """
    Syntactic dependency topology for language inputs.
    Edges represent grammatical relationships (subject-verb, verb-object).
    """

    def __init__(self, n_words: int, dependencies: List[Tuple[int, int, str]]):
        """
        Args:
            n_words: Number of words/tokens in sentence.
            dependencies: List of (head_idx, dependent_idx, relation_type).
        """
        self.n_words = n_words
        self.dependencies = dependencies
        self.n_qubits = n_words

        adjacency = np.zeros((n_words, n_words))
        for head, dep, _ in dependencies:
            adjacency[head, dep] = 1

        super().__init__(adjacency)

    def get_dependency_pairs(self, relation_type: str = None) -> List[Tuple[int, int]]:
        """
        Get pairs filtered by dependency relation type.

        Args:
            relation_type: Filter by specific relation (e.g., "subject", "object").
                          If None, returns all pairs.

        Returns:
            List of (head_idx, dependent_idx) tuples.
        """
        if relation_type is None:
            return [(h, d) for h, d, r in self.dependencies]
        else:
            return [(h, d) for h, d, r in self.dependencies if r == relation_type]


def create_vision_topology_2x4() -> VisionTopology:
    """
    Create standard 2×4 grid for vision experiments.

    Grid layout:
        [0] [1] [2] [3]
        [4] [5] [6] [7]

    Returns:
        VisionTopology instance with 8 qubits.
    """
    return VisionTopology(rows=2, cols=4)


def create_audio_topology_8steps() -> AudioTopology:
    """
    Create standard 8-timestep sequence for audio experiments.

    Timeline:
        [0] → [1] → [2] → [3] → [4] → [5] → [6] → [7]

    Returns:
        AudioTopology instance with 8 qubits.
    """
    return AudioTopology(n_timesteps=8, window_size=1)
