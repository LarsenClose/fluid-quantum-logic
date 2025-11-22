"""
Domain-Agnostic Binding Primitives.

Implements geometric quantum operations (CNOT + Rotation) applied over
the topological structures defined in `src.topology`.
"""

import pennylane as qml
from typing import List
from .topology import BindingTopology

def generic_binding(
    params,
    topology: BindingTopology,
    source_qubits: List[int],
    target_qubit: int,
    groups: List[List[int]] = None
):
    """
    Apply binding operations across a generic topology.

    If groups are provided, applies unique parameters per group.

    Args:
        params: Rotation parameters (one per group).
        topology: BindingTopology instance defining connectivity.
        source_qubits: List of input qubit indices.
        target_qubit: Output qubit to bind to.
        groups: Optional grouping of source qubits.
    """
    if groups is None:
        for qubit in source_qubits:
            qml.CNOT(wires=[qubit, target_qubit])
            qml.RY(params[0], wires=target_qubit)
    else:
        for group_idx, group_qubits in enumerate(groups):
            for qubit in group_qubits:
                if qubit in source_qubits:
                    qml.CNOT(wires=[qubit, target_qubit])
                    qml.RY(params[group_idx], wires=target_qubit)


def spatial_row_binding(params, source_qubits, target_qubit, rows: List[List[int]]):
    """
    Detects horizontal coherence (Vision).

    Binds pixels in the same row to the target feature detector.

    Args:
        params: Rotation parameters (one per row).
        source_qubits: Input qubit indices.
        target_qubit: Output feature detector qubit.
        rows: List of qubit index lists, one per row.
    """
    for row_idx, row_qubits in enumerate(rows):
        for qubit in row_qubits:
            qml.CNOT(wires=[qubit, target_qubit])
            qml.RY(params[row_idx], wires=target_qubit)


def spatial_col_binding(params, source_qubits, target_qubit, cols: List[List[int]]):
    """
    Detects vertical coherence (Vision).

    Binds pixels in the same column to the target feature detector.

    Args:
        params: Rotation parameters (one per column group).
        source_qubits: Input qubit indices.
        target_qubit: Output feature detector qubit.
        cols: List of qubit index lists, grouped by column.
    """
    for col_group_idx, col_qubits in enumerate(cols):
        for qubit in col_qubits:
            qml.CNOT(wires=[qubit, target_qubit])
            qml.RY(params[col_group_idx], wires=target_qubit)


def temporal_binding(params, source_qubits, target_qubit, lag: int = 1):
    """
    Detects temporal changes (Audio).

    Implements XOR change detection between timesteps t and t-lag.

    Args:
        params: Rotation parameters [param_recent, param_past].
        source_qubits: Timestep qubit indices (ordered chronologically).
        target_qubit: Output change detector qubit.
        lag: Time offset (1 = adjacent steps, 2 = skip one, etc.).
    """
    # Recent timesteps
    for qubit in source_qubits[-lag:]:
        qml.CNOT(wires=[qubit, target_qubit])
        qml.RY(params[0], wires=target_qubit)

    # Past timesteps
    for qubit in source_qubits[:-lag]:
        qml.CNOT(wires=[qubit, target_qubit])
        qml.RY(params[1], wires=target_qubit)


def syntactic_binding(params, source_qubits, target_qubit, head_dep_pairs: List[tuple]):
    """
    Detects grammatical relationships (Language).

    Binds Head and Dependent tokens to the syntactic detector.

    Args:
        params: Rotation parameters (one per relation type).
        source_qubits: Word/token qubit indices.
        target_qubit: Output syntactic detector qubit.
        head_dep_pairs: List of (head_idx, dependent_idx, relation_type_idx).
    """
    for head, dep, rel_type in head_dep_pairs:
        qml.CNOT(wires=[head, target_qubit])
        qml.CNOT(wires=[dep, target_qubit])
        qml.RY(params[rel_type], wires=target_qubit)


# --- Specialized Implementations for Grid Experiments ---

def horizontal_binding(params, wires_l1, wire_l2):
    """
    Specialized wrapper for 4×2 grid horizontal binding.

    Detects horizontal lines (pixels in same row). Used in bistability
    and hierarchical vision experiments.

    Args:
        params: [param_row0, param_row1] - One rotation per row.
        wires_l1: Input qubit indices (8 qubits for 4×2 grid).
        wire_l2: Output detector qubit.
    """
    rows = [
        list(range(4)),      # Row 0: indices [0,1,2,3]
        list(range(4, 8))    # Row 1: indices [4,5,6,7]
    ]
    spatial_row_binding(params, wires_l1, wire_l2, rows)


def vertical_binding(params, wires_l1, wire_l2):
    """
    Specialized wrapper for 4×2 grid vertical binding.

    Detects vertical lines (pixels in same column). Used in bistability
    and hierarchical vision experiments.

    Args:
        params: [param_left, param_remaining] - Two rotation groups.
        wires_l1: Input qubit indices (8 qubits for 4×2 grid).
        wire_l2: Output detector qubit.
    """
    cols = [
        [0, 4],              # Left column
        [1, 5, 2, 6, 3, 7]   # Remaining columns
    ]
    spatial_col_binding(params, wires_l1, wire_l2, cols)


def dense_binding(params, wires_l1, wire_l2):
    """
    Specialized wrapper for 4×2 grid quadrant density.

    Detects overall fill/brightness by quadrant. Used in bistability
    and hierarchical vision experiments.

    Args:
        params: [param_left, param_right] - One rotation per quadrant.
        wires_l1: Input qubit indices (8 qubits for 4×2 grid).
        wire_l2: Output density detector qubit.
    """
    groups = [
        [0, 1, 4, 5],  # Left quadrant
        [2, 3, 6, 7]   # Right quadrant
    ]
    generic_binding(params, None, wires_l1, wire_l2, groups)
