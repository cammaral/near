import numpy as np
from quantum.hamiltonian import bitstring_to_spins, hamiltonian_value_for_bitstring

def generate(n_qubits, edge_prob=0.3, rng=None, weight_mode="unit"):
    """
    Gerenates a random graph
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2")

    nodes = list(range(n_qubits))
    rng.shuffle(nodes)

    edges = set()

    for k in range(1, n_qubits):
        a = nodes[k]
        b = nodes[rng.integers(0, k)]
        i, j = sorted((a, b))
        edges.add((i, j))

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if (i, j) not in edges and rng.random() < edge_prob:
                edges.add((i, j))

    weighted_edges = []
    for i, j in sorted(edges):
        if weight_mode == "unit":
            w = 1.0
        elif weight_mode == "random":
            w = float(rng.uniform(0.5, 2.0))
        else:
            raise ValueError("weight_mode must be 'unit' or 'random'")
        weighted_edges.append((i, j, w))

    return weighted_edges

def integer_to_bitstring_q0_first(x, n_qubits):
    return "".join(str((x >> i) & 1) for i in range(n_qubits))

def maxcut_value(bitstring_q0_first, edges):
    x = np.array([int(b) for b in bitstring_q0_first], dtype=int)
    val = 0.0
    for i, j, w in edges:
        if x[i] != x[j]:
            val += w
    return float(val)

def solve_bruteforce(n_qubits, constant_term, zz_terms):
    best_energy = np.inf
    best_bitstrings = []

    for x in range(2 ** n_qubits):
        b = integer_to_bitstring_q0_first(x, n_qubits)
        e = hamiltonian_value_for_bitstring(b, constant_term, zz_terms)

        if e < best_energy - 1e-12:
            best_energy = e
            best_bitstrings = [b]
        elif abs(e - best_energy) < 1e-12:
            best_bitstrings.append(b)
    print(f"Best Energy: {float(best_energy)}, Best Bitstring: {best_bitstrings}\n")
    return float(best_energy), best_bitstrings

def adjacency_matrix_from_edges(n_qubits, edges):
    A = np.zeros((n_qubits, n_qubits), dtype=float)
    for i, j, w in edges:
        A[i, j] = w
        A[j, i] = w
    return A