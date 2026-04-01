from qiskit.quantum_info import SparsePauliOp

import numpy as np

def build_hamiltonian_from_terms(n_qubits, constant_term, zz_terms):
    paulis = []
    coeffs = []

    for i, j, coeff in zz_terms:
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"
        label[n_qubits - 1 - j] = "Z"
        paulis.append("".join(label))
        coeffs.append(coeff)

    paulis.append("I" * n_qubits)
    coeffs.append(constant_term)

    return SparsePauliOp(paulis, coeffs=coeffs)


def build_maxcut_hamiltonian(n_qubits, edges):
    constant_term = 0.0
    zz_terms = []

    for i, j, w in edges:
        constant_term += -0.5 * w
        coeff = 0.5 * w
        zz_terms.append((i, j, coeff))

    H_sparse = build_hamiltonian_from_terms(n_qubits, constant_term, zz_terms)
    return H_sparse, constant_term, zz_terms

def update_hamiltonian(n_qubits, constant_term, zz_terms, bitstring_q0_first):
    signs = np.array([-1.0 if bit == "1" else 1.0 for bit in bitstring_q0_first], dtype=float)

    new_zz_terms = []
    for i, j, coeff in zz_terms:
        new_coeff = coeff * signs[i] * signs[j]
        new_zz_terms.append((i, j, float(new_coeff)))

    H_next = build_hamiltonian_from_terms(n_qubits, constant_term, new_zz_terms)
    return H_next, constant_term, new_zz_terms

def serialize_hamiltonian(H_sparse, constant_term, zz_terms):
    return {
        "constant_term": float(constant_term),
        "zz_terms": [
            {"i": int(i), "j": int(j), "coeff": float(coeff)}
            for i, j, coeff in zz_terms
        ],
        "pauli_operator": [
            {
                "pauli": str(p),
                "coeff_real": float(np.real(c)),
                "coeff_imag": float(np.imag(c)),
            }
            for p, c in zip(H_sparse.paulis, H_sparse.coeffs)
        ],
    }


#===============================
# Abaixo apenas para conversão
#===============================

def bitstring_to_spins(bitstring_q0_first):
    return np.array([1 if b == "0" else -1 for b in bitstring_q0_first], dtype=float)

def hamiltonian_value_for_bitstring(bitstring_q0_first, constant_term, zz_terms):
    s = bitstring_to_spins(bitstring_q0_first)

    energy = constant_term
    for i, j, coeff in zz_terms:
        energy += coeff * s[i] * s[j]

    return float(energy)