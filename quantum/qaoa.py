from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def build_qaoa_p1_from_terms(n_qubits, zz_terms, h_terms=None):
    gamma = Parameter("gamma")
    beta = Parameter("beta")

    qc = QuantumCircuit(n_qubits)

    # |+>^n
    for q in range(n_qubits):
        qc.h(q)

    # phase separator exp(-i gamma H)
    if h_terms is not None:
        for i, h in h_terms:
            qc.rz(2.0 * h * gamma, i)

    for i, j, J in zz_terms:
        qc.rzz(2.0 * J * gamma, i, j)

    # X mixer exp(-i beta sum X_i)
    for q in range(n_qubits):
        qc.rx(2.0 * beta, q)

    return qc, [gamma, beta]