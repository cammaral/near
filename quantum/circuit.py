from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector


def build_brickwall(n_qubits, depth):
    n_params = 2 * depth * n_qubits
    params = ParameterVector("theta", length=n_params)

    qc = QuantumCircuit(n_qubits)
    p = 0
    for d in range(depth):
        for q in range(n_qubits):
            qc.ry(params[p], q)
            p += 1
            qc.rz(params[p], q)
            p += 1

        start = d % 2
        for q in range(start, n_qubits - 1, 2):
            qc.cx(q, q + 1)

    return qc, params


def apply_py(qc, bitstring_q0_first):
    """
    Apply P_y = ⊗_i X_i^{y_i}.
    bitstring_q0_first uses your convention: char i corresponds to qubit i.
    """
    for q, b in enumerate(bitstring_q0_first):
        if b == "1":
            qc.x(q)


def build_circuit(
    n_qubits,
    ansatz,
    transpile_backend,
    seed=42,
    gauge_bitstring_q0_first=None,
    gauge_mode="none",   # "none", "output_only", "sandwich"
):
    measured_template = QuantumCircuit(n_qubits, n_qubits)

    if gauge_bitstring_q0_first is not None and gauge_mode == "sandwich":
        apply_py(measured_template, gauge_bitstring_q0_first)

    measured_template.compose(ansatz, inplace=True)

    if gauge_bitstring_q0_first is not None and gauge_mode in ("output_only", "sandwich"):
        apply_py(measured_template, gauge_bitstring_q0_first)

    measured_template.measure(range(n_qubits), range(n_qubits))

    return transpile(
        measured_template,
        transpile_backend,
        optimization_level=1,
        seed_transpiler=seed,
    )