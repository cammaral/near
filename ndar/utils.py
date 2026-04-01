from quantum.hamiltonian import hamiltonian_value_for_bitstring
import numpy as np

def qiskit_to_q0_first(bitstring):
    return bitstring.replace(" ", "")[::-1]

def bind_and_run_circuit(theta, transpiled_measured_template, ansatz_params, simulator, shots, seed_simulator):
    bind_map = {p: float(v) for p, v in zip(ansatz_params, theta)}
    bound_circuit = transpiled_measured_template.assign_parameters(bind_map, inplace=False)

    job = simulator.run(
        bound_circuit,
        shots=shots,
        memory=True,
        seed_simulator=seed_simulator,
    )
    return job.result()


def evaluate_theta_expected_energy(
    theta,
    transpiled_measured_template,
    ansatz_params,
    simulator,
    shots,
    seed_simulator,
    H_constant_term,
    H_zz_terms,
):
    result = bind_and_run_circuit(
        theta=theta,
        transpiled_measured_template=transpiled_measured_template,
        ansatz_params=ansatz_params,
        simulator=simulator,
        shots=shots,
        seed_simulator=seed_simulator,
    )

    energies = []
    for raw_bs in result.get_memory():
        bs_q0 = qiskit_to_q0_first(raw_bs)
        energies.append(hamiltonian_value_for_bitstring(bs_q0, H_constant_term, H_zz_terms))

    return float(np.mean(energies))