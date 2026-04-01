from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error
from qiskit_aer.aererror import AerError
from qiskit_ibm_runtime.fake_provider import FakeBrisbane


def _build_noise_model_with_extra_damping(
    fake_backend,
    extra_amp_damp_1q=0.0,
    extra_amp_damp_2q=0.0,
):
    noise_model = NoiseModel.from_backend(fake_backend)
    basis = set(noise_model.basis_gates)
    target = fake_backend.target

    one_q_gates = [g for g in ["id", "rz", "sx", "x", "rx", "ry"] if g in basis and g in target]
    two_q_gates = [g for g in ["cx", "ecr"] if g in basis and g in target]

    if extra_amp_damp_1q > 0.0:
        err1 = amplitude_damping_error(extra_amp_damp_1q)
        for g in one_q_gates:
            for qargs in target[g]:
                if len(qargs) == 1:
                    noise_model.add_quantum_error(err1, g, list(qargs))

    if extra_amp_damp_2q > 0.0:
        err2 = amplitude_damping_error(extra_amp_damp_2q).tensor(
            amplitude_damping_error(extra_amp_damp_2q)
        )
        for g in two_q_gates:
            for qargs in target[g]:
                if len(qargs) == 2:
                    noise_model.add_quantum_error(err2, g, list(qargs))

    return noise_model


def _gpu_is_available():
    try:
        tmp = AerSimulator()
        devices = tmp.available_devices()
        return "GPU" in devices
    except Exception:
        return False


def get_simulator(
    n_qubits,
    noisy=True,
    extra_amp_damp_1q=0.0,
    extra_amp_damp_2q=0.0,
    use_gpu=False,
):
    fake_backend = FakeBrisbane()

    if n_qubits > fake_backend.num_qubits:
        raise ValueError(
            f"n_qubits={n_qubits} exceeds selected backend "
            f"({fake_backend.name}, {fake_backend.num_qubits} qubits)."
        )

    transpile_backend = fake_backend

    really_use_gpu = use_gpu and _gpu_is_available()

    sim_kwargs = {}
    if really_use_gpu:
        sim_kwargs["device"] = "GPU"

    if not noisy:
        if really_use_gpu:
            sim_kwargs["method"] = "statevector"
        try:
            run_backend = AerSimulator(**sim_kwargs)
        except (AerError, RuntimeError):
            run_backend = AerSimulator()
    else:
        if really_use_gpu:
            sim_kwargs["method"] = "density_matrix"

        if extra_amp_damp_1q == 0.0 and extra_amp_damp_2q == 0.0:
            try:
                run_backend = AerSimulator.from_backend(fake_backend, **sim_kwargs)
            except (AerError, RuntimeError):
                run_backend = AerSimulator.from_backend(fake_backend)
        else:
            noise_model = _build_noise_model_with_extra_damping(
                fake_backend,
                extra_amp_damp_1q=extra_amp_damp_1q,
                extra_amp_damp_2q=extra_amp_damp_2q,
            )
            try:
                run_backend = AerSimulator(
                    noise_model=noise_model,
                    basis_gates=noise_model.basis_gates,
                    **sim_kwargs,
                )
            except (AerError, RuntimeError):
                run_backend = AerSimulator(
                    noise_model=noise_model,
                    basis_gates=noise_model.basis_gates,
                )

    return run_backend, transpile_backend
