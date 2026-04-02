from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error
from qiskit_aer.aererror import AerError
from qiskit_ibm_runtime.fake_provider import FakeBrisbane


def _compose_extra_damping(err, extra_err):
    """
    Compõe o erro já existente com o damping extra.
    A ideia é aplicar primeiro o erro original do backend e depois o damping extra.
    """
    if extra_err is None:
        return err
    return err.compose(extra_err)


def _copy_readout_errors(src_model: NoiseModel, dst_model: NoiseModel):
    """
    Copia readout errors do noise model original para o novo.
    Usa atributos internos do Aer porque a API pública não oferece
    um método de 'replace' para erros já existentes.
    """
    # Erro de leitura global, se existir
    default_ro = getattr(src_model, "_default_readout_error", None)
    if default_ro is not None:
        dst_model.add_all_qubit_readout_error(default_ro)

    # Erros de leitura locais por qubit
    local_ro = getattr(src_model, "_local_readout_errors", {})
    for qubits, ro_err in local_ro.items():
        dst_model.add_readout_error(ro_err, list(qubits))


def _copy_and_update_quantum_errors(
    src_model: NoiseModel,
    dst_model: NoiseModel,
    target,
    extra_amp_damp_1q=0.0,
    extra_amp_damp_2q=0.0,
):
    """
    Copia os quantum errors do modelo original para o novo, mas
    para as portas desejadas compõe com um damping extra em vez de
    chamar add_quantum_error sobre um erro já existente.
    """
    err1 = amplitude_damping_error(extra_amp_damp_1q) if extra_amp_damp_1q > 0.0 else None
    err2 = (
        amplitude_damping_error(extra_amp_damp_2q).tensor(
            amplitude_damping_error(extra_amp_damp_2q)
        )
        if extra_amp_damp_2q > 0.0
        else None
    )

    # Descobre quais gates existem no backend/target
    basis = set(src_model.basis_gates)
    one_q_gates = [g for g in ["id", "rz", "sx", "x", "rx", "ry"] if g in basis and g in target]
    two_q_gates = [g for g in ["cx", "ecr"] if g in basis and g in target]

    # Copiar erros globais, se existirem
    default_qe = getattr(src_model, "_default_quantum_errors", {})
    for instr, qerr in default_qe.items():
        qerr_new = qerr

        # Se for um gate global de 1 ou 2 qubits, compõe também
        if instr in one_q_gates and err1 is not None and qerr.num_qubits == 1:
            qerr_new = _compose_extra_damping(qerr, err1)
        elif instr in two_q_gates and err2 is not None and qerr.num_qubits == 2:
            qerr_new = _compose_extra_damping(qerr, err2)

        dst_model.add_all_qubit_quantum_error(qerr_new, instr)

    # Copiar erros locais por instrução/qubits
    local_qe = getattr(src_model, "_local_quantum_errors", {})
    for instr, mapping in local_qe.items():
        for qubits, qerr in mapping.items():
            qerr_new = qerr
            nq = len(qubits)

            if instr in one_q_gates and err1 is not None and nq == 1:
                qerr_new = _compose_extra_damping(qerr, err1)
            elif instr in two_q_gates and err2 is not None and nq == 2:
                qerr_new = _compose_extra_damping(qerr, err2)

            dst_model.add_quantum_error(qerr_new, instr, list(qubits))


def _build_noise_model_with_extra_damping(
    fake_backend,
    extra_amp_damp_1q=0.0,
    extra_amp_damp_2q=0.0,
):
    """
    Cria um noise model do backend e, sem duplicar entradas,
    compõe damping extra nos erros já existentes.
    """
    base_noise = NoiseModel.from_backend(fake_backend)

    # Se não há damping extra, devolve direto
    if extra_amp_damp_1q == 0.0 and extra_amp_damp_2q == 0.0:
        return base_noise

    # Novo noise model vazio com a mesma basis
    new_noise = NoiseModel(basis_gates=list(base_noise.basis_gates))

    # Copia basis gates extras explicitamente, se necessário
    new_noise.add_basis_gates(list(base_noise.basis_gates))

    target = fake_backend.target

    # Copia e atualiza quantum errors
    _copy_and_update_quantum_errors(
        src_model=base_noise,
        dst_model=new_noise,
        target=target,
        extra_amp_damp_1q=extra_amp_damp_1q,
        extra_amp_damp_2q=extra_amp_damp_2q,
    )

    # Copia readout errors
    _copy_readout_errors(base_noise, new_noise)

    return new_noise


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
