import json
import math
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer.noise import NoiseModel, amplitude_damping_error


# ======================================================================================
# CONFIG
# ======================================================================================
CONFIG = {
    # Which panels to run: "left", "right", or "both"
    "run_mode": "both",

    # This script targets the simulated QAOA figure logic (supplementary Fig. 4 style)
    "n_qubits": 16,
    "seed": 1924,
    "n_instances": 10,
    "n_gauges": 20,

    # Left panel: correlation attractor AR vs optimized AR
    "left_shots": 256,
    "left_n_epochs": 100,
    "left_top_fracs": [1.0, 50 / 256, 1 / 256],

    # Right panel: NDAR vs plain QAOA vs random
    "right_K": 5,
    "right_shots": 100,
    "right_n_epochs_per_iter": 20,

    # Optimizer (same spirit as your current project)
    "theta_init": [0.1, 0.1],
    "adam_lr": 0.10,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "spsa_c": 0.15,
    "spsa_gamma": 0.101,

    # Noise
    # noisy=False -> ideal simulator
    # noisy=True with both extras = 0.0 -> pure Qiskit/FakeBrisbane default
    # noisy=True with extras > 0 -> FakeBrisbane + extra amplitude damping
    "run_noiseless": True,
    "run_noisy": True,
    "extra_amp_damp_1q": 0.005,
    "extra_amp_damp_2q": 0.02,

    # Output
    "out_dir": "fig4_qaoa_paper_like",
}


# ======================================================================================
# BASIC UTILS
# ======================================================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def qiskit_to_q0_first(bitstring):
    return bitstring.replace(" ", "")[::-1]


def xor_bitstrings(a, b):
    return "".join("1" if xa != xb else "0" for xa, xb in zip(a, b))


def random_bitstring(n, rng):
    return "".join(rng.choice(["0", "1"], size=n))


def ratio_label(frac):
    return f"optimized_ratio_topfrac_{str(frac).replace('.', 'p')}"


def summarize_top_fracs(ratios, top_fracs):
    r = np.asarray(ratios, dtype=float)
    r = np.sort(r)[::-1]

    out = {}
    n = len(r)
    for frac in top_fracs:
        k = max(1, int(np.ceil(frac * n)))
        out[ratio_label(frac)] = float(np.mean(r[:k]))

    out["optimized_ratio_best_sample"] = float(r[0])
    out["optimized_ratio_mean_all"] = float(np.mean(r))
    return out


def rankdata_average(x):
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    return pearson_corr(rankdata_average(x), rankdata_average(y))


# ======================================================================================
# SK PROBLEM
# ======================================================================================
def generate_sk_terms(n_qubits, rng):
    zz_terms = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            Jij = float(rng.choice([-1.0, 1.0]))
            zz_terms.append((i, j, Jij))
    return 0.0, zz_terms


def update_zz_terms_by_gauge(zz_terms, gauge_bitstring_q0_first):
    signs = np.array([-1.0 if b == "1" else 1.0 for b in gauge_bitstring_q0_first], dtype=float)
    out = []
    for i, j, coeff in zz_terms:
        out.append((i, j, float(coeff * signs[i] * signs[j])))
    return out


def bitstring_to_spins(bitstring_q0_first):
    return np.array([1.0 if b == "0" else -1.0 for b in bitstring_q0_first], dtype=float)


def energy_of_bitstring(bitstring_q0_first, constant_term, zz_terms):
    s = bitstring_to_spins(bitstring_q0_first)
    e = float(constant_term)
    for i, j, coeff in zz_terms:
        e += float(coeff) * s[i] * s[j]
    return float(e)


def brute_force_ground_energy(n_qubits, constant_term, zz_terms):
    best_energy = np.inf
    best_bitstring = None

    for state in range(1 << n_qubits):
        bs = format(state, f"0{n_qubits}b")[::-1]  # q0-first
        e = energy_of_bitstring(bs, constant_term, zz_terms)
        if e < best_energy:
            best_energy = e
            best_bitstring = bs

    return float(best_energy), best_bitstring


# ======================================================================================
# BACKEND / NOISE
# ======================================================================================
def _build_noise_model_with_extra_damping(fake_backend, extra_amp_damp_1q=0.0, extra_amp_damp_2q=0.0):
    noise_model = NoiseModel.from_backend(fake_backend)
    basis = set(noise_model.basis_gates)

    one_q_gates = [g for g in ["id", "rz", "sx", "x", "rx", "ry", "h"] if g in basis]
    two_q_gates = [g for g in ["cx", "ecr"] if g in basis]

    if extra_amp_damp_1q > 0.0:
        err1 = amplitude_damping_error(extra_amp_damp_1q)
        for g in one_q_gates:
            noise_model.add_all_qubit_quantum_error(err1, g)

    if extra_amp_damp_2q > 0.0:
        err2 = amplitude_damping_error(extra_amp_damp_2q).tensor(
            amplitude_damping_error(extra_amp_damp_2q)
        )
        for g in two_q_gates:
            noise_model.add_all_qubit_quantum_error(err2, g)

    return noise_model


def get_simulator_compat(n_qubits, noisy=True, extra_amp_damp_1q=0.0, extra_amp_damp_2q=0.0):
    """
    Uses your project backend if possible. Falls back to a local implementation otherwise.
    """
    try:
        from quantum.backend import get_simulator as project_get_simulator

        sig = inspect.signature(project_get_simulator)
        kwargs = {"n_qubits": n_qubits, "noisy": noisy}
        if "extra_amp_damp_1q" in sig.parameters:
            kwargs["extra_amp_damp_1q"] = extra_amp_damp_1q
        if "extra_amp_damp_2q" in sig.parameters:
            kwargs["extra_amp_damp_2q"] = extra_amp_damp_2q

        out = project_get_simulator(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out
    except Exception:
        pass

    fake_backend = FakeBrisbane()
    if n_qubits > fake_backend.num_qubits:
        raise ValueError(
            f"n_qubits={n_qubits} exceeds selected backend "
            f"({fake_backend.name}, {fake_backend.num_qubits} qubits)."
        )

    transpile_backend = fake_backend
    if not noisy:
        run_backend = AerSimulator()
    else:
        if extra_amp_damp_1q == 0.0 and extra_amp_damp_2q == 0.0:
            run_backend = AerSimulator.from_backend(fake_backend)
        else:
            noise_model = _build_noise_model_with_extra_damping(
                fake_backend,
                extra_amp_damp_1q=extra_amp_damp_1q,
                extra_amp_damp_2q=extra_amp_damp_2q,
            )
            run_backend = AerSimulator(
                noise_model=noise_model,
                basis_gates=noise_model.basis_gates,
            )

    return run_backend, transpile_backend


# ======================================================================================
# QAOA
# ======================================================================================
def build_qaoa_p1_from_terms(n_qubits, zz_terms):
    gamma = Parameter("gamma")
    beta = Parameter("beta")

    qc = QuantumCircuit(n_qubits, n_qubits)

    # |+>^n
    for q in range(n_qubits):
        qc.h(q)

    # exp(-i gamma sum_ij J_ij Z_i Z_j)
    # qiskit.rzz(theta) = exp(-i theta/2 Z⊗Z), so theta = 2 * gamma * J_ij
    for i, j, Jij in zz_terms:
        qc.rzz(2.0 * Jij * gamma, i, j)

    # exp(-i beta sum_i X_i)
    for q in range(n_qubits):
        qc.rx(2.0 * beta, q)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, [gamma, beta]


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
        energies.append(energy_of_bitstring(bs_q0, H_constant_term, H_zz_terms))

    return float(np.mean(energies))


def optimize_adam_spsa(
    label,
    theta_init,
    transpiled_measured_template,
    ansatz_params,
    simulator,
    shots,
    H_constant_term,
    H_zz_terms,
    seed_base,
    n_epochs,
    lr,
    beta1,
    beta2,
    eps,
    spsa_c,
    spsa_gamma,
    verbose=True,
):
    rng_local = np.random.default_rng(seed_base)

    theta = np.array(theta_init, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    theta_history = [theta.copy()]
    history_rows = []

    best_loss = np.inf
    best_theta = theta.copy()
    best_epoch = 0
    eval_counter = 0

    def objective(theta_local):
        nonlocal eval_counter
        val = evaluate_theta_expected_energy(
            theta=theta_local,
            transpiled_measured_template=transpiled_measured_template,
            ansatz_params=ansatz_params,
            simulator=simulator,
            shots=shots,
            seed_simulator=seed_base + 100000 + eval_counter,
            H_constant_term=H_constant_term,
            H_zz_terms=H_zz_terms,
        )
        eval_counter += 1
        return val

    for epoch in range(1, n_epochs + 1):
        ck = spsa_c / (epoch ** spsa_gamma)
        delta = rng_local.choice([-1.0, 1.0], size=theta.shape)

        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta

        loss_plus = objective(theta_plus)
        loss_minus = objective(theta_minus)
        grad_hat = ((loss_plus - loss_minus) / (2.0 * ck)) * delta

        m = beta1 * m + (1.0 - beta1) * grad_hat
        v = beta2 * v + (1.0 - beta2) * (grad_hat ** 2)

        m_hat = m / (1.0 - beta1 ** epoch)
        v_hat = v / (1.0 - beta2 ** epoch)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

        train_loss = objective(theta)
        grad_norm = float(np.linalg.norm(grad_hat))

        if train_loss < best_loss:
            best_loss = float(train_loss)
            best_theta = theta.copy()
            best_epoch = epoch

        history_rows.append(
            {
                "label": label,
                "epoch": epoch,
                "loss": float(train_loss),
                "loss_plus": float(loss_plus),
                "loss_minus": float(loss_minus),
                "grad_norm": grad_norm,
                "ck": float(ck),
                "best_loss_so_far": float(best_loss),
                "best_epoch_so_far": int(best_epoch),
            }
        )
        theta_history.append(theta.copy())

        if verbose:
            print(
                f"{label} | epoch={epoch:03d} | "
                f"loss={train_loss:.6f} | best={best_loss:.6f} | grad_norm={grad_norm:.6f}"
            )

    return {
        "theta_final": theta.copy(),
        "theta_best": best_theta.copy(),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "theta_history": np.array(theta_history, dtype=float),
        "df_train": pd.DataFrame(history_rows),
    }


def sample_qaoa_distribution(
    theta,
    transpiled_measured_template,
    ansatz_params,
    simulator,
    shots,
    seed_simulator,
    gauge_bitstring_q0_first,
    H0_constant_term,
    H0_zz_terms,
    E_GS_exact,
):
    """
    The circuit is built for H_y, so measured bitstring b is in the transformed frame.
    We map back to the original frame by x = b xor y and compute energies under H_0.
    """
    result = bind_and_run_circuit(
        theta=theta,
        transpiled_measured_template=transpiled_measured_template,
        ansatz_params=ansatz_params,
        simulator=simulator,
        shots=shots,
        seed_simulator=seed_simulator,
    )

    rows = []
    ratios = []
    energies = []
    mapped_back = []

    for k, raw_bs in enumerate(result.get_memory()):
        bs_transformed = qiskit_to_q0_first(raw_bs)
        bs_original = xor_bitstrings(bs_transformed, gauge_bitstring_q0_first)
        energy_original = energy_of_bitstring(bs_original, H0_constant_term, H0_zz_terms)
        ratio = energy_original / float(E_GS_exact)

        rows.append(
            {
                "shot": k,
                "bitstring_transformed_q0_first": bs_transformed,
                "bitstring_original_q0_first": bs_original,
                "energy_original": float(energy_original),
                "approx_ratio": float(ratio),
            }
        )
        energies.append(float(energy_original))
        ratios.append(float(ratio))
        mapped_back.append(bs_original)

    df = pd.DataFrame(rows)
    best_idx = int(np.argmin(energies))
    best_bitstring_original = mapped_back[best_idx]

    return {
        "df_shots": df,
        "ratios": np.array(ratios, dtype=float),
        "energies": np.array(energies, dtype=float),
        "best_bitstring_original": best_bitstring_original,
        "best_energy_original": float(np.min(energies)),
        "best_ratio_original": float(np.max(ratios)),
    }


# ======================================================================================
# LEFT PANEL
# ======================================================================================
def plot_left_panel(df, cfg, out_png):
    fracs = cfg["left_top_fracs"]
    cols = [ratio_label(f) for f in fracs]
    titles = [
        "Top 100%",
        f"Top {int(round(100 * fracs[1]))}%",
        f"Top {100 / cfg['left_shots']:.3f}% (best sample)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, col, title in zip(axes, cols, titles):
        x = df["attractor_ratio"].values
        y = df[col].values

        hb = ax.hist2d(x, y, bins=25)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "--", linewidth=1)
        ax.set_xlabel("Attractor AR")
        ax.set_ylabel("Optimized AR")
        ax.set_title(
            f"{title}\n"
            f"Pearson={pearson_corr(x, y):.3f}, Spearman={spearman_corr(x, y):.3f}"
        )
        fig.colorbar(hb[3], ax=ax)

    fig.suptitle("Fig. 4-like QAOA correlation panel")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def run_left_panel(condition_name, noisy, cfg, out_dir_root):
    out_dir = Path(out_dir_root) / condition_name / "left_panel"
    ensure_dir(out_dir)
    save_json(cfg, out_dir / "config.json")

    simulator, transpile_backend = get_simulator_compat(
        n_qubits=cfg["n_qubits"],
        noisy=noisy,
        extra_amp_damp_1q=cfg["extra_amp_damp_1q"],
        extra_amp_damp_2q=cfg["extra_amp_damp_2q"],
    )

    rows = []
    rng_root = np.random.default_rng(cfg["seed"])

    for instance_id in range(cfg["n_instances"]):
        seed_instance = int(rng_root.integers(0, 2**31 - 1))
        rng_instance = np.random.default_rng(seed_instance)

        H0_constant_term, H0_zz_terms = generate_sk_terms(cfg["n_qubits"], rng_instance)
        E_GS_exact, gs_bitstring = brute_force_ground_energy(cfg["n_qubits"], H0_constant_term, H0_zz_terms)

        for gauge_id in range(cfg["n_gauges"]):
            gauge_bitstring = random_bitstring(cfg["n_qubits"], rng_instance)
            Hy_zz_terms = update_zz_terms_by_gauge(H0_zz_terms, gauge_bitstring)

            qc, ansatz_params = build_qaoa_p1_from_terms(cfg["n_qubits"], Hy_zz_terms)
            transpiled = transpile(
                qc,
                transpile_backend,
                optimization_level=1,
                seed_transpiler=cfg["seed"] + instance_id,
            )

            theta_init = np.array(cfg["theta_init"], dtype=float)
            # fixed optimizer seed across gauges within the same instance
            seed_base = 10000 * instance_id + 123

            opt = optimize_adam_spsa(
                label=f"left_{condition_name}_inst{instance_id:03d}_g{gauge_id:03d}",
                theta_init=theta_init,
                transpiled_measured_template=transpiled,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots=cfg["left_shots"],
                H_constant_term=H0_constant_term,
                H_zz_terms=Hy_zz_terms,
                seed_base=seed_base,
                n_epochs=cfg["left_n_epochs"],
                lr=cfg["adam_lr"],
                beta1=cfg["adam_beta1"],
                beta2=cfg["adam_beta2"],
                eps=cfg["adam_eps"],
                spsa_c=cfg["spsa_c"],
                spsa_gamma=cfg["spsa_gamma"],
                verbose=False,
            )

            sample = sample_qaoa_distribution(
                theta=opt["theta_best"],
                transpiled_measured_template=transpiled,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots=cfg["left_shots"],
                seed_simulator=777000 + instance_id * 1000 + gauge_id,
                gauge_bitstring_q0_first=gauge_bitstring,
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
            )

            attractor_ratio = energy_of_bitstring(gauge_bitstring, H0_constant_term, H0_zz_terms) / E_GS_exact
            row = {
                "condition": condition_name,
                "instance_id": instance_id,
                "gauge_id": gauge_id,
                "gauge_bitstring": gauge_bitstring,
                "attractor_ratio": float(attractor_ratio),
                "E_GS_exact": float(E_GS_exact),
                "ground_state_bitstring": gs_bitstring,
            }
            row.update(summarize_top_fracs(sample["ratios"], cfg["left_top_fracs"]))
            rows.append(row)

            print(
                f"[LEFT][{condition_name}] instance={instance_id:03d} gauge={gauge_id:03d} | "
                f"attractor_ratio={attractor_ratio:.4f} | best_ratio={row['optimized_ratio_best_sample']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fig4_qaoa_left_data.csv", index=False)

    summary_rows = []
    for frac in cfg["left_top_fracs"]:
        col = ratio_label(frac)
        summary_rows.append(
            {
                "panel_frac": frac,
                "panel_col": col,
                "pearson": pearson_corr(df["attractor_ratio"], df[col]),
                "spearman": spearman_corr(df["attractor_ratio"], df[col]),
                "mean_attractor_ratio": float(df["attractor_ratio"].mean()),
                "mean_optimized_ratio": float(df[col].mean()),
            }
        )
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "fig4_qaoa_left_summary.csv", index=False)

    plot_left_panel(df, cfg, out_dir / "fig4_qaoa_left_hist2d.png")
    save_json(
        {
            "condition": condition_name,
            "data_csv": str(out_dir / "fig4_qaoa_left_data.csv"),
            "summary_csv": str(out_dir / "fig4_qaoa_left_summary.csv"),
            "plot": str(out_dir / "fig4_qaoa_left_hist2d.png"),
        },
        out_dir / "summary.json",
    )

    return df, df_summary


# ======================================================================================
# RIGHT PANEL
# ======================================================================================
def sample_random_baseline_best_ratio(n_qubits, total_samples, H0_constant_term, H0_zz_terms, E_GS_exact, rng):
    best_ratio = -np.inf
    best_bs = None
    for _ in range(total_samples):
        bs = "".join(rng.choice(["0", "1"], size=n_qubits))
        ratio = energy_of_bitstring(bs, H0_constant_term, H0_zz_terms) / E_GS_exact
        if ratio > best_ratio:
            best_ratio = float(ratio)
            best_bs = bs
    return float(best_ratio), best_bs


def run_right_panel(condition_name, noisy, cfg, out_dir_root):
    out_dir = Path(out_dir_root) / condition_name / "right_panel"
    ensure_dir(out_dir)
    save_json(cfg, out_dir / "config.json")

    simulator, transpile_backend = get_simulator_compat(
        n_qubits=cfg["n_qubits"],
        noisy=noisy,
        extra_amp_damp_1q=cfg["extra_amp_damp_1q"],
        extra_amp_damp_2q=cfg["extra_amp_damp_2q"],
    )

    rows = []
    rng_root = np.random.default_rng(cfg["seed"])

    for instance_id in range(cfg["n_instances"]):
        seed_instance = int(rng_root.integers(0, 2**31 - 1))
        rng_instance = np.random.default_rng(seed_instance)

        H0_constant_term, H0_zz_terms = generate_sk_terms(cfg["n_qubits"], rng_instance)
        E_GS_exact, gs_bitstring = brute_force_ground_energy(cfg["n_qubits"], H0_constant_term, H0_zz_terms)

        # ---------- NDAR ----------
        gauge_current = "0" * cfg["n_qubits"]
        best_ratio_so_far = -np.inf
        best_energy_so_far = np.inf
        best_bs_so_far = None

        for outer_iter in range(1, cfg["right_K"] + 1):
            Hy_zz_terms = update_zz_terms_by_gauge(H0_zz_terms, gauge_current)
            qc, ansatz_params = build_qaoa_p1_from_terms(cfg["n_qubits"], Hy_zz_terms)
            transpiled = transpile(
                qc,
                transpile_backend,
                optimization_level=1,
                seed_transpiler=cfg["seed"] + instance_id + outer_iter,
            )

            opt = optimize_adam_spsa(
                label=f"ndar_{condition_name}_inst{instance_id:03d}_it{outer_iter:03d}",
                theta_init=np.array(cfg["theta_init"], dtype=float),
                transpiled_measured_template=transpiled,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots=cfg["right_shots"],
                H_constant_term=H0_constant_term,
                H_zz_terms=Hy_zz_terms,
                seed_base=700000 + instance_id * 1000 + outer_iter,
                n_epochs=cfg["right_n_epochs_per_iter"],
                lr=cfg["adam_lr"],
                beta1=cfg["adam_beta1"],
                beta2=cfg["adam_beta2"],
                eps=cfg["adam_eps"],
                spsa_c=cfg["spsa_c"],
                spsa_gamma=cfg["spsa_gamma"],
                verbose=False,
            )

            sample = sample_qaoa_distribution(
                theta=opt["theta_best"],
                transpiled_measured_template=transpiled,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots=cfg["right_shots"],
                seed_simulator=880000 + instance_id * 1000 + outer_iter,
                gauge_bitstring_q0_first=gauge_current,
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
            )

            current_best_bs = sample["best_bitstring_original"]
            current_best_energy = sample["best_energy_original"]
            current_best_ratio = sample["best_ratio_original"]

            if current_best_energy < best_energy_so_far:
                best_energy_so_far = current_best_energy
                best_ratio_so_far = current_best_ratio
                best_bs_so_far = current_best_bs

            rows.append(
                {
                    "condition": condition_name,
                    "instance_id": instance_id,
                    "method": "ndar",
                    "outer_iter": outer_iter,
                    "best_ratio_so_far": float(best_ratio_so_far),
                    "best_energy_so_far": float(best_energy_so_far),
                    "best_bitstring_so_far": best_bs_so_far,
                    "ground_state_bitstring": gs_bitstring,
                }
            )

            gauge_current = current_best_bs

        # ---------- Standard QAOA budget-matched ----------
        best_ratio_plain_so_far = -np.inf
        best_energy_plain_so_far = np.inf
        best_bs_plain_so_far = None

        for outer_iter in range(1, cfg["right_K"] + 1):
            total_epochs = outer_iter * cfg["right_n_epochs_per_iter"]
            qc, ansatz_params = build_qaoa_p1_from_terms(cfg["n_qubits"], H0_zz_terms)
            transpiled = transpile(
                qc,
                transpile_backend,
                optimization_level=1,
                seed_transpiler=cfg["seed"] + 1000 + instance_id + outer_iter,
            )

            opt = optimize_adam_spsa(
                label=f"plain_{condition_name}_inst{instance_id:03d}_budget{outer_iter:03d}",
                theta_init=np.array(cfg["theta_init"], dtype=float),
                transpiled_measured_template=transpiled,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots=cfg["right_shots"],
                H_constant_term=H0_constant_term,
                H_zz_terms=H0_zz_terms,
                seed_base=900000 + instance_id * 1000 + outer_iter,
                n_epochs=total_epochs,
                lr=cfg["adam_lr"],
                beta1=cfg["adam_beta1"],
                beta2=cfg["adam_beta2"],
                eps=cfg["adam_eps"],
                spsa_c=cfg["spsa_c"],
                spsa_gamma=cfg["spsa_gamma"],
                verbose=False,
            )

            sample = sample_qaoa_distribution(
                theta=opt["theta_best"],
                transpiled_measured_template=transpiled,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots=cfg["right_shots"],
                seed_simulator=990000 + instance_id * 1000 + outer_iter,
                gauge_bitstring_q0_first="0" * cfg["n_qubits"],
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
            )

            if sample["best_energy_original"] < best_energy_plain_so_far:
                best_energy_plain_so_far = sample["best_energy_original"]
                best_ratio_plain_so_far = sample["best_ratio_original"]
                best_bs_plain_so_far = sample["best_bitstring_original"]

            rows.append(
                {
                    "condition": condition_name,
                    "instance_id": instance_id,
                    "method": "plain_qaoa",
                    "outer_iter": outer_iter,
                    "best_ratio_so_far": float(best_ratio_plain_so_far),
                    "best_energy_so_far": float(best_energy_plain_so_far),
                    "best_bitstring_so_far": best_bs_plain_so_far,
                    "ground_state_bitstring": gs_bitstring,
                }
            )

        # ---------- Random baseline ----------
        rng_random = np.random.default_rng(500000 + instance_id)
        best_ratio_rand_so_far = -np.inf
        best_energy_rand_so_far = np.inf
        best_bs_rand_so_far = None

        for outer_iter in range(1, cfg["right_K"] + 1):
            total_samples = outer_iter * cfg["right_n_epochs_per_iter"] * cfg["right_shots"]
            ratio_rand, bs_rand = sample_random_baseline_best_ratio(
                n_qubits=cfg["n_qubits"],
                total_samples=total_samples,
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
                rng=rng_random,
            )
            energy_rand = energy_of_bitstring(bs_rand, H0_constant_term, H0_zz_terms)

            if energy_rand < best_energy_rand_so_far:
                best_energy_rand_so_far = energy_rand
                best_ratio_rand_so_far = ratio_rand
                best_bs_rand_so_far = bs_rand

            rows.append(
                {
                    "condition": condition_name,
                    "instance_id": instance_id,
                    "method": "random_sampling",
                    "outer_iter": outer_iter,
                    "best_ratio_so_far": float(best_ratio_rand_so_far),
                    "best_energy_so_far": float(best_energy_rand_so_far),
                    "best_bitstring_so_far": best_bs_rand_so_far,
                    "ground_state_bitstring": gs_bitstring,
                }
            )

        print(f"[RIGHT][{condition_name}] finished instance={instance_id:03d}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fig4_qaoa_right_data.csv", index=False)

    # summary
    df_summary = (
        df.groupby(["method", "outer_iter"], as_index=False)
          .agg(
              mean_best_ratio=("best_ratio_so_far", "mean"),
              std_best_ratio=("best_ratio_so_far", "std"),
              min_best_ratio=("best_ratio_so_far", "min"),
              max_best_ratio=("best_ratio_so_far", "max"),
          )
    )
    df_summary.to_csv(out_dir / "fig4_qaoa_right_summary.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    method_order = ["ndar", "plain_qaoa", "random_sampling"]

    for method in method_order:
        sub = df[df["method"] == method].copy()
        for _, g in sub.groupby("instance_id"):
            ax.plot(g["outer_iter"], g["best_ratio_so_far"], alpha=0.25)

        mean_curve = (
            sub.groupby("outer_iter", as_index=False)["best_ratio_so_far"].mean()
        )
        ax.plot(mean_curve["outer_iter"], mean_curve["best_ratio_so_far"], linewidth=3, label=method)

    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Best approximation ratio so far")
    ax.set_title("Fig. 4-like NDAR vs baselines (QAOA p=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "fig4_qaoa_right_panel.png", dpi=220)
    plt.close(fig)

    save_json(
        {
            "condition": condition_name,
            "data_csv": str(out_dir / "fig4_qaoa_right_data.csv"),
            "summary_csv": str(out_dir / "fig4_qaoa_right_summary.csv"),
            "plot": str(out_dir / "fig4_qaoa_right_panel.png"),
        },
        out_dir / "summary.json",
    )

    return df, df_summary


# ======================================================================================
# MAIN
# ======================================================================================
def main():
    out_dir_root = Path(CONFIG["out_dir"])
    ensure_dir(out_dir_root)
    save_json(CONFIG, out_dir_root / "config_root.json")

    conditions = []
    if CONFIG["run_noiseless"]:
        conditions.append(("noiseless", False))
    if CONFIG["run_noisy"]:
        conditions.append(("noisy", True))

    all_results = []

    for condition_name, noisy in conditions:
        if CONFIG["run_mode"] in ("left", "both"):
            left_df, left_summary = run_left_panel(condition_name, noisy, CONFIG, out_dir_root)
            all_results.append(
                {
                    "condition": condition_name,
                    "panel": "left",
                    "n_rows": int(len(left_df)),
                    "summary_path": str(out_dir_root / condition_name / "left_panel" / "fig4_qaoa_left_summary.csv"),
                }
            )

        if CONFIG["run_mode"] in ("right", "both"):
            right_df, right_summary = run_right_panel(condition_name, noisy, CONFIG, out_dir_root)
            all_results.append(
                {
                    "condition": condition_name,
                    "panel": "right",
                    "n_rows": int(len(right_df)),
                    "summary_path": str(out_dir_root / condition_name / "right_panel" / "fig4_qaoa_right_summary.csv"),
                }
            )

    save_json(all_results, out_dir_root / "summary_master.json")
    print("\nDone. Results saved to:", out_dir_root.resolve())


if __name__ == "__main__":
    main()
