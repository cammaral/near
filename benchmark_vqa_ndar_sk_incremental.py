from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qubo.MaxCut import solve_bruteforce, maxcut_value
from quantum.backend import get_simulator
from quantum.circuit import build_brickwall, build_circuit
from quantum.hamiltonian import (
    build_maxcut_hamiltonian,
    update_hamiltonian,
    hamiltonian_value_for_bitstring,
)
from optimizer.adam_spsa import optimize
from ndar.run import sample_theta_full


# ============================================================
# Config
# ============================================================

@dataclass
class NoiseConfig:
    name: str
    noisy: bool
    extra_amp_damp_1q: float = 0.0
    extra_amp_damp_2q: float = 0.0


CONFIG = {
    # problem / ansatz
    "n_qubits": 10,
    "depth": 1,
    "n_instances": 5,
    "seed": 1924,

    # NDAR outer loop
    "K": 6,
    "n_epochs": 10,

    # shots for optimization objective and for final sampling each step
    "shots_opt": 256,
    "shots_sample": 256,

    # optimizer
    "adam_lr": 0.10,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "spsa_c": 0.15,
    "spsa_gamma": 0.101,

    # random-search baseline
    # optimizer uses roughly 3 objective calls per epoch (plus, minus, theta)
    # so this makes the random baseline budget similar per outer step.
    "n_random_thetas_per_step": None,

    # plotting
    "selected_instance_for_fig1": 0,

    # output
    "out_dir": "results_vqa_ndar_sk",
}

NOISE_SWEEP = [
    NoiseConfig(name="noiseless", noisy=False),
    NoiseConfig(name="default_noise", noisy=True),
    NoiseConfig(name="moderate_damping", noisy=True, extra_amp_damp_1q=0.005, extra_amp_damp_2q=0.02),
    NoiseConfig(name="strong_damping", noisy=True, extra_amp_damp_1q=0.02, extra_amp_damp_2q=0.08),
]


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def xor_bitstrings(a: str, b: str) -> str:
    return "".join("1" if x != y else "0" for x, y in zip(a, b))


def generate_sk_edges(n_qubits: int, rng: np.random.Generator) -> List[Tuple[int, int, float]]:
    edges = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            w = float(rng.choice([-1.0, 1.0]))
            edges.append((i, j, w))
    return edges


def random_theta(rng: np.random.Generator, n_params: int) -> np.ndarray:
    return rng.uniform(-np.pi, np.pi, size=n_params)


def eval_original_metrics(
    bitstring_original: str,
    edges,
    H0_constant_term: float,
    H0_zz_terms,
    E_GS_exact: float,
) -> Dict[str, float]:
    energy = hamiltonian_value_for_bitstring(bitstring_original, H0_constant_term, H0_zz_terms)
    ar = float(energy / E_GS_exact)
    cut = float(maxcut_value(bitstring_original, edges))
    return {
        "energy": float(energy),
        "ar": float(ar),
        "cut": cut,
    }


def make_history_row(
    method: str,
    step: int,
    cumulative_budget_units: int,
    gauge_before: str,
    best_current: str,
    best_original: str,
    attractor_original: str,
    sample_result: dict,
    H0_constant_term: float,
    H0_zz_terms,
    E_GS_exact: float,
    edges,
    note: str = "",
) -> Dict[str, object]:
    attractor_metrics = eval_original_metrics(
        attractor_original,
        edges,
        H0_constant_term,
        H0_zz_terms,
        E_GS_exact,
    )
    best_metrics = eval_original_metrics(
        best_original,
        edges,
        H0_constant_term,
        H0_zz_terms,
        E_GS_exact,
    )
    return {
        "method": method,
        "step": int(step),
        "cumulative_budget_units": int(cumulative_budget_units),
        "gauge_before": gauge_before,
        "best_bitstring_current_frame": best_current,
        "best_bitstring_original": best_original,
        "attractor_bitstring_original": attractor_original,
        "min_energy_current_frame": float(sample_result["min_energy"]),
        "expected_energy_current_frame": float(sample_result["expected_energy"]),
        "best_ratio_current_frame": float(sample_result["best_ratio"]),
        "expected_ratio_current_frame": float(sample_result["expected_ratio"]),
        "zero_ratio_current_frame": float(sample_result["zero_ratio"]),
        "best_energy_original": float(best_metrics["energy"]),
        "best_ar_original": float(best_metrics["ar"]),
        "best_cut_original": float(best_metrics["cut"]),
        "attractor_energy_original": float(attractor_metrics["energy"]),
        "attractor_ar_original": float(attractor_metrics["ar"]),
        "attractor_cut_original": float(attractor_metrics["cut"]),
        "note": note,
    }


def add_cumulative_best_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["method", "step"]).copy()
    out = []
    for method, g in df.groupby("method", sort=False):
        g = g.copy()
        g["cum_best_energy_original"] = np.minimum.accumulate(g["best_energy_original"].to_numpy())
        g["cum_best_ar_original"] = np.maximum.accumulate(g["best_ar_original"].to_numpy())
        out.append(g)
    return pd.concat(out, ignore_index=True)


def save_step_shots(df_shots: pd.DataFrame, gauge_before: str, out_csv: Path) -> None:
    df = df_shots.copy()
    if gauge_before:
        df["bitstring_original"] = [xor_bitstrings(gauge_before, bs) for bs in df["bitstring_q0_first"]]
    else:
        df["bitstring_original"] = df["bitstring_q0_first"]
    df.to_csv(out_csv, index=False)


# ============================================================
# Core runners
# ============================================================

def run_plain_vqa(
    *,
    n_qubits: int,
    theta_init: np.ndarray,
    transpiled_measured_template,
    ansatz_params,
    simulator,
    shots_opt: int,
    shots_sample: int,
    H0_constant_term: float,
    H0_zz_terms,
    E_GS_exact: float,
    edges,
    seed: int,
    K: int,
    n_epochs: int,
    opt_kwargs: dict,
    out_dir: Path,
) -> pd.DataFrame:
    method_dir = ensure_dir(out_dir / "plain_vqa")

    baseline_training = optimize(
        label="PLAIN_VQA",
        theta_init=theta_init.copy(),
        transpiled_measured_template=transpiled_measured_template,
        ansatz_params=ansatz_params,
        simulator=simulator,
        shots=shots_opt,
        H_constant_term=H0_constant_term,
        H_zz_terms=H0_zz_terms,
        out_dir_sub=method_dir,
        seed_base=seed + 110000,
        n_epochs=K * n_epochs,
        **opt_kwargs,
    )

    train_df = baseline_training["df_train"].copy()
    train_df.to_csv(method_dir / "training_history.csv", index=False)

    rows = []
    for block_idx in range(K):
        epoch_checkpoint = (block_idx + 1) * n_epochs
        short_training = optimize(
            label=f"PLAIN_VQA_REPLAY_{epoch_checkpoint}",
            theta_init=theta_init.copy(),
            transpiled_measured_template=transpiled_measured_template,
            ansatz_params=ansatz_params,
            simulator=simulator,
            shots=shots_opt,
            H_constant_term=H0_constant_term,
            H_zz_terms=H0_zz_terms,
            out_dir_sub=method_dir,
            seed_base=seed + 110000,
            n_epochs=epoch_checkpoint,
            **opt_kwargs,
        )
        theta_checkpoint = short_training["theta_best"].copy()

        chk_dir = ensure_dir(method_dir / f"checkpoint_{epoch_checkpoint:03d}")
        sample_result = sample_theta_full(
            n_qubits=n_qubits,
            E_GS_exact=E_GS_exact,
            theta=theta_checkpoint,
            iter_label=f"PLAIN_VQA_step_{block_idx}",
            transpiled_measured_template=transpiled_measured_template,
            ansatz_params=ansatz_params,
            simulator=simulator,
            shots=shots_sample,
            seed_simulator=seed + 120000 + block_idx,
            edges=edges,
            H_constant_term=H0_constant_term,
            H_zz_terms=H0_zz_terms,
            out_dir_sub=chk_dir,
        )
        save_step_shots(sample_result["df_shots"], "", chk_dir / "shots.csv")

        best_current = sample_result["best_bitstring"]
        row = make_history_row(
            method="plain_vqa",
            step=block_idx,
            cumulative_budget_units=epoch_checkpoint,
            gauge_before="0" * len(best_current),
            best_current=best_current,
            best_original=best_current,
            attractor_original="0" * len(best_current),
            sample_result=sample_result,
            H0_constant_term=H0_constant_term,
            H0_zz_terms=H0_zz_terms,
            E_GS_exact=E_GS_exact,
            edges=edges,
            note="Budget-matched cumulative optimization on original Hamiltonian.",
        )
        rows.append(row)

        df_partial = add_cumulative_best_columns(pd.DataFrame(rows))
        df_partial.to_csv(method_dir / "summary_partial.csv", index=False)

    df = add_cumulative_best_columns(pd.DataFrame(rows))
    df.to_csv(method_dir / "summary.csv", index=False)
    return df


def run_ndar_vqa(
    *,
    n_qubits: int,
    theta_init: np.ndarray,
    ansatz,
    transpile_backend,
    ansatz_params,
    simulator,
    shots_opt: int,
    shots_sample: int,
    H0_constant_term: float,
    H0_zz_terms,
    E_GS_exact: float,
    edges,
    seed: int,
    K: int,
    n_epochs: int,
    opt_kwargs: dict,
    out_dir: Path,
    gauge_mode: str = "none",
) -> pd.DataFrame:
    method_dir = ensure_dir(out_dir / "ndar_vqa")

    gauge = "0" * n_qubits
    theta_current = theta_init.copy()
    H_cur_const = H0_constant_term
    H_cur_terms = list(H0_zz_terms)
    rows = []

    for step in range(K):
        step_dir = ensure_dir(method_dir / f"step_{step:03d}")
        transpiled_measured_template = build_circuit(
            n_qubits=n_qubits,
            ansatz=ansatz,
            transpile_backend=transpile_backend,
            seed=seed + 2000 + step,
            gauge_bitstring_q0_first=gauge,
            gauge_mode=gauge_mode,
        )

        train = optimize(
            label=f"NDAR_VQA_step_{step}",
            theta_init=theta_current,
            transpiled_measured_template=transpiled_measured_template,
            ansatz_params=ansatz_params,
            simulator=simulator,
            shots=shots_opt,
            H_constant_term=H_cur_const,
            H_zz_terms=H_cur_terms,
            out_dir_sub=step_dir,
            seed_base=seed + 130000 + step * 1000,
            n_epochs=n_epochs,
            **opt_kwargs,
        )
        train["df_train"].to_csv(step_dir / "training_history.csv", index=False)
        theta_best = train["theta_best"].copy()
        theta_current = theta_best.copy()

        sample_result = sample_theta_full(
            n_qubits=n_qubits,
            E_GS_exact=E_GS_exact,
            theta=theta_best,
            iter_label=f"NDAR_VQA_step_{step}",
            transpiled_measured_template=transpiled_measured_template,
            ansatz_params=ansatz_params,
            simulator=simulator,
            shots=shots_sample,
            seed_simulator=seed + 140000 + step,
            edges=edges,
            H_constant_term=H_cur_const,
            H_zz_terms=H_cur_terms,
            out_dir_sub=step_dir,
        )
        save_step_shots(sample_result["df_shots"], gauge, step_dir / "shots.csv")

        best_current = sample_result["best_bitstring"]
        best_original = xor_bitstrings(gauge, best_current)
        attractor_original = gauge

        rows.append(
            make_history_row(
                method="ndar_vqa",
                step=step,
                cumulative_budget_units=(step + 1) * n_epochs,
                gauge_before=gauge,
                best_current=best_current,
                best_original=best_original,
                attractor_original=attractor_original,
                sample_result=sample_result,
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
                edges=edges,
                note="NDAR outer step: optimize current transformed Hamiltonian, then remap using best sampled bitstring.",
            )
        )

        gauge = xor_bitstrings(gauge, best_current)
        _, H_cur_const, H_cur_terms = update_hamiltonian(n_qubits, H0_constant_term, H0_zz_terms, gauge)

        df_partial = add_cumulative_best_columns(pd.DataFrame(rows))
        df_partial.to_csv(method_dir / "summary_partial.csv", index=False)

    df = add_cumulative_best_columns(pd.DataFrame(rows))
    df.to_csv(method_dir / "summary.csv", index=False)
    return df


def run_random_parameter_search(
    *,
    n_qubits: int,
    n_params: int,
    theta_init: np.ndarray,
    transpiled_measured_template,
    ansatz_params,
    simulator,
    shots_sample: int,
    H0_constant_term: float,
    H0_zz_terms,
    E_GS_exact: float,
    edges,
    seed: int,
    K: int,
    n_epochs: int,
    n_random_thetas_per_step: int,
    out_dir: Path,
) -> pd.DataFrame:
    method_dir = ensure_dir(out_dir / "random_search")
    rng = np.random.default_rng(seed + 150000)

    rows = []
    best_theta_global = theta_init.copy()
    best_energy_global = np.inf

    all_theta_pool = []
    for step in range(K):
        for _ in range(n_random_thetas_per_step):
            all_theta_pool.append(random_theta(rng, n_params))

        step_dir = ensure_dir(method_dir / f"step_{step:03d}")
        candidate_summaries = []
        for idx, theta in enumerate(all_theta_pool):
            sample_result = sample_theta_full(
                n_qubits=n_qubits,
                E_GS_exact=E_GS_exact,
                theta=theta,
                iter_label=f"RANDOM_SEARCH_eval_step_{step}_cand_{idx}",
                transpiled_measured_template=transpiled_measured_template,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots=max(32, shots_sample // 4),
                seed_simulator=seed + 151000 + 1000 * step + idx,
                edges=edges,
                H_constant_term=H0_constant_term,
                H_zz_terms=H0_zz_terms,
                out_dir_sub=step_dir,
            )
            candidate_summaries.append((sample_result["expected_energy"], theta.copy()))

        candidate_summaries.sort(key=lambda x: x[0])
        best_theta_step = candidate_summaries[0][1].copy()
        if candidate_summaries[0][0] < best_energy_global:
            best_energy_global = float(candidate_summaries[0][0])
            best_theta_global = best_theta_step.copy()

        final_sample = sample_theta_full(
            n_qubits=n_qubits,
            E_GS_exact=E_GS_exact,
            theta=best_theta_global,
            iter_label=f"RANDOM_SEARCH_step_{step}",
            transpiled_measured_template=transpiled_measured_template,
            ansatz_params=ansatz_params,
            simulator=simulator,
            shots=shots_sample,
            seed_simulator=seed + 152000 + step,
            edges=edges,
            H_constant_term=H0_constant_term,
            H_zz_terms=H0_zz_terms,
            out_dir_sub=step_dir,
        )
        save_step_shots(final_sample["df_shots"], "", step_dir / "shots.csv")

        best_current = final_sample["best_bitstring"]
        rows.append(
            make_history_row(
                method="random_search",
                step=step,
                cumulative_budget_units=(step + 1) * n_epochs,
                gauge_before="0" * n_qubits,
                best_current=best_current,
                best_original=best_current,
                attractor_original="0" * n_qubits,
                sample_result=final_sample,
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
                edges=edges,
                note="Random-parameter search on original Hamiltonian; keep the best parameter vector found so far.",
            )
        )

        df_partial = add_cumulative_best_columns(pd.DataFrame(rows))
        df_partial.to_csv(method_dir / "summary_partial.csv", index=False)

    df = add_cumulative_best_columns(pd.DataFrame(rows))
    df.to_csv(method_dir / "summary.csv", index=False)
    return df


def run_ndar_no_opt(
    *,
    n_qubits: int,
    theta_fixed: np.ndarray,
    ansatz,
    transpile_backend,
    ansatz_params,
    simulator,
    shots_sample: int,
    H0_constant_term: float,
    H0_zz_terms,
    E_GS_exact: float,
    edges,
    seed: int,
    K: int,
    n_epochs: int,
    out_dir: Path,
    gauge_mode: str = "none",
) -> pd.DataFrame:
    method_dir = ensure_dir(out_dir / "ndar_no_opt")

    gauge = "0" * n_qubits
    H_cur_const = H0_constant_term
    H_cur_terms = list(H0_zz_terms)
    rows = []

    for step in range(K):
        step_dir = ensure_dir(method_dir / f"step_{step:03d}")
        transpiled_measured_template = build_circuit(
            n_qubits=n_qubits,
            ansatz=ansatz,
            transpile_backend=transpile_backend,
            seed=seed + 3000 + step,
            gauge_bitstring_q0_first=gauge,
            gauge_mode=gauge_mode,
        )

        sample_result = sample_theta_full(
            n_qubits=n_qubits,
            E_GS_exact=E_GS_exact,
            theta=theta_fixed,
            iter_label=f"NDAR_NO_OPT_step_{step}",
            transpiled_measured_template=transpiled_measured_template,
            ansatz_params=ansatz_params,
            simulator=simulator,
            shots=shots_sample,
            seed_simulator=seed + 160000 + step,
            edges=edges,
            H_constant_term=H_cur_const,
            H_zz_terms=H_cur_terms,
            out_dir_sub=step_dir,
        )
        save_step_shots(sample_result["df_shots"], gauge, step_dir / "shots.csv")

        best_current = sample_result["best_bitstring"]
        best_original = xor_bitstrings(gauge, best_current)
        attractor_original = gauge
        rows.append(
            make_history_row(
                method="ndar_no_opt",
                step=step,
                cumulative_budget_units=(step + 1) * n_epochs,
                gauge_before=gauge,
                best_current=best_current,
                best_original=best_original,
                attractor_original=attractor_original,
                sample_result=sample_result,
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
                edges=edges,
                note="NDAR-style remap loop with fixed circuit parameters and no optimization.",
            )
        )

        gauge = xor_bitstrings(gauge, best_current)
        _, H_cur_const, H_cur_terms = update_hamiltonian(n_qubits, H0_constant_term, H0_zz_terms, gauge)

        df_partial = add_cumulative_best_columns(pd.DataFrame(rows))
        df_partial.to_csv(method_dir / "summary_partial.csv", index=False)

    df = add_cumulative_best_columns(pd.DataFrame(rows))
    df.to_csv(method_dir / "summary.csv", index=False)
    return df


# ============================================================
# Plotting
# ============================================================

def plot_fig1_like(
    instance_dir: Path,
    df_instance: pd.DataFrame,
    methods_to_overlay=("plain_vqa", "random_search", "ndar_no_opt"),
) -> None:
    df_ndar = df_instance[df_instance["method"] == "ndar_vqa"].sort_values("step")
    if df_ndar.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    for _, row in df_ndar.iterrows():
        step = int(row["step"])
        shots_path = instance_dir / "ndar_vqa" / f"step_{step:03d}" / "shots.csv"
        if not shots_path.exists():
            continue
        df_shots = pd.read_csv(shots_path)
        y = np.full(len(df_shots), step)
        ax.scatter(df_shots["approx_ratio"], y, s=8, alpha=0.18, color="tab:blue")

        x_attr = row["attractor_ar_original"]
        x_best = row["best_ar_original"]
        ax.scatter([x_attr], [step], s=90, marker="o", color="purple", zorder=5)
        ax.scatter([x_best], [step], s=90, marker="D", color="cyan", edgecolor="k", zorder=6)
        ax.plot([x_attr, x_best], [step, step], color="gray", lw=2, alpha=0.8)

    rows = list(df_ndar.to_dict("records"))
    for k in range(len(rows) - 1):
        x_best = rows[k]["best_ar_original"]
        x_next_attr = rows[k + 1]["attractor_ar_original"]
        ax.plot([x_best, x_next_attr], [rows[k]["step"], rows[k + 1]["step"]], color="black", lw=1.6, alpha=0.9)

    style_map = {
        "plain_vqa": ("tab:orange", "Plain VQA"),
        "random_search": ("tab:green", "Random param search"),
        "ndar_no_opt": ("tab:red", "NDAR no opt"),
    }
    for method in methods_to_overlay:
        df_m = df_instance[df_instance["method"] == method].sort_values("step")
        if df_m.empty:
            continue
        color, label = style_map[method]
        ax.plot(
            df_m["cum_best_ar_original"],
            df_m["step"],
            marker="o",
            lw=2.0,
            color=color,
            label=label,
        )

    ax.set_xlabel("Approximation ratio (AR)")
    ax.set_ylabel("Outer step")
    ax.set_title("Fig. 1-style NDAR staircase with baseline overlays")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(instance_dir / "fig1_like_overlay.png", dpi=220)
    plt.close(fig)


def plot_metric_evolution(
    df_all: pd.DataFrame,
    out_path: Path,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    if df_all.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    label_map = {
        "ndar_vqa": "NDAR-VQA",
        "plain_vqa": "Plain VQA",
        "random_search": "Random search",
        "ndar_no_opt": "NDAR no opt",
    }

    for method, g in df_all.groupby("method", sort=False):
        for _, gi in g.groupby(["noise", "instance_idx", "method"]):
            gi = gi.sort_values("step")
            ax.plot(gi["step"], gi[metric], alpha=0.15, lw=1)

        agg = g.groupby("step")[metric].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0.0)
        ax.plot(
            agg["step"],
            agg["mean"],
            lw=2.8,
            marker="o",
            label=label_map.get(method, method),
        )
        ax.fill_between(
            agg["step"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.18,
        )

    ax.set_xlabel("Outer step / matched budget checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_all_noises_combined(
    df_all: pd.DataFrame,
    out_path: Path,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    if df_all.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 7))

    method_name = {
        "ndar_vqa": "NDAR-VQA",
        "plain_vqa": "Plain VQA",
        "random_search": "Random search",
        "ndar_no_opt": "NDAR no opt",
    }

    for (noise_name, method), g in df_all.groupby(["noise", "method"], sort=False):
        agg = g.groupby("step")[metric].mean().reset_index()
        ax.plot(
            agg["step"],
            agg[metric],
            marker="o",
            lw=2,
            label=f"{method_name.get(method, method)} | {noise_name}",
        )

    ax.set_xlabel("Outer step / matched budget checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_instance_progress(df_instance: pd.DataFrame, out_path: Path, metric: str, ylabel: str, title: str) -> None:
    if df_instance.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    name_map = {
        "ndar_vqa": "NDAR-VQA",
        "plain_vqa": "Plain VQA",
        "random_search": "Random search",
        "ndar_no_opt": "NDAR no opt",
    }

    for method, g in df_instance.groupby("method", sort=False):
        g = g.sort_values("step")
        ax.plot(g["step"], g[metric], marker="o", lw=2.2, label=name_map.get(method, method))

    ax.set_xlabel("Outer step / matched budget checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


# ============================================================
# Incremental persistence helpers
# ============================================================

def refresh_instance_outputs(instance_dir: Path, partial_frames: List[pd.DataFrame], make_fig1: bool) -> None:
    if not partial_frames:
        return

    df_instance = pd.concat(partial_frames, ignore_index=True)
    df_instance.to_csv(instance_dir / "all_methods_summary_partial.csv", index=False)

    plot_instance_progress(
        df_instance,
        instance_dir / "compare_instance_ar_progress.png",
        metric="cum_best_ar_original",
        ylabel="Cumulative best AR",
        title="Instance-level AR progress (updated after each method)",
    )
    plot_instance_progress(
        df_instance,
        instance_dir / "compare_instance_energy_progress.png",
        metric="cum_best_energy_original",
        ylabel="Cumulative best energy",
        title="Instance-level energy progress (updated after each method)",
    )

    if make_fig1:
        plot_fig1_like(instance_dir, df_instance)


def refresh_noise_outputs(noise_dir: Path, noise_rows: List[pd.DataFrame], noise_name: str) -> None:
    if not noise_rows:
        return

    df_noise = pd.concat(noise_rows, ignore_index=True)
    df_noise.to_csv(noise_dir / "all_instances_summary_partial.csv", index=False)

    plot_metric_evolution(
        df_noise,
        noise_dir / "compare_ar_evolution.png",
        metric="cum_best_ar_original",
        ylabel="Cumulative best AR",
        title=f"AR evolution across methods | noise={noise_name}",
    )
    plot_metric_evolution(
        df_noise,
        noise_dir / "compare_energy_evolution.png",
        metric="cum_best_energy_original",
        ylabel="Cumulative best energy",
        title=f"Energy evolution across methods | noise={noise_name}",
    )


def refresh_global_outputs(out_dir: Path, all_rows: List[pd.DataFrame]) -> None:
    if not all_rows:
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(out_dir / "all_noises_all_instances_summary_partial.csv", index=False)

    plot_all_noises_combined(
        df_all,
        out_dir / "all_noises_combined_ar.png",
        metric="cum_best_ar_original",
        ylabel="Cumulative best AR",
        title="All noise regimes together: AR evolution",
    )
    plot_all_noises_combined(
        df_all,
        out_dir / "all_noises_combined_energy.png",
        metric="cum_best_energy_original",
        ylabel="Cumulative best energy",
        title="All noise regimes together: energy evolution",
    )


# ============================================================
# Main benchmark
# ============================================================

def main() -> None:
    cfg = CONFIG.copy()
    out_dir = ensure_dir(Path(cfg["out_dir"]))

    if cfg["n_random_thetas_per_step"] is None:
        cfg["n_random_thetas_per_step"] = 3 * cfg["n_epochs"]

    save_json(out_dir / "config.json", cfg)
    save_json(out_dir / "noise_sweep.json", [asdict(x) for x in NOISE_SWEEP])

    global_rng = np.random.default_rng(cfg["seed"])
    all_rows: List[pd.DataFrame] = []

    for noise in NOISE_SWEEP:
        noise_dir = ensure_dir(out_dir / noise.name)
        print(f"\n{'=' * 80}\nNoise regime: {noise.name}\n{'=' * 80}", flush=True)

        simulator, transpile_backend = get_simulator(
            n_qubits=cfg["n_qubits"],
            noisy=noise.noisy,
            extra_amp_damp_1q=noise.extra_amp_damp_1q,
            extra_amp_damp_2q=noise.extra_amp_damp_2q,
        )

        ansatz, ansatz_params = build_brickwall(cfg["n_qubits"], cfg["depth"])
        n_params = len(ansatz_params)
        transpiled_plain_template = build_circuit(
            n_qubits=cfg["n_qubits"],
            ansatz=ansatz,
            transpile_backend=transpile_backend,
            seed=cfg["seed"],
            gauge_mode="none",
        )

        noise_rows: List[pd.DataFrame] = []
        for instance_idx in range(cfg["n_instances"]):
            print(f"\n--- instance {instance_idx + 1}/{cfg['n_instances']} | noise={noise.name} ---", flush=True)
            instance_dir = ensure_dir(noise_dir / f"instance_{instance_idx:03d}")
            rng = np.random.default_rng(cfg["seed"] + 100 * instance_idx)

            edges = generate_sk_edges(cfg["n_qubits"], rng)
            _, H0_constant_term, H0_zz_terms = build_maxcut_hamiltonian(cfg["n_qubits"], edges)
            E_GS_exact, best_bitstrings = solve_bruteforce(cfg["n_qubits"], H0_constant_term, H0_zz_terms)
            save_json(
                instance_dir / "instance_info.json",
                {
                    "noise": asdict(noise),
                    "instance_idx": instance_idx,
                    "edges": [{"i": i, "j": j, "w": w} for i, j, w in edges],
                    "E_GS_exact": E_GS_exact,
                    "best_bitstrings": best_bitstrings,
                },
            )

            theta_init = random_theta(global_rng, n_params)

            opt_kwargs = {
                "lr": cfg["adam_lr"],
                "beta1": cfg["adam_beta1"],
                "beta2": cfg["adam_beta2"],
                "eps": cfg["adam_eps"],
                "spsa_c": cfg["spsa_c"],
                "spsa_gamma": cfg["spsa_gamma"],
            }

            partial_frames: List[pd.DataFrame] = []
            make_fig1 = instance_idx == cfg["selected_instance_for_fig1"]

            print("  -> running plain_vqa", flush=True)
            df_plain = run_plain_vqa(
                n_qubits=cfg["n_qubits"],
                theta_init=theta_init,
                transpiled_measured_template=transpiled_plain_template,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots_opt=cfg["shots_opt"],
                shots_sample=cfg["shots_sample"],
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
                edges=edges,
                seed=cfg["seed"] + 10000 * instance_idx,
                K=cfg["K"],
                n_epochs=cfg["n_epochs"],
                opt_kwargs=opt_kwargs,
                out_dir=instance_dir,
            )
            df_plain["instance_idx"] = instance_idx
            df_plain["noise"] = noise.name
            partial_frames.append(df_plain)
            refresh_instance_outputs(instance_dir, partial_frames, make_fig1)

            print("  -> running ndar_vqa", flush=True)
            df_ndar = run_ndar_vqa(
                n_qubits=cfg["n_qubits"],
                theta_init=theta_init,
                ansatz=ansatz,
                transpile_backend=transpile_backend,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots_opt=cfg["shots_opt"],
                shots_sample=cfg["shots_sample"],
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
                edges=edges,
                seed=cfg["seed"] + 10000 * instance_idx,
                K=cfg["K"],
                n_epochs=cfg["n_epochs"],
                opt_kwargs=opt_kwargs,
                out_dir=instance_dir,
                gauge_mode="none",
            )
            df_ndar["instance_idx"] = instance_idx
            df_ndar["noise"] = noise.name
            partial_frames.append(df_ndar)
            refresh_instance_outputs(instance_dir, partial_frames, make_fig1)

            print("  -> running random_search", flush=True)
            df_rand = run_random_parameter_search(
                n_qubits=cfg["n_qubits"],
                n_params=n_params,
                theta_init=theta_init,
                transpiled_measured_template=transpiled_plain_template,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots_sample=cfg["shots_sample"],
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
                edges=edges,
                seed=cfg["seed"] + 10000 * instance_idx,
                K=cfg["K"],
                n_epochs=cfg["n_epochs"],
                n_random_thetas_per_step=cfg["n_random_thetas_per_step"],
                out_dir=instance_dir,
            )
            df_rand["instance_idx"] = instance_idx
            df_rand["noise"] = noise.name
            partial_frames.append(df_rand)
            refresh_instance_outputs(instance_dir, partial_frames, make_fig1)

            print("  -> running ndar_no_opt", flush=True)
            df_noopt = run_ndar_no_opt(
                n_qubits=cfg["n_qubits"],
                theta_fixed=theta_init,
                ansatz=ansatz,
                transpile_backend=transpile_backend,
                ansatz_params=ansatz_params,
                simulator=simulator,
                shots_sample=cfg["shots_sample"],
                H0_constant_term=H0_constant_term,
                H0_zz_terms=H0_zz_terms,
                E_GS_exact=E_GS_exact,
                edges=edges,
                seed=cfg["seed"] + 10000 * instance_idx,
                K=cfg["K"],
                n_epochs=cfg["n_epochs"],
                out_dir=instance_dir,
                gauge_mode="none",
            )
            df_noopt["instance_idx"] = instance_idx
            df_noopt["noise"] = noise.name
            partial_frames.append(df_noopt)
            refresh_instance_outputs(instance_dir, partial_frames, make_fig1)

            df_instance = pd.concat(partial_frames, ignore_index=True)
            df_instance.to_csv(instance_dir / "all_methods_summary.csv", index=False)

            noise_rows.append(df_instance)
            all_rows.append(df_instance)
            refresh_noise_outputs(noise_dir, noise_rows, noise.name)
            refresh_global_outputs(out_dir, all_rows)

        df_noise = pd.concat(noise_rows, ignore_index=True)
        df_noise.to_csv(noise_dir / "all_instances_summary.csv", index=False)
        refresh_noise_outputs(noise_dir, noise_rows, noise.name)
        refresh_global_outputs(out_dir, all_rows)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(out_dir / "all_noises_all_instances_summary.csv", index=False)
    refresh_global_outputs(out_dir, all_rows)

    print("\nDone. Results saved in:", out_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
