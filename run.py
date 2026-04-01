# benchmark_ndar.py

from pathlib import Path
from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from qubo.MaxCut import generate, solve_bruteforce, maxcut_value
from quantum.backend import get_simulator
from quantum.circuit import build_brickwall, build_circuit
from quantum.hamiltonian import build_maxcut_hamiltonian, update_hamiltonian
from ndar.run import sample_theta_full
from ndar.utils import evaluate_theta_expected_energy


# ============================================================
# 1) CONFIG
# ============================================================
@dataclass
class Config:
    # problem
    n_qubits: int = 20
    edge_prob: float = 0.8 
    depth: int = 2
    shots: int = 1000
    seed: int = 1924
    weight_mode: str = "unit"

    # NDAR outer loop
    K: int = 6
    n_epochs: int = 10

    # optimizer hyperparameters
    adam_lr: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    spsa_c: float = 0.15
    spsa_gamma: float = 0.101

    # NDAR options
    warm_start_next_iter: bool = False
    stop_rule: str = "none"   # "none" or "paper"

    # exact solve
    use_exact: bool = True
    max_exact_qubits: int = 20

    # output
    out_dir: str = "benchmark_ndar_results"

    @property
    def total_baseline_epochs(self):
        return self.K * self.n_epochs


# ============================================================
# 2) HELPERS
# ============================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def xor_bitstrings(a: str, b: str) -> str:
    return "".join("1" if x != y else "0" for x, y in zip(a, b))


def adjacency_matrix_from_edges(n_qubits, edges):
    A = np.zeros((n_qubits, n_qubits), dtype=float)
    for i, j, w in edges:
        A[i, j] = w
        A[j, i] = w
    return A


def get_exact_solution(cfg: Config, n_qubits, H_constant_term, H_zz_terms):
    if cfg.use_exact and n_qubits <= cfg.max_exact_qubits:
        E_GS_exact, exact_ground_states = solve_bruteforce(
            n_qubits=n_qubits,
            constant_term=H_constant_term,
            zz_terms=H_zz_terms,
        )
        return {
            "exact_available": True,
            "E_GS_exact": float(E_GS_exact),
            "maxcut_exact": float(-E_GS_exact),
            "degeneracy": int(len(exact_ground_states)),
            "ground_states_q0_first": exact_ground_states[:50],
        }
    else:
        raise ValueError(
            f"Exact solution disabled or too large for n_qubits={n_qubits}. "
            f"Current sample_theta_full needs E_GS_exact."
        )


def build_problem_instance(cfg: Config, graph_seed: int):
    rng = np.random.default_rng(graph_seed)

    simulator = get_simulator(n_qubits=cfg.n_qubits)

    edges = generate(
        n_qubits=cfg.n_qubits,
        edge_prob=cfg.edge_prob,
        rng=rng,
        weight_mode=cfg.weight_mode,
    )

    ansatz, params = build_brickwall(cfg.n_qubits, cfg.depth)
    transpiled_circuit = build_circuit(
        n_qubits=cfg.n_qubits,
        ansatz=ansatz,
        simulator=simulator,
        seed=graph_seed,
    )

    H_initial, H_constant_term_initial, H_zz_terms_initial = build_maxcut_hamiltonian(
        cfg.n_qubits, edges
    )

    exact_info = get_exact_solution(
        cfg,
        cfg.n_qubits,
        H_constant_term_initial,
        H_zz_terms_initial,
    )

    return {
        "simulator": simulator,
        "edges": edges,
        "adjacency": adjacency_matrix_from_edges(cfg.n_qubits, edges),
        "ansatz_params": params,
        "transpiled_circuit": transpiled_circuit,
        "H_initial": H_initial,
        "H_constant_term_initial": H_constant_term_initial,
        "H_zz_terms_initial": H_zz_terms_initial,
        "exact_info": exact_info,
    }


# ============================================================
# 3) PRINT HELPERS
# ============================================================
def print_run_summary(df_summary_run: pd.DataFrame, sweep_name, sweep_value, graph_seed):
    print("\n" + "=" * 100)
    print(f"FINISHED RUN | {sweep_name} = {sweep_value} | graph_seed = {graph_seed}")
    print("=" * 100)

    cols = [
        "method",
        "final_epochs",
        "final_best_ratio",
        "final_expected_ratio",
        "final_min_energy",
        "final_expected_energy",
        "best_ratio_gain",
        "stopped_early",
        "stop_iteration",
        "stop_reason",
    ]

    df_show = df_summary_run[cols].copy()
    print(df_show.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    best_row = df_summary_run.sort_values("final_best_ratio", ascending=False).iloc[0]
    print(
        f"\nWinner: {best_row['method']} "
        f"| final_best_ratio = {best_row['final_best_ratio']:.6f}"
    )

    print("=" * 100, flush=True)


def print_value_aggregate(df_summary_value: pd.DataFrame, sweep_name, sweep_value):
    print("\n" + "#" * 100)
    print(f"AGGREGATE OVER SEEDS | {sweep_name} = {sweep_value}")
    print("#" * 100)

    agg = (
        df_summary_value
        .groupby("method")[["final_best_ratio", "final_expected_ratio", "best_ratio_gain"]]
        .agg(["mean", "std"])
    )

    print(agg.to_string(float_format=lambda x: f"{x:.6f}"))
    print("#" * 100, flush=True)


# ============================================================
# 4) LOCAL OPTIMIZER WITH HISTORY + TQDM
# ============================================================
def optimize_with_history(
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
    show_tqdm=True,
):
    rng_local = np.random.default_rng(seed_base + abs(hash(label)) % (10**6))

    theta = theta_init.copy()
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

    epoch_iter = range(1, n_epochs + 1)
    if show_tqdm:
        epoch_iter = tqdm(
            epoch_iter,
            desc=label,
            leave=False,
            dynamic_ncols=True,
        )

    for epoch in epoch_iter:
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

        if show_tqdm and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                loss=f"{train_loss:.5f}",
                best=f"{best_loss:.5f}",
                grad=f"{grad_norm:.5f}",
            )

        if verbose and not show_tqdm:
            print(
                f"{label} | epoch={epoch:03d} | "
                f"loss={train_loss:.6f} | best={best_loss:.6f} | grad_norm={grad_norm:.6f}"
            )

    df_train = pd.DataFrame(history_rows)

    return {
        "theta_final": theta.copy(),
        "theta_best": best_theta.copy(),
        "theta_history": np.array(theta_history, dtype=float),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "df_train": df_train.copy(),
    }


# ============================================================
# 5) NDAR RUN
# ============================================================
def run_ndar(cfg: Config, instance: dict, graph_seed: int, verbose=True, show_tqdm=True):
    E_GS_exact = instance["exact_info"]["E_GS_exact"]

    rng = np.random.default_rng(graph_seed + 12345)
    n_params = len(instance["ansatz_params"])

    theta_init_shared = rng.uniform(-0.1, 0.1, size=n_params)
    theta_for_next_iter = theta_init_shared.copy()

    current_H_constant = instance["H_constant_term_initial"]
    current_H_terms = instance["H_zz_terms_initial"].copy()

    cumulative_gauge = "0" * cfg.n_qubits

    rows = []
    ratio_distributions = []
    train_histories = []

    prev_min_energy = None
    prev_expected_energy = None

    stopped_early = False
    stop_iteration = None
    stop_reason = None

    outer_iter = range(cfg.K)
    if show_tqdm:
        outer_iter = tqdm(
            outer_iter,
            desc=f"NDAR outer | seed={graph_seed}",
            leave=False,
            dynamic_ncols=True,
        )

    for k_iter in outer_iter:
        if verbose:
            print("=" * 90)
            print(f"NDAR ITERATION {k_iter + 1}/{cfg.K}")
            print("=" * 90)

        train_result = optimize_with_history(
            label=f"NDAR_iter_{k_iter}",
            theta_init=theta_for_next_iter,
            transpiled_measured_template=instance["transpiled_circuit"],
            ansatz_params=instance["ansatz_params"],
            simulator=instance["simulator"],
            shots=cfg.shots,
            H_constant_term=current_H_constant,
            H_zz_terms=current_H_terms,
            seed_base=graph_seed + 1000 * (k_iter + 1),
            n_epochs=cfg.n_epochs,
            lr=cfg.adam_lr,
            beta1=cfg.adam_beta1,
            beta2=cfg.adam_beta2,
            eps=cfg.adam_eps,
            spsa_c=cfg.spsa_c,
            spsa_gamma=cfg.spsa_gamma,
            verbose=verbose,
            show_tqdm=show_tqdm,
        )

        theta_best = train_result["theta_best"].copy()
        train_histories.append(train_result["df_train"].copy())

        sample_result = sample_theta_full(
            n_qubits=cfg.n_qubits,
            E_GS_exact=E_GS_exact,
            theta=theta_best,
            iter_label=f"NDAR_iter_{k_iter}",
            transpiled_measured_template=instance["transpiled_circuit"],
            ansatz_params=instance["ansatz_params"],
            simulator=instance["simulator"],
            shots=cfg.shots,
            seed_simulator=graph_seed + 500000 + k_iter,
            edges=instance["edges"],
            H_constant_term=current_H_constant,
            H_zz_terms=current_H_terms,
            out_dir_sub=None,
        )

        best_bitstring_current = sample_result["best_bitstring"]
        best_bitstring_original = xor_bitstrings(best_bitstring_current, cumulative_gauge)
        best_cut_original = maxcut_value(best_bitstring_original, instance["edges"])

        rows.append(
            {
                "method": "NDAR",
                "graph_seed": graph_seed,
                "iteration": k_iter,
                "cumulative_epochs": (k_iter + 1) * cfg.n_epochs,
                "best_bitstring_current_frame": best_bitstring_current,
                "best_bitstring_original_frame": best_bitstring_original,
                "min_energy": sample_result["min_energy"],
                "expected_energy": sample_result["expected_energy"],
                "best_cut_original": best_cut_original,
                "best_ratio": sample_result["best_ratio"],
                "expected_ratio": sample_result["expected_ratio"],
                "zero_ratio": sample_result["zero_ratio"],
                "best_train_loss": train_result["best_loss"],
                "best_epoch": train_result["best_epoch"],
                "stopped_early": False,
                "stop_iteration": np.nan,
                "stop_reason": "",
            }
        )

        ratio_distributions.append(np.array(sample_result["ratios"], dtype=float))

        if cfg.stop_rule == "paper" and prev_min_energy is not None:
            no_min_improvement = sample_result["min_energy"] >= prev_min_energy
            no_mean_improvement = sample_result["expected_energy"] >= prev_expected_energy

            if no_min_improvement and no_mean_improvement:
                stopped_early = True
                stop_iteration = k_iter
                stop_reason = "paper_rule_no_min_and_no_mean_improvement"

                rows[-1]["stopped_early"] = True
                rows[-1]["stop_iteration"] = k_iter
                rows[-1]["stop_reason"] = stop_reason

                if verbose:
                    print(f"[STOP] NDAR stopped at iteration {k_iter} due to paper rule.")
                break

        prev_min_energy = sample_result["min_energy"]
        prev_expected_energy = sample_result["expected_energy"]

        _, next_H_constant, next_H_terms = update_hamiltonian(
            n_qubits=cfg.n_qubits,
            constant_term=current_H_constant,
            zz_terms=current_H_terms,
            bitstring_q0_first=best_bitstring_current,
        )

        cumulative_gauge = xor_bitstrings(cumulative_gauge, best_bitstring_current)

        current_H_constant = next_H_constant
        current_H_terms = next_H_terms.copy()

        if cfg.warm_start_next_iter:
            theta_for_next_iter = theta_best.copy()
        else:
            theta_for_next_iter = theta_init_shared.copy()

    df = pd.DataFrame(rows)

    return {
        "df": df,
        "ratio_distributions": ratio_distributions,
        "train_histories": train_histories,
        "stopped_early": stopped_early,
        "stop_iteration": stop_iteration,
        "stop_reason": stop_reason,
    }


# ============================================================
# 6) FIXED-H TOTAL BASELINE
# ============================================================
def run_fixed_h_total(cfg: Config, instance: dict, graph_seed: int, verbose=True, show_tqdm=True):
    """
    One optimizer run on the original Hamiltonian for K*n_epochs,
    then sample at checkpoints n_epochs, 2*n_epochs, ..., K*n_epochs.
    """
    E_GS_exact = instance["exact_info"]["E_GS_exact"]

    rng = np.random.default_rng(graph_seed + 12345)
    n_params = len(instance["ansatz_params"])
    theta_init_shared = rng.uniform(-0.1, 0.1, size=n_params)

    train_result = optimize_with_history(
        label="FIXED_H_TOTAL",
        theta_init=theta_init_shared,
        transpiled_measured_template=instance["transpiled_circuit"],
        ansatz_params=instance["ansatz_params"],
        simulator=instance["simulator"],
        shots=cfg.shots,
        H_constant_term=instance["H_constant_term_initial"],
        H_zz_terms=instance["H_zz_terms_initial"],
        seed_base=graph_seed + 700000,
        n_epochs=cfg.total_baseline_epochs,
        lr=cfg.adam_lr,
        beta1=cfg.adam_beta1,
        beta2=cfg.adam_beta2,
        eps=cfg.adam_eps,
        spsa_c=cfg.spsa_c,
        spsa_gamma=cfg.spsa_gamma,
        verbose=verbose,
        show_tqdm=show_tqdm,
    )

    theta_history = train_result["theta_history"]

    rows = []
    ratio_distributions = []

    checkpoint_iter = range(cfg.K)
    if show_tqdm:
        checkpoint_iter = tqdm(
            checkpoint_iter,
            desc=f"FIXED checkpoints | seed={graph_seed}",
            leave=False,
            dynamic_ncols=True,
        )

    for block_idx in checkpoint_iter:
        epoch_checkpoint = (block_idx + 1) * cfg.n_epochs
        theta_checkpoint = theta_history[epoch_checkpoint]

        sample_result = sample_theta_full(
            n_qubits=cfg.n_qubits,
            E_GS_exact=E_GS_exact,
            theta=theta_checkpoint,
            iter_label=f"FIXED_H_TOTAL_epoch_{epoch_checkpoint}",
            transpiled_measured_template=instance["transpiled_circuit"],
            ansatz_params=instance["ansatz_params"],
            simulator=instance["simulator"],
            shots=cfg.shots,
            seed_simulator=graph_seed + 710000 + block_idx,
            edges=instance["edges"],
            H_constant_term=instance["H_constant_term_initial"],
            H_zz_terms=instance["H_zz_terms_initial"],
            out_dir_sub=None,
        )

        rows.append(
            {
                "method": "FIXED_H_TOTAL",
                "graph_seed": graph_seed,
                "iteration": block_idx,
                "cumulative_epochs": epoch_checkpoint,
                "best_bitstring_current_frame": sample_result["best_bitstring"],
                "best_bitstring_original_frame": sample_result["best_bitstring"],
                "min_energy": sample_result["min_energy"],
                "expected_energy": sample_result["expected_energy"],
                "best_cut_original": maxcut_value(sample_result["best_bitstring"], instance["edges"]),
                "best_ratio": sample_result["best_ratio"],
                "expected_ratio": sample_result["expected_ratio"],
                "zero_ratio": sample_result["zero_ratio"],
                "best_train_loss": train_result["best_loss"],
                "best_epoch": train_result["best_epoch"],
                "stopped_early": False,
                "stop_iteration": np.nan,
                "stop_reason": "",
            }
        )

        ratio_distributions.append(np.array(sample_result["ratios"], dtype=float))

    df = pd.DataFrame(rows)

    return {
        "df": df,
        "ratio_distributions": ratio_distributions,
        "train_histories": [train_result["df_train"].copy()],
    }


# ============================================================
# 7) SUMMARIES
# ============================================================
def summarize_df(df: pd.DataFrame):
    if len(df) == 0:
        return {}

    final_row = df.iloc[-1]
    best_ratios = df["best_ratio"].to_numpy(dtype=float)
    zero_ratios = df["zero_ratio"].to_numpy(dtype=float)
    epochs = df["cumulative_epochs"].to_numpy(dtype=float)

    summary = {
        "method": final_row["method"],
        "graph_seed": int(final_row["graph_seed"]),
        "final_iteration": int(final_row["iteration"]),
        "final_epochs": int(final_row["cumulative_epochs"]),
        "final_best_ratio": float(final_row["best_ratio"]),
        "final_expected_ratio": float(final_row["expected_ratio"]),
        "final_zero_ratio": float(final_row["zero_ratio"]),
        "final_min_energy": float(final_row["min_energy"]),
        "final_expected_energy": float(final_row["expected_energy"]),
        "best_ratio_max_over_run": float(np.max(best_ratios)),
        "best_ratio_gain": float(best_ratios[-1] - best_ratios[0]),
        "auc_best_ratio": float(np.trapezoid(best_ratios, x=epochs)),
        "zero_ratio_monotonic_fraction": float(
            np.mean(np.diff(zero_ratios) >= -1e-12)
        ) if len(zero_ratios) > 1 else 1.0,
        "stopped_early": bool(final_row["stopped_early"]),
        "stop_iteration": final_row["stop_iteration"],
        "stop_reason": final_row["stop_reason"],
    }
    return summary


def attach_cfg_cols(df: pd.DataFrame, cfg: Config):
    out = df.copy()
    for k, v in asdict(cfg).items():
        out[k] = v
    return out


# ============================================================
# 8) SINGLE EXPERIMENT
# ============================================================
def run_all_methods_for_graph(cfg: Config, graph_seed: int, verbose=True, show_tqdm=True):
    instance = build_problem_instance(cfg, graph_seed)

    ndar = run_ndar(cfg, instance, graph_seed, verbose=verbose, show_tqdm=show_tqdm)
    fixed_total = run_fixed_h_total(cfg, instance, graph_seed, verbose=verbose, show_tqdm=show_tqdm)

    df_all = pd.concat(
        [
            attach_cfg_cols(ndar["df"], cfg),
            attach_cfg_cols(fixed_total["df"], cfg),
        ],
        ignore_index=True,
    )

    df_summary = pd.concat(
        [
            attach_cfg_cols(pd.DataFrame([summarize_df(ndar["df"])]), cfg),
            attach_cfg_cols(pd.DataFrame([summarize_df(fixed_total["df"])]), cfg),
        ],
        ignore_index=True,
    )

    return df_all, df_summary, {
        "instance": instance,
        "ndar": ndar,
        "fixed_total": fixed_total,
    }


# ============================================================
# 9) SWEEPS
# ============================================================
def run_one_factor_sweep(base_cfg: Config, sweep_name: str, sweep_values, graph_seeds, out_dir: Path, verbose=False, show_tqdm=True):
    all_runs = []
    all_summaries = []

    sweep_iter = sweep_values
    if show_tqdm:
        sweep_iter = tqdm(sweep_values, desc=f"Sweep: {sweep_name}", leave=True, dynamic_ncols=True)

    for value in sweep_iter:
        cfg_dict = asdict(base_cfg).copy()
        cfg_dict[sweep_name] = value
        cfg = Config(**cfg_dict)

        print("\n" + "#" * 100)
        print(f"SWEEP {sweep_name} = {value}")
        print("#" * 100, flush=True)

        summaries_this_value = []

        seed_iter = graph_seeds
        if show_tqdm:
            seed_iter = tqdm(graph_seeds, desc=f"{sweep_name}={value} | seeds", leave=False, dynamic_ncols=True)

        for graph_seed in seed_iter:
            print(f"\n--- STARTING graph_seed = {graph_seed} ---", flush=True)

            df_all, df_summary, _ = run_all_methods_for_graph(
                cfg,
                graph_seed,
                verbose=verbose,
                show_tqdm=show_tqdm,
            )

            df_all["sweep_name"] = sweep_name
            df_all["sweep_value"] = value

            df_summary["sweep_name"] = sweep_name
            df_summary["sweep_value"] = value

            all_runs.append(df_all)
            all_summaries.append(df_summary)
            summaries_this_value.append(df_summary)

            print_run_summary(
                df_summary_run=df_summary,
                sweep_name=sweep_name,
                sweep_value=value,
                graph_seed=graph_seed,
            )

            pd.concat(all_runs, ignore_index=True).to_csv(
                out_dir / f"all_runs_{sweep_name}_partial.csv",
                index=False
            )
            pd.concat(all_summaries, ignore_index=True).to_csv(
                out_dir / f"summary_{sweep_name}_partial.csv",
                index=False
            )

            print(
                f"Saved partial results for {sweep_name} = {value}, graph_seed = {graph_seed}",
                flush=True
            )

        df_summary_value = pd.concat(summaries_this_value, ignore_index=True)
        print_value_aggregate(
            df_summary_value=df_summary_value,
            sweep_name=sweep_name,
            sweep_value=value,
        )

    df_all_runs = pd.concat(all_runs, ignore_index=True)
    df_summary_runs = pd.concat(all_summaries, ignore_index=True)

    df_all_runs.to_csv(out_dir / f"all_runs_{sweep_name}.csv", index=False)
    df_summary_runs.to_csv(out_dir / f"summary_{sweep_name}.csv", index=False)

    return df_all_runs, df_summary_runs


# ============================================================
# 10) PLOTS
# ============================================================
def plot_metric_vs_sweep(df_summary, sweep_name, metric, out_path):
    plt.figure(figsize=(9, 6))

    for method, dfg in df_summary.groupby("method"):
        grouped = (
            dfg.groupby("sweep_value")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("sweep_value")
        )

        x = grouped["sweep_value"].to_numpy()
        y = grouped["mean"].to_numpy()
        s = grouped["std"].fillna(0.0).to_numpy()

        plt.plot(x, y, marker="o", linewidth=2, label=method)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel(sweep_name)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {sweep_name}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_learning_curves(df_all, sweep_name, sweep_value, out_path):
    dff = df_all[df_all["sweep_value"] == sweep_value].copy()

    plt.figure(figsize=(9, 6))

    for method, dfg in dff.groupby("method"):
        grouped = (
            dfg.groupby("cumulative_epochs")["best_ratio"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("cumulative_epochs")
        )

        x = grouped["cumulative_epochs"].to_numpy()
        y = grouped["mean"].to_numpy()
        s = grouped["std"].fillna(0.0).to_numpy()

        plt.plot(x, y, marker="o", linewidth=2, label=method)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Cumulative epochs")
    plt.ylabel("Best approximation ratio")
    plt.title(f"Best ratio curves | {sweep_name} = {sweep_value}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# 11) MAIN
# ============================================================
def main():
    base_cfg = Config(
        n_qubits=20,
        edge_prob=0.8,
        depth=10,
        shots=1000,
        seed=1924,
        K=6,
        n_epochs=10,
        adam_lr=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        spsa_c=0.15,
        spsa_gamma=0.101,
        warm_start_next_iter=True,
        stop_rule="paper",
        out_dir="benchmark_ndar_results",
    )

    out_dir = ensure_dir(Path(base_cfg.out_dir))

    graph_seeds = [11, 29, 47, 83, 101]

    with open(out_dir / "base_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(base_cfg), f, indent=2, ensure_ascii=False)

    sweep_plan = {
        "adam_lr": [0.01, 0.1, 0.2],
        #"depth": [4, 6, 8, 10, 12],
        #"n_qubits": [8, 10, 12, 14, 16],
        #"K": [2, 4, 6, 8],
        #"n_epochs": [20, 50, 100, 150],
        "shots": [32, 64, 100, 256, 512],
        #"stop_rule": ["none", "paper"],
    }

    meta_rows = []

    main_sweep_iter = tqdm(
        sweep_plan.items(),
        desc="All sweeps",
        leave=True,
        dynamic_ncols=True,
        total=len(sweep_plan),
    )

    for sweep_name, sweep_values in main_sweep_iter:
        df_all, df_summary = run_one_factor_sweep(
            base_cfg=base_cfg,
            sweep_name=sweep_name,
            sweep_values=sweep_values,
            graph_seeds=graph_seeds,
            out_dir=out_dir,
            verbose=False,
            show_tqdm=True,
        )

        plot_metric_vs_sweep(
            df_summary,
            sweep_name=sweep_name,
            metric="final_best_ratio",
            out_path=out_dir / f"{sweep_name}_final_best_ratio.png",
        )
        plot_metric_vs_sweep(
            df_summary,
            sweep_name=sweep_name,
            metric="final_expected_ratio",
            out_path=out_dir / f"{sweep_name}_final_expected_ratio.png",
        )
        plot_metric_vs_sweep(
            df_summary,
            sweep_name=sweep_name,
            metric="best_ratio_gain",
            out_path=out_dir / f"{sweep_name}_best_ratio_gain.png",
        )

        for val in sweep_values:
            safe_val = str(val).replace(".", "_")
            plot_learning_curves(
                df_all,
                sweep_name=sweep_name,
                sweep_value=val,
                out_path=out_dir / f"{sweep_name}_curves_{safe_val}.png",
            )

        meta_rows.append(
            {
                "sweep_name": sweep_name,
                "values_tested": json.dumps(list(sweep_values)),
                "n_rows_all": int(len(df_all)),
                "n_rows_summary": int(len(df_summary)),
            }
        )

        pd.DataFrame(meta_rows).to_csv(out_dir / "meta_summary_partial.csv", index=False)

    pd.DataFrame(meta_rows).to_csv(out_dir / "meta_summary.csv", index=False)

    print("\nDone.")
    print(f"Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()