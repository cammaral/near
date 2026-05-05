
import os
import json
import pickle
from pathlib import Path
from itertools import combinations

import numpy as np
import rustworkx as rx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import Estimator as AerEstimator


# ============================================
# Utilidades
# ============================================

def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False)


def save_pickle(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ============================================
# Cargar grafo
# ============================================

def load_graph_from_file(path):
    edges = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # primera línea: metadata
    first_line = lines[0].strip().split()
    num_nodes = int(first_line[0])

    edge_lines = lines[1:]

    for line in edge_lines:
        parts = line.strip().split()

        if len(parts) < 2:
            continue  # línea vacía o rara

        i = int(parts[0])
        j = int(parts[1])

        # convertir a índice base 0
        i -= 1
        j -= 1

        edges.append((i, j))

    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from_no_data(edges)

    return graph, num_nodes, edges


# ============================================
# Observables
# ============================================

def generate_all_pauli_pairs(n_qubits, k=2):
    obs_list = []
    for combo in combinations(range(n_qubits), k):
        for pauli in ["X", "Y", "Z"]:
            paulis = ["I"] * n_qubits
            paulis[combo[0]] = pauli
            paulis[combo[1]] = pauli
            pauli_str = "".join(paulis)[::-1]
            obs_list.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
    return obs_list


# ============================================
# GW aproximado
# ============================================

def gw_sdp(graph, n, n_random=50, seed=0):
    rng = np.random.RandomState(seed)

    W = np.zeros((n, n))
    for i, j in graph.edge_list():
        W[i, j] = 1
        W[j, i] = 1

    _, eigvecs = np.linalg.eigh(W)
    X = eigvecs[:, -n:]

    best_cut = None
    best_val = -np.inf

    for _ in range(n_random):
        r = rng.randn(X.shape[1])
        r /= np.linalg.norm(r)
        x = np.sign(X @ r)

        val = sum(1 for i, j in graph.edge_list() if x[i] != x[j])

        if val > best_val:
            best_val = val
            best_cut = x

    return {i: (0 if best_cut[i] < 0 else 1) for i in range(n)}


def regularize_c(c, epsilon=0.2):
    return {
        i: epsilon if v < epsilon else (1 - epsilon if v > 1 - epsilon else v)
        for i, v in c.items()
    }


# ============================================
# Métrica de corte
# ============================================

def compute_cut(graph, node_exp):
    return sum(
        1 for i, j in graph.edge_list()
        if np.sign(node_exp[i]) != np.sign(node_exp[j])
    )


# ============================================
# SPSA + Adam
# ============================================

def spsa_adam(loss_func, theta, n_epochs=200, lr=0.05, c=0.1, seed=0, desc="SPSA"):
    rng = np.random.RandomState(seed)

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    history = []

    progress_bar = tqdm(range(1, n_epochs + 1), desc=desc, leave=False)

    for t in progress_bar:
        delta = rng.choice([-1, 1], size=theta.shape)

        loss_plus = loss_func(theta + c * delta)
        loss_minus = loss_func(theta - c * delta)

        g = (loss_plus - loss_minus) / (2 * c * delta)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

        loss_val = loss_func(theta)
        history.append(loss_val)

        progress_bar.set_postfix(loss=f"{loss_val:.6f}")

    return history, theta


# ============================================
# Optimización
# ============================================

def run_optimization(
    graph,
    observables,
    node_map,
    num_qubits,
    alpha,
    beta,
    epsilon,
    maxiter,
    shots=None,
    seed=0,
):
    ansatz = efficient_su2(num_qubits, ["ry", "rz"], reps=2)
    rng = np.random.RandomState(seed)
    theta0 = rng.rand(ansatz.num_parameters)

    c = gw_sdp(graph, len(graph.nodes()), seed=seed)
    c_hat = regularize_c(c, epsilon)

    if shots is None:
        estimator = StatevectorEstimator()

        def eval_expectations(params):
            job = estimator.run([(ansatz, observables, params)])
            return job.result()[0].data.evs

    else:
        estimator = AerEstimator(run_options={"shots": shots, "method": "statevector"})

        def eval_expectations(params):
            circuits = [ansatz] * len(observables)
            param_values = [params] * len(observables)
            job = estimator.run(
                circuits=circuits,
                observables=observables,
                parameter_values=param_values,
            )
            return job.result().values

    history = []
    cut_history = []

    def loss_func(params):
        evs = eval_expectations(params)
        node_exp = {i: evs[idx] for i, idx in node_map.items()}

        cut_history.append(compute_cut(graph, node_exp))

        loss = 0.0
        for i, j in graph.edge_list():
            w = 1 + abs(c_hat[i] - c_hat[j])
            # w = 1  # Descomentar para tratar el problema sin warm-start
            loss += w * np.tanh(alpha * node_exp[i]) * np.tanh(alpha * node_exp[j])

        reg = np.mean([np.tanh(alpha * node_exp[i]) ** 2 for i in node_exp])
        reg = beta * (len(graph.edges()) / 2) * (reg ** 2)

        total = loss + reg
        history.append(total)
        return total

    history, theta = spsa_adam(
        loss_func,
        theta0,
        n_epochs=maxiter,
        seed=seed,
        desc=f"SPSA (shots={shots if shots is not None else 'statevector'}, seed={seed})",
    )

    # evaluación final
    evs = eval_expectations(theta)
    node_exp = {i: evs[idx] for i, idx in node_map.items()}
    final_cut = compute_cut(graph, node_exp)

    return {
        "history": history,
        "cut_history": cut_history,
        "final_cut": final_cut,
        "theta": theta,
        "final_expectations": node_exp,
        "warm_start_c": c,
        "warm_start_c_hat": c_hat,
    }


# ============================================
# Graficar resultados
# ============================================

def plot_cut_vs_iters(all_cut_histories, output_dir):
    plt.figure(figsize=(8, 5))

    for name, cut_histories in all_cut_histories.items():
        min_len = min(len(h) for h in cut_histories)
        aligned = np.array([h[:min_len] for h in cut_histories])

        mean_curve = aligned.mean(axis=0)
        std_curve = aligned.std(axis=0)

        x = np.arange(min_len)

        plt.plot(x, mean_curve, label=name)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Iterations")
    plt.ylabel("Cut value")
    plt.title("########## CUT vs ITERS ##########")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "cut_vs_iters_com_warm_start_correccion.png", dpi=150)
    plt.close()


def plot_cut_histogram(all_final_cuts, output_dir):
    fig, axes = plt.subplots(1, len(all_final_cuts), figsize=(12, 4))

    if len(all_final_cuts) == 1:
        axes = [axes]

    for ax, (name, cuts) in zip(axes, all_final_cuts.items()):
        ax.hist(cuts, bins=10)
        ax.set_title(name)
        ax.set_xlabel("Cut")
        ax.set_ylabel("Freq")

    plt.suptitle("########## CUT vs NUM_RUNS ##########")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "cut_vs_num_runs_com_warm_start_correccion.png", dpi=150)
    plt.close()


def plot_loss_curves(all_histories, output_dir):
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = {
        "statevector": "blue",
        "100 shots": "green",
        "1500 shots": "red",
    }

    # Gráfica 1: todas las curvas individuales
    plt.figure(figsize=(12, 6))
    for name, histories in all_histories.items():
        for hist in histories:
            plt.plot(hist, color=colors.get(name, "black"), alpha=0.3, linewidth=0.8)
    plt.xlabel("Iteración")
    plt.ylabel("Pérdida")
    plt.title("Evolución de la pérdida en corridas por configuración")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors.get(name, "black"), lw=2, label=name)
        for name in all_histories
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "todas_corridas.png", dpi=150)
    plt.close()

    # Gráfica 2: promedios por grupo con bandas de desviación estándar
    plt.figure(figsize=(12, 6))
    for name, histories in all_histories.items():
        max_len = max(len(h) for h in histories)
        padded = [h + [np.nan] * (max_len - len(h)) for h in histories]
        arr = np.array(padded)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        iterations = range(1, max_len + 1)
        plt.plot(iterations, mean, color=colors.get(name, "black"), linewidth=2, label=name)
        plt.fill_between(
            iterations,
            mean - std,
            mean + std,
            color=colors.get(name, "black"),
            alpha=0.2,
        )

    plt.xlabel("Iteración")
    plt.ylabel("Pérdida promedio")
    plt.title("Promedio de la pérdida por configuración utilizando warm-start")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "promedios_SLSQP_com_warm_start_correccion.png", dpi=150)
    plt.close()


# ============================================
# Main
# ============================================

def main():
    # Parámetros
    k = 3
    beta = 0.5
    num_qubits = 13
    alpha = 1.5 * num_qubits
    maxiter = 300
    epsilon = 0.2

    graph_path = "G1.txt"
    output_dir = Path("results_G1_SPSA_ADAM_WarmStart")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grafo
    graph, num_nodes, edges = load_graph_from_file(graph_path)
    print(f"Grafo cargado: {num_nodes} nodos, {len(edges)} aristas")

    # Observables
    observables = generate_all_pauli_pairs(num_qubits, k)[:num_nodes]
    node_obs_index = {i: i for i in range(num_nodes)}

    # Guardar configuración inicial
    config_data = {
        "k": k,
        "beta": beta,
        "num_qubits": num_qubits,
        "alpha": alpha,
        "maxiter": maxiter,
        "epsilon": epsilon,
        "graph_path": graph_path,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "configs": [
            ("statevector", None, 10),
            ("100 shots", 100, 10),
            ("1500 shots", 1500, 10),
        ],
    }
    save_json(config_data, output_dir / "config.json")
    save_json({"edges": edges}, output_dir / "graph_data.json")

    # Experimentos
    configs = [
        ("statevector", None, 10),
        ("100 shots", 100, 10),
        ("1500 shots", 1500, 10),
    ]

    all_histories = {}
    all_cut_histories = {}
    all_final_cuts = {}
    all_run_results = {}

    for name, shots, n_runs in configs:
        print(f"\n=== {name} ===")

        histories = []
        cut_histories = []
        cuts = []
        run_results = []

        for run in tqdm(range(n_runs), desc=f"Runs - {name}"):
            result = run_optimization(
                graph,
                observables,
                node_obs_index,
                num_qubits=num_qubits,
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                maxiter=maxiter,
                shots=shots,
                seed=run,
            )

            hist = result["history"]
            cut_hist = result["cut_history"]
            final_cut = result["final_cut"]

            histories.append(hist)
            cut_histories.append(cut_hist)
            cuts.append(final_cut)
            run_results.append(result)

            # Guardar cada corrida
            run_dir = output_dir / name.replace(" ", "_") / f"run_{run:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            np.save(run_dir / "theta_final.npy", result["theta"])
            save_json(
                {
                    "final_cut": final_cut,
                    "history": hist,
                    "cut_history": cut_hist,
                    "final_expectations": result["final_expectations"],
                    "warm_start_c": result["warm_start_c"],
                    "warm_start_c_hat": result["warm_start_c_hat"],
                },
                run_dir / "run_results.json",
            )

        all_histories[name] = histories
        all_cut_histories[name] = cut_histories
        all_final_cuts[name] = cuts
        all_run_results[name] = run_results

    # Guardar resultados agregados
    save_json({"all_final_cuts": all_final_cuts}, output_dir / "all_final_cuts.json")
    save_json({"all_histories": all_histories}, output_dir / "all_histories.json")
    save_json({"all_cut_histories": all_cut_histories}, output_dir / "all_cut_histories.json")
    save_pickle(all_run_results, output_dir / "all_run_results.pkl")

    summary = {}
    for name, cuts in all_final_cuts.items():
        summary[name] = {
            "n_runs": len(cuts),
            "mean_final_cut": float(np.mean(cuts)),
            "std_final_cut": float(np.std(cuts)),
            "min_final_cut": float(np.min(cuts)),
            "max_final_cut": float(np.max(cuts)),
        }
    save_json(summary, output_dir / "summary.json")

    # Plots
    plot_cut_vs_iters(all_cut_histories, output_dir)
    plot_cut_histogram(all_final_cuts, output_dir)
    plot_loss_curves(all_histories, output_dir)

    print("DONE")


if __name__ == "__main__":
    main()

