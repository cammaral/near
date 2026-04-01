from ndar.utils import bind_and_run_circuit, qiskit_to_q0_first
from quantum.hamiltonian import hamiltonian_value_for_bitstring
from qubo.MaxCut import maxcut_value

import numpy as np
import pandas as pd

def sample_theta_full(n_qubits,
    E_GS_exact,
    theta,
    iter_label,
    transpiled_measured_template,
    ansatz_params,
    simulator,
    shots,
    seed_simulator,
    edges,
    H_constant_term,
    H_zz_terms,
    out_dir_sub,
):
    #out_dir_sub.mkdir(exist_ok=True, parents=True)

    result = bind_and_run_circuit(
        theta=theta,
        transpiled_measured_template=transpiled_measured_template,
        ansatz_params=ansatz_params,
        simulator=simulator,
        shots=shots,
        seed_simulator=seed_simulator,
    )

    memory = result.get_memory()
    counts_raw = result.get_counts()

    rows = []
    energies = []
    ratios = []

    for k, raw_bs in enumerate(memory):
        bs_q0 = qiskit_to_q0_first(raw_bs)
        energy = hamiltonian_value_for_bitstring(bs_q0, H_constant_term, H_zz_terms)
        cut = maxcut_value(bs_q0, edges)
        ratio = energy / E_GS_exact ########

        rows.append(
            {
                "label": iter_label,
                "shot": k,
                "bitstring_qiskit": raw_bs.replace(" ", ""),
                "bitstring_q0_first": bs_q0,
                "hamiltonian_value": energy,
                "cut_value": cut,
                "approx_ratio": ratio,
            }
        )
        energies.append(energy)
        ratios.append(ratio)

    df_shots = pd.DataFrame(rows)
    #df_shots.to_csv(out_dir_sub / "shots_individuais.csv", index=False)

    #with open(out_dir_sub / "counts_raw.json", "w", encoding="utf-8") as f:
    #    json.dump(counts_raw, f, indent=2, ensure_ascii=False)

    min_energy = float(np.min(energies))
    expected_energy = float(np.mean(energies))
    best_idx = int(np.argmin(energies))
    best_bitstring = df_shots.iloc[best_idx]["bitstring_q0_first"]
    best_cut = float(df_shots.iloc[best_idx]["cut_value"])

    best_ratio = min_energy / E_GS_exact
    expected_ratio = expected_energy / E_GS_exact

    zero_bitstring = "0" * n_qubits
    zero_energy = hamiltonian_value_for_bitstring(zero_bitstring, H_constant_term, H_zz_terms)
    zero_ratio = zero_energy / E_GS_exact
    bitstrings_q0_first = df_shots["bitstring_q0_first"].tolist()
    bitstrings_qiskit = df_shots["bitstring_qiskit"].tolist()
    info = {
        "label": iter_label,
        "best_bitstring": best_bitstring,
        "min_energy": min_energy,
        "expected_energy": expected_energy,
        "best_cut": best_cut,
        "best_ratio": float(best_ratio),
        "expected_ratio": float(expected_ratio),
        "zero_ratio": float(zero_ratio),
    }

    #with open(out_dir_sub / "sample_summary.json", "w", encoding="utf-8") as f:
    #    json.dump(info, f, indent=2, ensure_ascii=False)

    return {
    "df_shots": df_shots,
    "bitstrings_q0_first": bitstrings_q0_first,
    "bitstrings_qiskit": bitstrings_qiskit,
    "best_bitstring": best_bitstring,
    "min_energy": min_energy,
    "expected_energy": expected_energy,
    "best_cut": best_cut,
    "best_ratio": float(best_ratio),
    "expected_ratio": float(expected_ratio),
    "zero_ratio": float(zero_ratio),
    "ratios": np.array(ratios, dtype=float),
}