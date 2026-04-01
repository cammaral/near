

baseline_dir = out_dir / "plain_adam_baseline"
baseline_dir.mkdir(exist_ok=True)

theta_plain = theta_init_shared.copy()

baseline_training = adam_optimize(
    label="PLAIN_ADAM",
    theta_init=theta_plain,
    transpiled_measured_template=transpiled_measured_template,
    ansatz_params=ansatz_params,
    simulator=simulator,
    shots=shots,
    H_constant_term=H_constant_term_initial,
    H_zz_terms=H_zz_terms_initial,
    out_dir_sub=baseline_dir,
    seed_base=seed + 999999,
    n_epochs=total_baseline_epochs,
    lr=adam_lr,
    beta1=adam_beta1,
    beta2=adam_beta2,
    eps=adam_eps,
    spsa_c=spsa_c,
    spsa_gamma=spsa_gamma,
)

theta_history_plain = np.load(baseline_dir / "theta_history.npy")

baseline_rows = []
baseline_ratio_distributions = []

for block_idx in range(K):
    epoch_checkpoint = (block_idx + 1) * n_epochs
    theta_checkpoint = theta_history_plain[epoch_checkpoint]

    chk_dir = baseline_dir / f"checkpoint_{epoch_checkpoint:03d}"
    chk_dir.mkdir(exist_ok=True)

    sample_result = sample_theta_full(
        theta=theta_checkpoint,
        iter_label=f"PLAIN_epoch_{epoch_checkpoint}",
        transpiled_measured_template=transpiled_measured_template,
        ansatz_params=ansatz_params,
        simulator=simulator,
        shots=shots,
        seed_simulator=seed + 800000 + block_idx,
        edges=edges,
        H_constant_term=H_constant_term_initial,
        H_zz_terms=H_zz_terms_initial,
        out_dir_sub=chk_dir,
    )

    baseline_rows.append(
        {
            "block": block_idx,
            "cumulative_epochs": epoch_checkpoint,
            "best_bitstring": sample_result["best_bitstring"],
            "min_energy": sample_result["min_energy"],
            "expected_energy": sample_result["expected_energy"],
            "best_cut": sample_result["best_cut"],
            "best_ratio": sample_result["best_ratio"],
            "expected_ratio": sample_result["expected_ratio"],
            "zero_ratio": sample_result["zero_ratio"],
        }
    )

    baseline_ratio_distributions.append(sample_result["ratios"])

df_baseline = pd.DataFrame(baseline_rows)
df_baseline.to_csv(baseline_dir / "baseline_summary.csv", index=False)