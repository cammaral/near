import numpy as np
from ndar.utils import evaluate_theta_expected_energy
import pandas as pd

def optimize(
    label,
    theta_init,
    transpiled_measured_template,
    ansatz_params,
    simulator,
    shots,
    H_constant_term,
    H_zz_terms,
    out_dir_sub,
    seed_base,
    n_epochs,
    lr,
    beta1,
    beta2,
    eps,
    spsa_c,
    spsa_gamma,
):
    #out_dir_sub.mkdir(exist_ok=True, parents=True)

    rng_local = np.random.default_rng(seed_base)

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

        print(
            f"{label} | epoch={epoch:03d} | "
            f"loss={train_loss:.6f} | best={best_loss:.6f} | grad_norm={grad_norm:.6f}"
        )

    df_train = pd.DataFrame(history_rows)
    #df_train.to_csv(out_dir_sub / "training_history.csv", index=False)
    #np.save(out_dir_sub / "theta_history.npy", np.array(theta_history, dtype=float))
    #np.save(out_dir_sub / "best_theta.npy", best_theta)
    #np.save(out_dir_sub / "final_theta.npy", theta)

    #plt.figure(figsize=(8, 5))
    #plt.plot(df_train["epoch"], df_train["loss"], marker="o", markersize=2)
    #plt.xlabel("Epoch")
    #plt.ylabel("Training loss")
    #plt.title(f"Training loss - {label}")
    #plt.tight_layout()
    #plt.savefig(out_dir_sub / "training_loss.png", dpi=200)
    #plt.show()

    return {
        "theta_final": theta.copy(),
        "theta_best": best_theta.copy(),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "df_train": df_train.copy(),
    }
