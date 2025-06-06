import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import sys
import os



def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

# Specify the GAMS system directory (Update this path according to your GAMS installation)
root_dir = setting_directory(1)  # Example path for Windows

print(root_dir)

from deepkoopformer.train import set_seed, train
from deepkoopformer.backbone import SimpleLSTMForecaster, Koopformer_PatchTST, Koopformer_Informer, Koopformer_Autoformer
from deepkoopformer.dataset import build_dataset
from deepkoopformer.plot import plot_training, plot_per_feature
from deepkoopformer.plot import save_results



# --------------------------------------------------------------------------- #
# 12)  Main loop                                                              #
# --------------------------------------------------------------------------- #
def main(file="pressure_surface_2020.npy", epochs=200, seq_len=120,
         patch_lens=None, horizons=None, d_models=None, indices=None,
         save_dir="results"):

    patch_lens = patch_lens or [120]  # fixed length as in your original
    horizons   = horizons   or [10, 15, 20, 25, 30]
    d_models   = d_models   or [8, 16, 24, 32, 40, 48]
    indices    = indices    or [0, 5000, 10000, 20000]
    
    

    save_dir   = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.csv"
    pd.DataFrame(columns=["Model", "PatchLen", "Horizon", "d_model", "MSE", "MAE"]
        ).to_csv(metrics_file, index=False)

    raw = np.load(file)

    for patch_len in patch_lens:
        for horizon in horizons:
            for d_model in d_models:
                print(f"\n>>> patch_len={patch_len}, horizon={horizon}, d_model={d_model}\n")
                x_in, x_out, _ = build_dataset(raw[:400,:], indices,
                                               patch_len, horizon)
                F, H = len(indices), horizon
                res_plot = {}
                x_np = x_out.numpy()

                # ----- LSTM -----
                print("== LSTM ==")
                lstm = SimpleLSTMForecaster(F, d_model, 2, horizon)
                pred, loss, eig = train(lstm, x_in, x_out,
                                        epochs=epochs, koop_attr=None, lyap_weight=0)
                res_plot["LSTM"] = {"pred": pred, "loss": loss,
                                    "eig": eig, "x_out": x_np}
                prefix = save_dir / f"LSTM_{patch_len}_{horizon}_d{d_model}"
                save_results(res_plot["LSTM"], prefix, metrics_file,
                             patch_len, horizon, d_model)
                torch.save(lstm.cpu().state_dict(), f"{prefix}.pt")

                # ----- PatchTST -----
                print("== PatchTST ==")
                m = Koopformer_PatchTST(F, patch_len, horizon,
                                        patch_len=patch_len, d_model=d_model)
                pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
                res_plot["Koop-PatchTST"] = {"pred": pred, "loss": loss,
                                             "eig": eig, "x_out": x_np}
                prefix = save_dir / f"PatchTST_{patch_len}_{horizon}_d{d_model}"
                save_results(res_plot["Koop-PatchTST"], prefix, metrics_file,
                             patch_len, horizon, d_model)
                torch.save(m.cpu().state_dict(), f"{prefix}.pt")

                # ----- Autoformer -----
                print("== Autoformer ==")
                m = Koopformer_Autoformer(F, patch_len, horizon,
                                          patch_len=patch_len, d_model=d_model)
                pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
                res_plot["Koop-Autoformer"] = {"pred": pred, "loss": loss,
                                               "eig": eig, "x_out": x_np}
                prefix = save_dir / f"Autoformer_{patch_len}_{horizon}_d{d_model}"
                save_results(res_plot["Koop-Autoformer"], prefix, metrics_file,
                             patch_len, horizon, d_model)
                torch.save(m.cpu().state_dict(), f"{prefix}.pt")

                # ----- Informer -----
                print("== Informer ==")
                m = Koopformer_Informer(F, patch_len, horizon,
                                        patch_len=patch_len, d_model=d_model)
                pred, loss, eig = train(m, x_in, x_out, epochs=epochs, koop_attr="koop")
                res_plot["Koop-Informer"] = {"pred": pred, "loss": loss,
                                             "eig": eig, "x_out": x_np}
                prefix = save_dir / f"Informer_{patch_len}_{horizon}_d{d_model}"
                save_results(res_plot["Koop-Informer"], prefix, metrics_file,
                             patch_len, horizon, d_model)
                torch.save(m.cpu().state_dict(), f"{prefix}.pt")

                # ----- Figures -----
                plot_training(res_plot,
                    save_dir / f"train_patch{patch_len}_h{horizon}_d{d_model}")
                plot_per_feature(x_np,
                    {n: r["pred"] for n, r in res_plot.items()},
                    F, H, save_dir / f"feat_patch{patch_len}_h{horizon}_d{d_model}")

                # Console summary
                print("\nFinal metrics (train set):")
                for n, r in res_plot.items():
                    e = r["x_out"] - r["pred"]
                    print(f"{n:<15s} MSE {(e**2).mean():.4e} "
                          f"MAE {np.abs(e).mean():.4e}")


# --------------------------------------------------------------------------- #
# 13)  CLI                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Koopformer-PRO benchmark (HPC-ready, multi-d_model)")
    p.add_argument("--file", type=str, default="./pressure_surface_2020.npy")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--save_dir", type=str, default="results")
    p.add_argument("--d_models", type=str, default="8,16,24,32,40,48")
    args = p.parse_args()
    d_models = [int(x) for x in args.d_models.split(",")]
    main(file=args.file, epochs=args.epochs,
         seq_len=args.seq_len, save_dir=args.save_dir,
         d_models=d_models)




