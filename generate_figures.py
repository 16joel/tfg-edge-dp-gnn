#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
generate_figures.py
===================
Reprodueix TOTES les figures de l'informe a partir dels CSV de `results/`.

Figures generades (mateixos noms que els \includegraphics de l'informe):
  - figA_utility_multidataset_vertical.pdf
  - figB_s1_gap_multidataset.pdf
  - fig2_attack_auc.pdf
  - fig3_privacy_utility_map.pdf
  - fig4_tradeoff_window.pdf
  - fig5_operating_point.pdf

Ús:
  python generate_figures.py                 # autodetecta results/ i escriu a figures_generated/
  python generate_figures.py --out-dir figs  # carpeta de sortida personalitzada
  python generate_figures.py --results-dir results

Només depèn de numpy i matplotlib (ja a requirements.txt).
"""
import argparse, csv, os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DATASETS = ("Cora", "Citeseer", "Pubmed")
SCORES = ["common_neighbors", "jaccard", "adamic_adar",
          "resource_allocation", "preferential_attachment", "embedding_cosine"]

# Colors exactes extrets dels PDF originals
C_S1, C_S2, C_S3 = "#e8705a", "#1c9a8f", "#1b2a4a"     # escenaris (figA/figB)
C_BASE = "#8a94a6"; C_TITLE = "#33415c"; GRID = "#e0e4ea"
A_EMB, A_PA, A_HEUR = "#7838a0", "#e08028", "#286888"  # fig2
F4_S1, F4_S2, F4_AUC = "#e1812c", "#2b6e8f", "#7b3fa0" # fig4
J_U, J_P, J_J = "#286888", "#18a898", "#e08028"        # fig5


# ===========================================================================
# CÀRREGA DE DADES
# ===========================================================================
def _latest(results_dir, prefix):
    cands = sorted(glob.glob(os.path.join(results_dir, prefix + "_*")))
    if not cands:
        raise FileNotFoundError(f"No s'ha trobat cap carpeta '{prefix}_*' a {results_dir}")
    return cands[-1]

def load_utility(results_dir, dataset):
    """Retorna (eps_list, U, baseline_mean) on U[meth][sc][eps]=(mean,std)."""
    d = _latest(results_dir, dataset)
    det = os.path.join(d, "detailed_results")
    # baseline
    bvals = [float(r["Test Accuracy"]) for r in csv.DictReader(open(os.path.join(det, "baseline.csv")))]
    base = float(np.mean(bvals))
    U = {m: {s: {} for s in ("s1", "s2", "s3")} for m in ("rr_dp", "laplace_dp")}
    eps_set = set()
    for sub in glob.glob(os.path.join(det, "epsilon_*")):
        e = float(os.path.basename(sub).split("_")[1])
        if e < 0.1:            # excloem 0.01 (les figures comencen a 0,1)
            continue
        eps_set.add(e)
        for m in ("rr_dp", "laplace_dp"):
            for s in ("s1", "s2", "s3"):
                f = os.path.join(sub, f"{m}_{s}.csv")
                if not os.path.exists(f):
                    continue
                v = [float(r["Test Accuracy"]) for r in csv.DictReader(open(f))]
                U[m][s][e] = (float(np.mean(v)),
                              float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    return sorted(eps_set), U, base

def load_attack(results_dir, dataset):
    """Retorna A[config][score]=(auc,ap); config='base' o (mech,eps)."""
    d = _latest(results_dir, "InferenceAttack_" + dataset)
    f = os.path.join(d, "summary.txt")
    cur, mech, A = None, None, {}
    for line in open(f, encoding="utf-8"):
        m = re.match(r"\s*EPSILON = ([\d.]+)", line)
        if m:
            cur = float(m.group(1)); continue
        if re.match(r"\s*Baseline \(no DP\)", line):
            mech = "base"; continue
        if re.match(r"\s{2}RR\s", line):
            mech = "RR"; continue
        if re.match(r"\s{2}Laplace\s", line):
            mech = "Lap"; continue
        m = re.match(r"\s*([a-z_]+)\s+([0-9.]+)±[0-9.]+\s+([0-9.]+)±", line)
        if m and m.group(1) in SCORES:
            key = "base" if mech == "base" else (mech, cur)
            A.setdefault(key, {})[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return A

def eps_label(e):
    return ("%g" % e).replace(".", ",")


# ===========================================================================
# FIGURA A — utilitat per escenari, 3 conjunts (vertical)
# ===========================================================================
def fig_A(out, data):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
    fig, axes = plt.subplots(3, 1, figsize=(4.1, 9.3), sharex=True)
    for ax, ds in zip(axes, DATASETS):
        eps, U, base = data[ds]["util"]
        x = np.array(eps)
        for sc, col in (("s1", C_S1), ("s2", C_S2), ("s3", C_S3)):
            mean = np.array([U["rr_dp"][sc][e][0] for e in eps])
            std = np.array([U["rr_dp"][sc][e][1] for e in eps])
            ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.18, lw=0)
            ax.plot(x, mean, "-o", color=col, ms=4, lw=1.6)
        ax.axhline(base, ls="--", color=C_BASE, lw=1.3)
        ax.text(15.2, base, f"{base:.2f}".replace(".", ","), color=C_TITLE,
                fontsize=9, va="center", ha="left", fontweight="bold")
        ax.set_title(ds, color=C_TITLE, fontweight="bold", fontsize=13)
        ax.set_ylabel("Precisió de test")
        ax.set_ylim(0.30, 0.95); ax.set_xlim(-0.5, 16.5)
        ax.set_xticks([0, 3, 6, 9, 12, 15])
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[-1].set_xlabel(r"$\varepsilon$ (pressupost de privadesa)")
    handles = [plt.Line2D([], [], color=C_S1, marker="o", label="S1 · tot privat"),
               plt.Line2D([], [], color=C_S2, marker="o", label="S2 · inferència real"),
               plt.Line2D([], [], color=C_S3, marker="o", label="S3 · selecció real"),
               plt.Line2D([], [], color=C_BASE, ls="--", label="Baseline (per conjunt)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out, "figA_utility_multidataset_vertical.pdf"))
    plt.close(fig)


# ===========================================================================
# FIGURA B — dissociació S1->S2 a eps=1 (barres)
# ===========================================================================
def fig_B(out, data):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    w = 0.38
    for i, ds in enumerate(DATASETS):
        eps, U, base = data[ds]["util"]
        s1 = U["rr_dp"]["s1"][1.0][0]; s2 = U["rr_dp"]["s2"][1.0][0]
        ax.bar(i - w/2, s1, w, color=C_S1, zorder=3)
        ax.bar(i + w/2, s2, w, color=C_S2, zorder=3)
        for xx, val in ((i - w/2, s1), (i + w/2, s2)):
            ax.text(xx, val + 0.008, f"{val:.2f}".replace(".", ","),
                    ha="center", va="bottom", fontweight="bold", color=C_S3, fontsize=11)
        ax.plot([i - w, i + w], [base, base], ls="--", color=C_S3, lw=1.6, zorder=2)
        ax.text(i + w + 0.02, base, "baseline\n" + f"{base:.2f}".replace(".", ","),
                color=C_TITLE, fontsize=9, va="center")
        ax.annotate("", xy=(i + w/2, s2 - 0.01), xytext=(i - w/2, s1 + 0.01),
                    arrowprops=dict(arrowstyle="->", ls=":", color=C_BASE, lw=1.4))
        ax.text(i, (s1 + s2)/2, f"+{round((s2 - s1)*100)} pts", color=C_BASE,
                style="italic", fontsize=10, ha="center")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, fontweight="bold", color=C_TITLE)
    ax.set_ylabel("Precisió de test"); ax.set_ylim(0.30, 1.0)
    ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(handles=[plt.matplotlib.patches.Patch(color=C_S1, label="S1 · tot privat"),
                       plt.matplotlib.patches.Patch(color=C_S2, label="S2 · inferència real")],
              loc="upper right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figB_s1_gap_multidataset.pdf"))
    plt.close(fig)


# ===========================================================================
# FIGURA 2 — AUC de l'atac vs eps (Cora, RR)
# ===========================================================================
def fig_2(out, data):
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    A = data["Cora"]["atk"]
    eps = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    xi = np.arange(len(eps)); labels = [eps_label(e) for e in eps]
    emb = [A[("RR", e)]["embedding_cosine"][0] for e in eps]
    pa = [A[("RR", e)]["preferential_attachment"][0] for e in eps]
    heur = [np.mean([A[("RR", e)][s][0] for s in SCORES[:4]]) for e in eps]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(xi, emb, "-^", color=A_EMB, ms=5, lw=1.7, label="Similitud d'embeddings")
    ax.plot(xi, pa, "-D", color=A_PA, ms=5, lw=1.7, label="Preferential Attachment")
    ax.plot(xi, heur, "-s", color=A_HEUR, ms=5, lw=1.7, label="Heurístiques de veïns (CN/Jac/AA/RA)")
    ax.axhline(0.5, ls="--", color="#8a8a8a", lw=1.2, label="Atzar (protecció perfecta, 0,5)")
    ax.set_xlabel(r"Pressupost de privadesa $\varepsilon$")
    ax.set_ylabel("AUC-ROC de l'atac")
    ax.set_xticks(xi); ax.set_xticklabels(labels)
    ax.set_ylim(0.47, 1.0); ax.set_xlim(-0.4, len(eps) - 0.6)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_attack_auc.pdf"), bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# FIGURA 3 — mapa privadesa-utilitat (Cora, RR)
# ===========================================================================
def fig_3(out, data):
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    eps_u, U, base_u = data["Cora"]["util"]; A = data["Cora"]["atk"]
    eps = [e for e in eps_u if e >= 0.1]
    X = np.array([A[("RR", e)]["embedding_cosine"][0] for e in eps])
    Y = np.array([U["rr_dp"]["s2"][e][0] for e in eps])
    C = np.log10(np.array(eps))
    bx = A["base"]["embedding_cosine"][0]; by = base_u
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    norm = Normalize(vmin=-1, vmax=1)
    sc = ax.scatter(X, Y, c=C, cmap="viridis", norm=norm, s=240,
                    edgecolor="white", linewidth=0.8, zorder=3)
    for e, x, y in zip(eps, X, Y):
        if e > 10:
            continue
        ax.annotate(eps_label(e), (x, y), ha="center", va="center",
                    fontsize=7.5, color="white", zorder=4)
    ax.scatter([bx], [by], marker="*", s=420, color="#d62728",
               edgecolor="white", linewidth=0.6, zorder=5, label="Baseline sense DP")
    ax.text(0.03, 0.96, "zona ideal\n(alta utilitat, baixa filtració)",
            transform=ax.transAxes, color="#0d8a7a", style="italic",
            fontsize=10.5, va="top", ha="left")
    ax.set_xlabel("AUC-ROC de l’atac (embedding cosine) $\\rightarrow$ menys privat")
    ax.set_ylabel("Precisió de test (escenari S2)")
    ax.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)
    cb = fig.colorbar(sc, ax=ax, ticks=np.arange(-1, 1.01, 0.25))
    cb.set_label("$\\log_{10}\\varepsilon$"); cb.ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_privacy_utility_map.pdf"))
    plt.close(fig)


# ===========================================================================
# FIGURA 4 — finestra de compromís (eixos dobles, Cora, RR)
# ===========================================================================
def fig_4(out, data):
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    eps_u, U, base_u = data["Cora"]["util"]; A = data["Cora"]["atk"]
    eps = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    xi = np.arange(len(eps)); labels = [eps_label(e) for e in eps]
    s1 = [U["rr_dp"]["s1"][e][0] for e in eps]
    s2 = [U["rr_dp"]["s2"][e][0] for e in eps]
    auc = [A[("RR", e)]["embedding_cosine"][0] for e in eps]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    l1, = ax.plot(xi, s1, "-o", color=F4_S1, ms=5, lw=1.7, label="Utilitat S1 (priv. total)")
    l2, = ax.plot(xi, s2, "-s", color=F4_S2, ms=5, lw=1.7, label="Utilitat S2 (aval. real)")
    ax.set_xlabel(r"Pressupost de privadesa $\varepsilon$")
    ax.set_ylabel("Precisió de test (accuracy)")
    ax.set_ylim(0.3, 1.0); ax.set_xticks(xi); ax.set_xticklabels(labels)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax2 = ax.twinx()
    l3, = ax2.plot(xi, auc, "--^", color=F4_AUC, ms=5, lw=1.7, label="AUC atac (embedding cosine)")
    ax2.set_ylabel("AUC-ROC de l'atac"); ax2.set_ylim(0.5, 1.0)
    fig.legend(handles=[l1, l2, l3], loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "fig4_tradeoff_window.pdf"))
    plt.close(fig)


# ===========================================================================
# FIGURA 5 — índex de compromís J = 1/2 (U+P)  (Cora, RR)
# ===========================================================================
def fig_5(out, data):
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    eps_u, U, base_u = data["Cora"]["util"]; A = data["Cora"]["atk"]
    eps = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    xi = np.arange(len(eps)); labels = [eps_label(e) for e in eps]
    Uv = np.array([U["rr_dp"]["s2"][e][0] for e in eps])
    AUC = np.array([A[("RR", e)]["embedding_cosine"][0] for e in eps])
    P = 2 * (1 - AUC); J = (Uv + P) / 2
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(xi, Uv, "-s", color=J_U, ms=6, lw=1.8, label=r"Utilitat $U(\varepsilon)$ (S2)")
    ax.plot(xi, P, "-o", color=J_P, ms=6, lw=1.8, label=r"Protecció $P(\varepsilon)=2(1-\mathrm{AUC})$")
    ax.plot(xi, J, "-o", color=J_J, ms=6, lw=1.8, label=r"Compromís $J(\varepsilon)=\frac{1}{2}(U+P)$")
    im = int(np.argmax(J))
    ax.axvline(xi[im], color="grey", ls="--", lw=1.1)
    ax.annotate("màxim $\\varepsilon = 1$\n$J \\approx %s$" % (("%.2f" % J[im]).replace(".", ",")),
                xy=(xi[im], J[im]), xytext=(4.2, 0.93), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
    ax.set_xlabel(r"Pressupost de privadesa $\varepsilon$")
    ax.set_ylabel("Valor (escala normalitzada)")
    ax.set_xticks(xi); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.4, len(eps) - 0.6)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.70), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_operating_point.pdf"))
    plt.close(fig)


# ===========================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Reprodueix totes les figures de l'informe.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures_generated")
    args = ap.parse_args()
    # Les rutes relatives es resolen respecte a la ubicació de l'script,
    # de manera que funciona des de qualsevol carpeta de treball.
    results_dir = args.results_dir if os.path.isabs(args.results_dir) else os.path.join(here, args.results_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(here, args.out_dir)
    if not os.path.isdir(results_dir):
        raise SystemExit(f"No trobo la carpeta de resultats: {results_dir}\n"
                         f"Comprova que 'results/' és al costat de generate_figures.py, "
                         f"o passa-la amb --results-dir RUTA.")
    os.makedirs(out_dir, exist_ok=True)
    data = {}
    for ds in DATASETS:
        data[ds] = {"util": load_utility(results_dir, ds),
                    "atk": load_attack(results_dir, ds)}
    fig_A(out_dir, data); print("  figA_utility_multidataset_vertical.pdf")
    fig_B(out_dir, data); print("  figB_s1_gap_multidataset.pdf")
    fig_2(out_dir, data); print("  fig2_attack_auc.pdf")
    fig_3(out_dir, data); print("  fig3_privacy_utility_map.pdf")
    fig_4(out_dir, data); print("  fig4_tradeoff_window.pdf")
    fig_5(out_dir, data); print("  fig5_operating_point.pdf")
    print("Fet. Figures a:", out_dir)


if __name__ == "__main__":
    main()