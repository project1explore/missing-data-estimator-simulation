#!/usr/bin/env python3
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    if x in (None, "", "None"):
        return None
    return float(x)


def find_value(rows, *, mechanism, method, metric, n_sample=700, condition="correct", null_case="False"):
    for r in rows:
        if (
            r["mechanism"] == mechanism
            and r["method"] == method
            and int(float(r["n_sample"])) == n_sample
            and r["condition"] == condition
            and r["null_case"] == null_case
        ):
            return f(r[metric])
    return None


def grouped_bar(ax, categories, methods, values_by_method, ylabel, title, ylim=None, hline=None, errors_by_method=None):
    n_cat = len(categories)
    n_m = len(methods)
    x = list(range(n_cat))
    width = 0.8 / n_m
    offsets = [(-0.4 + width / 2) + i * width for i in range(n_m)]

    colors = {
        "CC": "#4C78A8",
        "OM": "#F58518",
        "IPW": "#54A24B",
        "AIPW": "#E45756",
    }

    for i, m in enumerate(methods):
        vals = values_by_method[m]
        xpos = [xx + offsets[i] for xx in x]
        safe_vals = [v if v is not None else float("nan") for v in vals]
        yerr = None
        if errors_by_method is not None and m in errors_by_method:
            yerr = [e if e is not None else float("nan") for e in errors_by_method[m]]
        ax.bar(
            xpos,
            safe_vals,
            width=width,
            label=m,
            color=colors.get(m, None),
            edgecolor="black",
            linewidth=0.4,
            yerr=yerr,
            ecolor="black",
            capsize=2,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        ncols=1,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    ax.grid(axis="y", alpha=0.25)
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    if ylim is not None:
        ax.set_ylim(*ylim)


def make_figures(rows):
    os.makedirs("results", exist_ok=True)

    mech_order = {"MAR": 0, "MNARmod": 1, "MNARstrong": 2, "MNARyOnly": 3, "MNARintStrong": 4}
    mech_labels_map = {
        "MAR": "MAR",
        "MNARmod": "MNAR-mod",
        "MNARstrong": "MNAR-strong",
        "MNARyOnly": "MNAR-Y",
        "MNARintStrong": "MNAR-A×Y",
    }
    mechs = sorted(
        {
            r["mechanism"]
            for r in rows
            if r["null_case"] == "False" and int(float(r["n_sample"])) == 700 and r["condition"] == "correct"
        },
        key=lambda m: mech_order.get(m, 99),
    )
    mech_labels = [mech_labels_map.get(m, m) for m in mechs]

    methods_all = ["CC", "OM", "IPW", "AIPW"]

    bias = {m: [find_value(rows, mechanism=mech, method=m, metric="bias") for mech in mechs] for m in methods_all}
    rmse = {m: [find_value(rows, mechanism=mech, method=m, metric="rmse") for mech in mechs] for m in methods_all}

    bias_mcse = {m: [find_value(rows, mechanism=mech, method=m, metric="mcse_bias") for mech in mechs] for m in methods_all}
    rmse_mcse = {m: [find_value(rows, mechanism=mech, method=m, metric="mcse_rmse") for mech in mechs] for m in methods_all}

    methods_cov = ["CC", "OM", "IPW", "AIPW"]
    coverage = {m: [find_value(rows, mechanism=mech, method=m, metric="coverage") for mech in mechs] for m in methods_cov}
    cov_mcse = {m: [find_value(rows, mechanism=mech, method=m, metric="mcse_coverage") for mech in mechs] for m in methods_cov}

    bias_ci = {
        m: [None if e is None else 1.96 * e for e in bias_mcse[m]]
        for m in methods_all
    }
    rmse_ci = {
        m: [None if e is None else 1.96 * e for e in rmse_mcse[m]]
        for m in methods_all
    }
    cov_ci = {
        m: [None if e is None else 1.96 * e for e in cov_mcse[m]]
        for m in methods_cov
    }

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    # Bias
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    grouped_bar(
        ax,
        mech_labels,
        methods_all,
        bias,
        ylabel="Bias",
        title="Bias by missingness mechanism (N=700, baseline-model specification)",
        hline=0.0,
        errors_by_method=bias_ci,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    fig.savefig("results/fig_bias_python.pdf")
    plt.close(fig)

    # RMSE
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    grouped_bar(
        ax,
        mech_labels,
        methods_all,
        rmse,
        ylabel="RMSE",
        title="RMSE by missingness mechanism (N=700, baseline-model specification)",
        errors_by_method=rmse_ci,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    fig.savefig("results/fig_rmse_python.pdf")
    plt.close(fig)

    # Coverage
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    grouped_bar(
        ax,
        mech_labels,
        methods_cov,
        coverage,
        ylabel="Empirical 95% coverage",
        title="Coverage by missingness mechanism (N=700, baseline-model specification)",
        ylim=(0.4, 1.0),
        hline=0.95,
        errors_by_method=cov_ci,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    fig.savefig("results/fig_coverage_python.pdf")
    plt.close(fig)

    # Sample-size sensitivity (RMSE, MNARstrong, correct)
    nvals = [350, 700]
    x = [0, 1]
    labels = ["N=350", "N=700"]

    rmse_sz = {m: [] for m in methods_all}
    for m in methods_all:
        for n in nvals:
            val = None
            for r in rows:
                if (
                    r["null_case"] == "False"
                    and int(float(r["n_sample"])) == n
                    and r["mechanism"] == "MNARstrong"
                    and r["condition"] == "correct"
                    and r["method"] == m
                ):
                    val = f(r["rmse"])
                    break
            rmse_sz[m].append(val)

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    markers = {"CC": "o", "OM": "s", "IPW": "^", "AIPW": "D"}
    colors = {"CC": "#4C78A8", "OM": "#F58518", "IPW": "#54A24B", "AIPW": "#E45756"}
    for m in methods_all:
        vals = [v if v is not None else float("nan") for v in rmse_sz[m]]
        ax.plot(x, vals, marker=markers[m], color=colors[m], linewidth=1.8, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Sample size (strong MNAR-like, baseline-model setting)")
    ax.set_title("Sample-size sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        ncols=1,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    fig.savefig("results/fig_samplesize_python.pdf")
    plt.close(fig)


def main():
    rows = read_csv("results/summary_metrics.csv")
    make_figures(rows)
    print("Wrote matplotlib PDF figures: results/fig_*_python.pdf")


if __name__ == "__main__":
    main()
