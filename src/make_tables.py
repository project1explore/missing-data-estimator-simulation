#!/usr/bin/env python3
import csv


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f3(x):
    if x in (None, "", "None"):
        return "-"
    return f"{float(x):.3f}"


def f1(x):
    if x in (None, "", "None"):
        return "-"
    return f"{float(x):.1f}"


def fpct(x):
    if x in (None, "", "None"):
        return "-"
    return f"{100.0 * float(x):.1f}"


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def mechanism_label(x):
    return {
        "MAR": "MAR",
        "MNARmod": "MNAR-like (moderate)",
        "MNARstrong": "MNAR-like (strong)",
        "MNARyOnly": "MNAR-like (Y-only)",
        "MNARintStrong": "MNAR-like (strong A×Y)",
    }.get(x, x)


def condition_label(x):
    return {
        "correct": "Correct nuisance models",
        "outcome_misspec": "Outcome model misspecified",
        "missing_misspec": "Missingness model misspecified",
        "both_misspec": "Both misspecified",
    }.get(x, x)


def mechanism_order(x):
    return {"MAR": 0, "MNARmod": 1, "MNARstrong": 2, "MNARyOnly": 3, "MNARintStrong": 4}.get(x, 99)


def method_order(x):
    return {"CC": 0, "OM": 1, "IPW": 2, "AIPW": 3}.get(x, 99)


def condition_order(x):
    return {"correct": 0, "outcome_misspec": 1, "missing_misspec": 2, "both_misspec": 3}.get(x, 99)


def enrich_type1(rows, null_lookup):
    out = []
    for r in rows:
        z = dict(r)
        key = (int(float(r["n_sample"])), r["mechanism"], r["condition"], r["method"])
        if r["null_case"] == "False":
            t = null_lookup.get(key)
            if t not in (None, "", "None"):
                z["type1"] = t
        out.append(z)
    return out


def write_table(rows, out_path, include_n=False, include_condition=False):
    rows = sorted(
        rows,
        key=lambda r: (
            int(float(r["n_sample"])),
            mechanism_order(r["mechanism"]),
            condition_order(r["condition"]),
            method_order(r["method"]),
        ),
    )

    cols = []
    hdr = []
    if include_n:
        cols.append("n")
        hdr.append("N")
    if include_condition:
        cols.append("condition")
        hdr.append("Nuisance-model setting")
    cols.extend(["mechanism", "method", "bias", "rmse", "coverage", "type1", "obs_rate", "n_reps"])
    hdr.extend(["Mechanism", "Method", "Bias", "RMSE", "Coverage", "Type I (matched null)", "Obs. rate", "Reps"])

    align = "l" * (len(cols) - 5) + "r" * 5

    with open(out_path, "w") as f:
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\hline\n")
        f.write(" & ".join(hdr) + r" \\\\" + "\n")
        f.write("\\hline\n")

        for r in rows:
            parts = []
            if include_n:
                parts.append(str(int(float(r["n_sample"]))))
            if include_condition:
                parts.append(latex_escape(condition_label(r["condition"])))
            parts.append(latex_escape(mechanism_label(r["mechanism"])))
            parts.append(latex_escape(r["method"]))
            parts.append(f3(r["bias"]))
            parts.append(f3(r["rmse"]))
            parts.append(f3(r["coverage"]))
            parts.append(f3(r["type1"]))
            parts.append(f3(r["obs_rate"]))
            parts.append(str(int(float(r["n_reps"]))))
            f.write(" & ".join(parts) + r" \\" + "\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def write_weight_table(rows, out_path):
    rows = sorted(rows, key=lambda r: (mechanism_order(r["mechanism"]), method_order(r["method"])))

    with open(out_path, "w") as f:
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\hline\n")
        f.write("Mechanism & Method & min($\\hat p$) & med($\\hat p$) & max($\\hat p$) & max($w$) & ESS & Trunc. (\\%) " + r" \\" + "\n")
        f.write("\\hline\n")
        for r in rows:
            f.write(
                f"{latex_escape(mechanism_label(r['mechanism']))} & {latex_escape(r['method'])} & "
                f"{f3(r['p_hat_min'])} & {f3(r['p_hat_med'])} & {f3(r['p_hat_max'])} & "
                f"{f3(r['w_max'])} & {f1(r['ess'])} & {fpct(r['trunc_frac'])}" + r" \\" + "\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def write_counts(summary_rows, raw_rows, out_path):
    n_scen = len({r["scenario"] for r in summary_rows})
    n_methods = len({r["method"] for r in summary_rows})
    n_reps = int(float(summary_rows[0]["n_reps"])) if summary_rows else 0

    n_datasets = len({(r["scenario"], r["rep"]) for r in raw_rows})
    n_method_fits = len(raw_rows)
    sample_sizes = sorted({int(float(r["n_sample"])) for r in summary_rows if r["null_case"] == "False"})

    with open(out_path, "w") as f:
        f.write(f"\\newcommand{{\\NumScenarios}}{{{n_scen}}}\n")
        f.write(f"\\newcommand{{\\NumMethods}}{{{n_methods}}}\n")
        f.write(f"\\newcommand{{\\NumReps}}{{{n_reps}}}\n")
        f.write(f"\\newcommand{{\\NumDatasets}}{{{n_datasets}}}\n")
        f.write(f"\\newcommand{{\\NumMethodFits}}{{{n_method_fits}}}\n")
        f.write("\\newcommand{\\SampleSizes}{" + ", ".join(str(x) for x in sample_sizes) + "}\n")


def main():
    summary = read("results/summary_metrics.csv")
    raw = read("results/simulation_raw.csv")

    null_rows_all = [r for r in summary if r["null_case"] == "True"]
    null_lookup = {
        (int(float(r["n_sample"])), r["mechanism"], r["condition"], r["method"]): r["type1"]
        for r in null_rows_all
    }

    non_null = [r for r in summary if r["null_case"] == "False"]
    non_null = enrich_type1(non_null, null_lookup)

    n700_correct = [r for r in non_null if int(float(r["n_sample"])) == 700 and r["condition"] == "correct"]
    n700_misspec = [r for r in non_null if int(float(r["n_sample"])) == 700 and r["condition"] != "correct"]
    sample_size = [r for r in non_null if r["condition"] == "correct"]

    # Keep null table readable: show correct-model null scenarios across both sample sizes.
    null_rows = [r for r in null_rows_all if r["condition"] == "correct"]

    write_table(n700_correct, "results/table_correct.tex", include_n=False, include_condition=False)
    write_table(n700_misspec, "results/table_misspec.tex", include_n=False, include_condition=True)
    write_table(sample_size, "results/table_samplesize.tex", include_n=True, include_condition=False)
    write_table(null_rows, "results/table_null.tex", include_n=True, include_condition=False)

    weight_rows = [
        r
        for r in non_null
        if int(float(r["n_sample"])) == 700 and r["condition"] == "correct" and r["method"] in {"IPW", "AIPW"}
    ]
    write_weight_table(weight_rows, "results/table_weights.tex")

    write_counts(summary, raw, "results/study_counts.tex")

    print(
        "Wrote tables: table_correct.tex, table_misspec.tex, table_samplesize.tex, "
        "table_null.tex, table_weights.tex"
    )
    print("Wrote counts macros: results/study_counts.tex")


if __name__ == "__main__":
    main()
