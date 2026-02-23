#!/usr/bin/env python3
import csv


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    return float(x) if x not in (None, "", "None") else None


def method_order(m):
    return {"CC": 0, "OM": 1, "IPW": 2, "AIPW": 3}.get(m, 99)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as fobj:
        w = csv.DictWriter(fobj, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    rows = read("results/summary_metrics.csv")

    mech_order = ["MAR", "MNARmod", "MNARstrong"]

    # Figure 1: Bias & RMSE for N=700, non-null, correct models
    fig1_base = [
        r
        for r in rows
        if r["null_case"] == "False"
        and int(float(r["n_sample"])) == 700
        and r["condition"] == "correct"
    ]

    wide1 = []
    for i, mech in enumerate(mech_order, start=1):
        subset = [r for r in fig1_base if r["mechanism"] == mech]
        rec = {"x": i, "mechanism": mech}
        for m in ["CC", "OM", "IPW", "AIPW"]:
            rr = next((z for z in subset if z["method"] == m), None)
            rec[f"{m}_bias"] = "" if rr is None else f(rr["bias"])
            rec[f"{m}_rmse"] = "" if rr is None else f(rr["rmse"])
        wide1.append(rec)

    write_csv(
        "results/fig_bias_rmse.csv",
        [
            "x",
            "mechanism",
            "CC_bias",
            "OM_bias",
            "IPW_bias",
            "AIPW_bias",
            "CC_rmse",
            "OM_rmse",
            "IPW_rmse",
            "AIPW_rmse",
        ],
        wide1,
    )

    # Figure 2: Coverage for N=700, non-null, correct models
    wide2 = []
    for i, mech in enumerate(mech_order, start=1):
        subset = [r for r in fig1_base if r["mechanism"] == mech]
        rec = {"x": i, "mechanism": mech}
        for m in ["CC", "OM", "IPW", "AIPW"]:
            rr = next((z for z in subset if z["method"] == m), None)
            rec[f"{m}_coverage"] = "" if rr is None else f(rr["coverage"])
        wide2.append(rec)

    write_csv(
        "results/fig_coverage.csv",
        ["x", "mechanism", "CC_coverage", "OM_coverage", "IPW_coverage", "AIPW_coverage"],
        wide2,
    )

    # Figure 3: sample-size sensitivity under strongest mechanism (correct model)
    fig3_base = [
        r
        for r in rows
        if r["null_case"] == "False"
        and r["condition"] == "correct"
        and r["mechanism"] == "MNARstrong"
    ]

    nvals = sorted({int(float(r["n_sample"])) for r in fig3_base})
    wide3 = []
    for i, n in enumerate(nvals, start=1):
        subset = [r for r in fig3_base if int(float(r["n_sample"])) == n]
        rec = {"x": i, "n_sample": n}
        for m in ["CC", "OM", "IPW", "AIPW"]:
            rr = next((z for z in subset if z["method"] == m), None)
            rec[f"{m}_rmse"] = "" if rr is None else f(rr["rmse"])
        wide3.append(rec)

    write_csv(
        "results/fig_samplesize.csv",
        ["x", "n_sample", "CC_rmse", "OM_rmse", "IPW_rmse", "AIPW_rmse"],
        wide3,
    )

    print("Wrote plot data: fig_bias_rmse.csv, fig_coverage.csv, fig_samplesize.csv")


if __name__ == "__main__":
    main()
