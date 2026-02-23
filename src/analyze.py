import csv
import math
from collections import defaultdict


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(rows, k):
    return [float(r[k]) for r in rows]


def summarize(vals, ses, true_beta):
    n = len(vals)
    bias = sum(v - true_beta for v in vals) / n
    rmse = math.sqrt(sum((v - true_beta) ** 2 for v in vals) / n)
    if ses is not None:
        cov = 0
        for v, s in zip(vals, ses):
            lo, hi = v - 1.96 * s, v + 1.96 * s
            if lo <= true_beta <= hi:
                cov += 1
        cov /= n
    else:
        cov = float("nan")
    return bias, rmse, cov


def main():
    rows = read_csv("results/simulation_raw.csv")
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)

    out = []
    for scn, rr in by.items():
        true_beta = float(rr[0]["true_beta"])
        obs_rate = sum(float(x["obs_rate"]) for x in rr) / len(rr)

        cc = to_float(rr, "cc_beta")
        cc_se = to_float(rr, "cc_se")
        ipw = to_float(rr, "ipw_beta")
        ipw_se = to_float(rr, "ipw_se")
        aipw = to_float(rr, "aipw_beta")

        for m, vals, ses in [("CC", cc, cc_se), ("IPW", ipw, ipw_se), ("AIPW", aipw, None)]:
            bias, rmse, cov = summarize(vals, ses, true_beta)
            type1 = ""
            if abs(true_beta) < 1e-12 and ses is not None:
                rej = 0
                for v, s in zip(vals, ses):
                    z = abs(v / max(s, 1e-6))
                    if z > 1.96:
                        rej += 1
                type1 = f"{rej/len(vals):.3f}"
            out.append(
                {
                    "scenario": scn,
                    "method": m,
                    "bias": f"{bias:.3f}",
                    "rmse": f"{rmse:.3f}",
                    "coverage": "" if math.isnan(cov) else f"{cov:.3f}",
                    "type1": type1,
                    "obs_rate": f"{obs_rate:.3f}",
                    "n_reps": str(len(vals)),
                }
            )

    with open("results/summary_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    with open("results/table_main.tex", "w") as f:
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\hline\n")
        f.write("Scenario & Method & Bias & RMSE & Coverage & Type I & ObsRate \\\\ \n")
        f.write("\\hline\n")
        for r in out:
            f.write(
                f"{r['scenario']} & {r['method']} & {r['bias']} & {r['rmse']} & {r['coverage'] or '-'} & {r['type1'] or '-'} & {r['obs_rate']} \\\\ \n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    print("Wrote results/summary_metrics.csv and results/table_main.tex")


if __name__ == "__main__":
    main()
