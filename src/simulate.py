import argparse
import csv
import math
import random
from dataclasses import dataclass, asdict


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def fit_logistic_sgd(X, y, lr=0.01, epochs=300):
    p = len(X[0])
    w = [0.0] * p
    n = len(X)
    if n == 0:
        return w
    for _ in range(epochs):
        for i in range(n):
            pi = sigmoid(dot(w, X[i]))
            err = y[i] - pi
            for j in range(p):
                w[j] += lr * err * X[i][j]
    return w


def predict_logistic(w, X):
    return [sigmoid(dot(w, x)) for x in X]


@dataclass
class Scenario:
    name: str
    n: int
    mnar_strength: float
    misspecified_outcome: bool
    beta_treat: float


def generate_data(n, beta_treat, rng):
    rows = []
    for _ in range(n):
        age = rng.gauss(60, 12)
        comorb = max(0, int(rng.expovariate(0.5)))
        sev = rng.gauss(0, 1)
        biom = rng.gauss(0, 1)

        lp_a = -0.8 + 0.015 * (age - 60) + 0.25 * sev + 0.12 * comorb
        p_a = sigmoid(lp_a)
        A = 1 if rng.random() < p_a else 0

        lp_y = (
            -0.3
            + beta_treat * A
            + 0.5 * sev
            + 0.25 * comorb
            + 0.18 * biom
            + 0.22 * A * sev
            - 0.08 * sev * sev
        )
        p_y = sigmoid(lp_y)
        Y = 1 if rng.random() < p_y else 0
        rows.append({"age": age, "comorb": comorb, "sev": sev, "biom": biom, "A": A, "Y": Y})
    return rows


def impose_missingness(rows, mnar_strength, rng):
    out = []
    for r in rows:
        lp_r = 1.1 - 0.35 * r["sev"] - 0.12 * r["comorb"] + 0.08 * r["A"] - mnar_strength * r["Y"]
        p_r = sigmoid(lp_r)
        R = 1 if rng.random() < p_r else 0
        z = dict(r)
        z["R"] = R
        z["Y_obs"] = r["Y"] if R == 1 else None
        out.append(z)
    return out


def safe_risk(num, den):
    if den <= 0:
        return 0.5
    return min(max(num / den, 1e-6), 1 - 1e-6)


def cc_estimator(rows):
    obs = [r for r in rows if r["R"] == 1]
    a1 = [r for r in obs if r["A"] == 1]
    a0 = [r for r in obs if r["A"] == 0]

    y1 = sum(r["Y_obs"] for r in a1)
    y0 = sum(r["Y_obs"] for r in a0)
    n1, n0 = len(a1), len(a0)
    p1 = safe_risk(y1, n1)
    p0 = safe_risk(y0, n0)
    beta = logit(p1) - logit(p0)

    # rough Wald SE from 2-sample logits
    se = math.sqrt(1 / max(y1, 1) + 1 / max(n1 - y1, 1) + 1 / max(y0, 1) + 1 / max(n0 - y0, 1))
    return beta, se


def fit_obs_model(rows):
    X, y = [], []
    for r in rows:
        X.append([1.0, r["A"], (r["age"] - 60) / 10.0, r["comorb"], r["sev"], r["biom"]])
        y.append(r["R"])
    w = fit_logistic_sgd(X, y, lr=0.01, epochs=200)
    p = predict_logistic(w, X)
    return [min(max(pi, 0.02), 0.98) for pi in p]


def fit_outcome_model(rows, misspecified=False):
    obs = [r for r in rows if r["R"] == 1]
    X, y = [], []
    for r in obs:
        if misspecified:
            x = [1.0, r["A"], (r["age"] - 60) / 10.0, r["comorb"], r["sev"], r["biom"]]
        else:
            x = [1.0, r["A"], (r["age"] - 60) / 10.0, r["comorb"], r["sev"], r["biom"], r["A"] * r["sev"], r["sev"] * r["sev"]]
        X.append(x)
        y.append(r["Y_obs"])
    w = fit_logistic_sgd(X, y, lr=0.01, epochs=250)

    def mu(r, a_val):
        if misspecified:
            x = [1.0, a_val, (r["age"] - 60) / 10.0, r["comorb"], r["sev"], r["biom"]]
        else:
            x = [1.0, a_val, (r["age"] - 60) / 10.0, r["comorb"], r["sev"], r["biom"], a_val * r["sev"], r["sev"] * r["sev"]]
        return sigmoid(dot(w, x))

    return mu


def ipw_estimator(rows, p_r):
    num1 = den1 = num0 = den0 = 0.0
    for r, pr in zip(rows, p_r):
        if r["R"] == 1:
            w = 1.0 / pr
            if r["A"] == 1:
                num1 += w * r["Y_obs"]
                den1 += w
            else:
                num0 += w * r["Y_obs"]
                den0 += w
    p1 = safe_risk(num1, den1)
    p0 = safe_risk(num0, den0)
    beta = logit(p1) - logit(p0)
    se = math.sqrt(1 / max(num1, 1e-3) + 1 / max(den1 - num1, 1e-3) + 1 / max(num0, 1e-3) + 1 / max(den0 - num0, 1e-3))
    return beta, se


def aipw_estimator(rows, p_r, mu):
    n = len(rows)
    pa = sum(r["A"] for r in rows) / n
    pa = min(max(pa, 0.05), 0.95)

    psi1 = 0.0
    psi0 = 0.0
    for r, pr in zip(rows, p_r):
        y = r["Y"]
        rr = r["R"]
        a = r["A"]
        m1 = mu(r, 1)
        m0 = mu(r, 0)
        psi1 += m1 + (rr * (1 if a == 1 else 0) * (y - m1)) / (pr * pa)
        psi0 += m0 + (rr * (1 if a == 0 else 0) * (y - m0)) / (pr * (1 - pa))
    psi1 /= n
    psi0 /= n
    psi1 = min(max(psi1, 1e-6), 1 - 1e-6)
    psi0 = min(max(psi0, 1e-6), 1 - 1e-6)
    return logit(psi1) - logit(psi0)


def run_scenario(scn, n_reps, seed):
    rng = random.Random(seed)
    out = []
    for rep in range(n_reps):
        rows = generate_data(scn.n, scn.beta_treat, rng)
        rows = impose_missingness(rows, scn.mnar_strength, rng)

        cc_beta, cc_se = cc_estimator(rows)
        p_r = fit_obs_model(rows)
        ipw_beta, ipw_se = ipw_estimator(rows, p_r)
        mu = fit_outcome_model(rows, misspecified=scn.misspecified_outcome)
        aipw_beta = aipw_estimator(rows, p_r, mu)

        out.append(
            {
                "scenario": scn.name,
                "rep": rep,
                "true_beta": scn.beta_treat,
                "cc_beta": cc_beta,
                "cc_se": cc_se,
                "ipw_beta": ipw_beta,
                "ipw_se": ipw_se,
                "aipw_beta": aipw_beta,
                "obs_rate": sum(r["R"] for r in rows) / len(rows),
            }
        )
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260221)
    ap.add_argument("--out", default="results/simulation_raw.csv")
    args = ap.parse_args()

    scenarios = [
        Scenario("MAR_correct", 600, 0.0, False, 0.45),
        Scenario("MNARmod_correct", 600, 0.5, False, 0.45),
        Scenario("MNARstrong_correct", 600, 1.0, False, 0.45),
        Scenario("MAR_misspec", 600, 0.0, True, 0.45),
        Scenario("MNARmod_misspec", 600, 0.5, True, 0.45),
        Scenario("MNARstrong_misspec", 600, 1.0, True, 0.45),
        Scenario("NULL_MNARmod_correct", 600, 0.5, False, 0.0),
    ]

    all_rows = []
    for s in scenarios:
        all_rows.extend(run_scenario(s, args.n_reps, args.seed + hash(s.name) % 10000))

    write_csv(args.out, all_rows)
    print(f"Wrote {args.out} with {len(all_rows)} rows")


if __name__ == "__main__":
    main()
