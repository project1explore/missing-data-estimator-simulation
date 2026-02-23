#!/usr/bin/env python3
import argparse
import csv
import hashlib
import math
import multiprocessing as mp
import os
import random
from collections import defaultdict
from dataclasses import dataclass


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p):
    p = min(max(p, 1e-8), 1 - 1e-8)
    return math.log(p / (1 - p))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def poisson_knuth(lam, rng):
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return k - 1


def solve_linear_system(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("Singular matrix")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]

        pivval = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivval

        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f == 0:
                continue
            for j in range(col, n + 1):
                M[r][j] -= f * M[col][j]

    return [M[i][n] for i in range(n)]


def invert_matrix(A):
    n = len(A)
    inv = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        inv.append(solve_linear_system(A, e))
    return [[inv[c][r] for c in range(n)] for r in range(n)]


def vec_mat_mul(v, M):
    return [sum(vj * M[j][k] for j, vj in enumerate(v)) for k in range(len(M[0]))]


def mat_vec_mul(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sample_var(xs):
    n = len(xs)
    if n <= 1:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def quantile(xs, q):
    if not xs:
        return 0.0
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def fit_logistic(X, y, weights=None, max_iter=60, tol=1e-7, ridge=1e-6):
    n = len(X)
    p = len(X[0])
    if weights is None:
        weights = [1.0] * n

    beta = [0.0] * p

    for _ in range(max_iter):
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]

        for i in range(n):
            xi = X[i]
            yi = y[i]
            wi = weights[i]
            pi = min(max(sigmoid(dot(beta, xi)), 1e-8), 1 - 1e-8)

            err = wi * (yi - pi)
            for j in range(p):
                g[j] += err * xi[j]

            v = wi * pi * (1 - pi)
            for j in range(p):
                for k in range(p):
                    H[j][k] += v * xi[j] * xi[k]

        for j in range(p):
            H[j][j] += ridge

        step = solve_linear_system(H, g)
        beta_new = [beta[j] + step[j] for j in range(p)]

        if max(abs(beta_new[j] - beta[j]) for j in range(p)) < tol:
            beta = beta_new
            break
        beta = beta_new

    H_final = [[0.0] * p for _ in range(p)]
    for i in range(n):
        xi = X[i]
        wi = weights[i]
        pi = min(max(sigmoid(dot(beta, xi)), 1e-8), 1 - 1e-8)
        v = wi * pi * (1 - pi)
        for j in range(p):
            for k in range(p):
                H_final[j][k] += v * xi[j] * xi[k]
    for j in range(p):
        H_final[j][j] += ridge

    cov = invert_matrix(H_final)
    return beta, cov


@dataclass
class Scenario:
    name: str
    n: int
    mechanism: str
    condition: str
    null_case: bool
    mnar_gamma: float
    mnar_ay: float
    misspec_outcome: bool
    misspec_missing: bool
    beta_treat: float
    beta_int: float


P_TRUNC_LO = 0.02
P_TRUNC_HI = 0.98


def sample_covariates(rng):
    age = rng.gauss(60, 11)
    comorb = poisson_knuth(1.8, rng)
    sev = rng.gauss(0, 1)
    biom = rng.gauss(0, 1)
    return age, comorb, sev, biom


def outcome_prob(age, comorb, sev, biom, a, beta_treat, beta_int=0.28):
    age_c = (age - 60) / 10.0
    lp = (
        -0.8
        + beta_treat * a
        + 0.18 * age_c
        + 0.22 * comorb
        + 0.65 * sev
        + 0.20 * biom
        + beta_int * a * sev
        - 0.12 * sev * sev
    )
    return sigmoid(lp)


def generate_dataset(n, beta_treat, beta_int, gamma, gamma_ay, rng):
    rows = []
    for _ in range(n):
        age, comorb, sev, biom = sample_covariates(rng)
        a = 1 if rng.random() < 0.5 else 0
        py = outcome_prob(age, comorb, sev, biom, a, beta_treat, beta_int=beta_int)
        y = 1 if rng.random() < py else 0

        age_c = (age - 60) / 10.0
        lp_r = 1.4 - 0.10 * age_c - 0.10 * comorb - 0.35 * sev + 0.05 * a - gamma * y - gamma_ay * a * y
        pr = sigmoid(lp_r)
        r = 1 if rng.random() < pr else 0

        rows.append(
            {
                "age": age,
                "age_c": age_c,
                "comorb": comorb,
                "sev": sev,
                "biom": biom,
                "A": a,
                "Y": y,
                "R": r,
                "Y_obs": y if r == 1 else None,
            }
        )
    return rows


def x_outcome(row, a_override=None, misspecified=False):
    a = row["A"] if a_override is None else a_override
    if misspecified:
        return [1.0, a, row["age_c"], row["comorb"], row["sev"]]
    return [1.0, a, row["age_c"], row["comorb"], row["sev"], row["biom"], a * row["sev"], row["sev"] * row["sev"]]


def x_missing(row, misspecified=False):
    if misspecified:
        return [1.0, row["A"], row["age_c"], row["comorb"]]
    return [1.0, row["A"], row["age_c"], row["comorb"], row["sev"], row["biom"]]


def fit_outcome_model(rows, misspecified=False):
    obs = [r for r in rows if r["R"] == 1]
    X = [x_outcome(r, misspecified=misspecified) for r in obs]
    y = [r["Y_obs"] for r in obs]
    beta, cov = fit_logistic(X, y)

    def mu(r, a_val):
        x = x_outcome(r, a_override=a_val, misspecified=misspecified)
        return min(max(sigmoid(dot(beta, x)), 1e-8), 1 - 1e-8)

    return beta, cov, mu


def fit_missing_model(rows, misspecified=False):
    X = [x_missing(r, misspecified=misspecified) for r in rows]
    y = [r["R"] for r in rows]
    beta, cov = fit_logistic(X, y)

    p_raw = [sigmoid(dot(beta, x)) for x in X]
    p = [min(max(pr, P_TRUNC_LO), P_TRUNC_HI) for pr in p_raw]
    return beta, cov, X, p_raw, p


def estimate_cc(rows):
    obs = [r for r in rows if r["R"] == 1]
    n1 = sum(1 for r in obs if r["A"] == 1)
    n0 = sum(1 for r in obs if r["A"] == 0)
    y1 = sum(r["Y_obs"] for r in obs if r["A"] == 1)
    y0 = sum(r["Y_obs"] for r in obs if r["A"] == 0)

    p1 = min(max(y1 / max(n1, 1), 1e-8), 1 - 1e-8)
    p0 = min(max(y0 / max(n0, 1), 1e-8), 1 - 1e-8)
    est = logit(p1) - logit(p0)

    var_theta = 1.0 / max(n1 * p1 * (1 - p1), 1e-12) + 1.0 / max(n0 * p0 * (1 - p0), 1e-12)
    se = math.sqrt(max(var_theta, 1e-12))
    return est, se


def estimate_om(rows, misspec_outcome):
    beta, _, mu = fit_outcome_model(rows, misspecified=misspec_outcome)
    n = len(rows)
    p = len(beta)

    m1 = [mu(r, 1) for r in rows]
    m0 = [mu(r, 0) for r in rows]
    psi1 = min(max(mean(m1), 1e-8), 1 - 1e-8)
    psi0 = min(max(mean(m0), 1e-8), 1 - 1e-8)
    est = logit(psi1) - logit(psi0)

    # Outcome-model score and Jacobian (observed-data logistic model)
    A = [[0.0] * p for _ in range(p)]
    U = []
    for r in rows:
        if r["R"] == 1:
            x = x_outcome(r, misspecified=misspec_outcome)
            m_obs = min(max(sigmoid(dot(beta, x)), 1e-8), 1 - 1e-8)
            err = r["Y_obs"] - m_obs
            ui = [xj * err for xj in x]
            v = m_obs * (1 - m_obs)
            for j in range(p):
                for k in range(p):
                    A[j][k] += v * x[j] * x[k]
        else:
            ui = [0.0] * p
        U.append(ui)

    for j in range(p):
        A[j][j] += 1e-8
    A_inv = invert_matrix(A)

    # Mean derivatives of psi_a wrt outcome-model parameters
    b1 = [0.0] * p
    b0 = [0.0] * p
    for i, r in enumerate(rows):
        x1 = x_outcome(r, a_override=1, misspecified=misspec_outcome)
        x0 = x_outcome(r, a_override=0, misspecified=misspec_outcome)
        d1 = m1[i] * (1 - m1[i])
        d0 = m0[i] * (1 - m0[i])
        for j in range(p):
            b1[j] += d1 * x1[j] / n
            b0[j] += d0 * x0[j] / n

    adj1 = vec_mat_mul(b1, A_inv)
    adj0 = vec_mat_mul(b0, A_inv)

    if1 = []
    if0 = []
    for i in range(n):
        if1.append((m1[i] - psi1) + n * dot(adj1, U[i]))
        if0.append((m0[i] - psi0) + n * dot(adj0, U[i]))

    g1 = 1.0 / (psi1 * (1 - psi1))
    g0 = -1.0 / (psi0 * (1 - psi0))
    if_theta = [g1 * a + g0 * b for a, b in zip(if1, if0)]
    se = math.sqrt(max(sample_var(if_theta) / n, 1e-12))
    return est, se


def estimate_ipw(rows, misspec_missing):
    _, _, z_list, p_raw, p = fit_missing_model(rows, misspecified=misspec_missing)
    n = len(rows)
    pa = 0.5
    q = len(z_list[0])

    phi1 = []
    phi0 = []
    U = []
    A = [[0.0] * q for _ in range(q)]

    for i, r in enumerate(rows):
        rr = r["R"]
        a = r["A"]
        y = r["Y"] if rr == 1 else 0.0
        pr = p[i]
        z = z_list[i]

        phi1_i = rr * (1 if a == 1 else 0) * y / (pa * pr)
        phi0_i = rr * (1 if a == 0 else 0) * y / ((1 - pa) * pr)
        phi1.append(phi1_i)
        phi0.append(phi0_i)

        ui = [zj * (rr - pr) for zj in z]
        U.append(ui)

        v = pr * (1 - pr)
        for j in range(q):
            for k in range(q):
                A[j][k] += v * z[j] * z[k]

    for j in range(q):
        A[j][j] += 1e-8
    A_inv = invert_matrix(A)

    b1 = [0.0] * q
    b0 = [0.0] * q
    for i, r in enumerate(rows):
        rr = r["R"]
        a = r["A"]
        y = r["Y"] if rr == 1 else 0.0
        pr = p[i]
        z = z_list[i]

        c1 = -rr * (1 if a == 1 else 0) * y * (1 - pr) / (pa * pr)
        c0 = -rr * (1 if a == 0 else 0) * y * (1 - pr) / ((1 - pa) * pr)
        for j in range(q):
            b1[j] += c1 * z[j] / n
            b0[j] += c0 * z[j] / n

    adj1 = vec_mat_mul(b1, A_inv)
    adj0 = vec_mat_mul(b0, A_inv)

    psi1 = min(max(mean(phi1), 1e-8), 1 - 1e-8)
    psi0 = min(max(mean(phi0), 1e-8), 1 - 1e-8)
    est = logit(psi1) - logit(psi0)

    if1 = [(phi1[i] - psi1) + n * dot(adj1, U[i]) for i in range(n)]
    if0 = [(phi0[i] - psi0) + n * dot(adj0, U[i]) for i in range(n)]
    g1 = 1.0 / (psi1 * (1 - psi1))
    g0 = -1.0 / (psi0 * (1 - psi0))
    if_theta = [g1 * a + g0 * b for a, b in zip(if1, if0)]
    se = math.sqrt(max(sample_var(if_theta) / n, 1e-12))

    obs_idx = [i for i, r in enumerate(rows) if r["R"] == 1]
    obs_p = [p[i] for i in obs_idx]
    obs_w = [1.0 / p[i] for i in obs_idx]
    ess = (sum(obs_w) ** 2 / max(sum(w * w for w in obs_w), 1e-12)) if obs_w else 0.0
    trunc_frac = sum(1 for pr, pt in zip(p_raw, p) if abs(pr - pt) > 1e-12) / n
    diagnostics = {
        "p_hat_min": min(obs_p) if obs_p else None,
        "p_hat_med": quantile(obs_p, 0.5) if obs_p else None,
        "p_hat_max": max(obs_p) if obs_p else None,
        "w_max": max(obs_w) if obs_w else None,
        "ess": ess,
        "trunc_frac": trunc_frac,
    }

    return est, se, diagnostics


def estimate_aipw(rows, misspec_outcome, misspec_missing):
    beta, _, mu = fit_outcome_model(rows, misspecified=misspec_outcome)
    _, _, z_list, p_raw, p = fit_missing_model(rows, misspecified=misspec_missing)

    n = len(rows)
    pa = 0.5
    p_dim = len(beta)
    q_dim = len(z_list[0])

    # Outcome-model score pieces
    A_beta = [[0.0] * p_dim for _ in range(p_dim)]
    U_beta = []
    for r in rows:
        if r["R"] == 1:
            x_obs = x_outcome(r, misspecified=misspec_outcome)
            m_obs = min(max(sigmoid(dot(beta, x_obs)), 1e-8), 1 - 1e-8)
            err = r["Y_obs"] - m_obs
            ui = [xj * err for xj in x_obs]
            v = m_obs * (1 - m_obs)
            for j in range(p_dim):
                for k in range(p_dim):
                    A_beta[j][k] += v * x_obs[j] * x_obs[k]
        else:
            ui = [0.0] * p_dim
        U_beta.append(ui)
    for j in range(p_dim):
        A_beta[j][j] += 1e-8
    A_beta_inv = invert_matrix(A_beta)

    # Missingness-model score pieces
    A_alpha = [[0.0] * q_dim for _ in range(q_dim)]
    U_alpha = []
    for i, r in enumerate(rows):
        pr = p[i]
        z = z_list[i]
        rr = r["R"]
        ui = [zj * (rr - pr) for zj in z]
        U_alpha.append(ui)
        v = pr * (1 - pr)
        for j in range(q_dim):
            for k in range(q_dim):
                A_alpha[j][k] += v * z[j] * z[k]
    for j in range(q_dim):
        A_alpha[j][j] += 1e-8
    A_alpha_inv = invert_matrix(A_alpha)

    phi1 = []
    phi0 = []
    b1_beta = [0.0] * p_dim
    b0_beta = [0.0] * p_dim
    b1_alpha = [0.0] * q_dim
    b0_alpha = [0.0] * q_dim

    for i, r in enumerate(rows):
        rr = r["R"]
        a = r["A"]
        y = r["Y"] if rr == 1 else 0.0
        pr = p[i]
        z = z_list[i]

        m1 = mu(r, 1)
        m0 = mu(r, 0)
        x1 = x_outcome(r, a_override=1, misspecified=misspec_outcome)
        x0 = x_outcome(r, a_override=0, misspecified=misspec_outcome)

        phi1_i = m1 + rr * (1 if a == 1 else 0) * (y - m1) / (pa * pr)
        phi0_i = m0 + rr * (1 if a == 0 else 0) * (y - m0) / ((1 - pa) * pr)
        phi1.append(phi1_i)
        phi0.append(phi0_i)

        c1_beta = 1.0 - rr * (1 if a == 1 else 0) / (pa * pr)
        c0_beta = 1.0 - rr * (1 if a == 0 else 0) / ((1 - pa) * pr)
        d1 = m1 * (1 - m1)
        d0 = m0 * (1 - m0)
        for j in range(p_dim):
            b1_beta[j] += c1_beta * d1 * x1[j] / n
            b0_beta[j] += c0_beta * d0 * x0[j] / n

        c1_alpha = -rr * (1 if a == 1 else 0) * (y - m1) * (1 - pr) / (pa * pr)
        c0_alpha = -rr * (1 if a == 0 else 0) * (y - m0) * (1 - pr) / ((1 - pa) * pr)
        for j in range(q_dim):
            b1_alpha[j] += c1_alpha * z[j] / n
            b0_alpha[j] += c0_alpha * z[j] / n

    psi1 = min(max(mean(phi1), 1e-8), 1 - 1e-8)
    psi0 = min(max(mean(phi0), 1e-8), 1 - 1e-8)
    est = logit(psi1) - logit(psi0)

    adj1_beta = vec_mat_mul(b1_beta, A_beta_inv)
    adj0_beta = vec_mat_mul(b0_beta, A_beta_inv)
    adj1_alpha = vec_mat_mul(b1_alpha, A_alpha_inv)
    adj0_alpha = vec_mat_mul(b0_alpha, A_alpha_inv)

    if1 = []
    if0 = []
    for i in range(n):
        if1_i = (phi1[i] - psi1) + n * dot(adj1_beta, U_beta[i]) + n * dot(adj1_alpha, U_alpha[i])
        if0_i = (phi0[i] - psi0) + n * dot(adj0_beta, U_beta[i]) + n * dot(adj0_alpha, U_alpha[i])
        if1.append(if1_i)
        if0.append(if0_i)

    g1 = 1.0 / (psi1 * (1 - psi1))
    g0 = -1.0 / (psi0 * (1 - psi0))
    if_theta = [g1 * a + g0 * b for a, b in zip(if1, if0)]
    se = math.sqrt(max(sample_var(if_theta) / n, 1e-12))

    obs_idx = [i for i, r in enumerate(rows) if r["R"] == 1]
    obs_p = [p[i] for i in obs_idx]
    obs_w = [1.0 / p[i] for i in obs_idx]
    ess = (sum(obs_w) ** 2 / max(sum(w * w for w in obs_w), 1e-12)) if obs_w else 0.0
    trunc_frac = sum(1 for pr, pt in zip(p_raw, p) if abs(pr - pt) > 1e-12) / n
    diagnostics = {
        "p_hat_min": min(obs_p) if obs_p else None,
        "p_hat_med": quantile(obs_p, 0.5) if obs_p else None,
        "p_hat_max": max(obs_p) if obs_p else None,
        "w_max": max(obs_w) if obs_w else None,
        "ess": ess,
        "trunc_frac": trunc_frac,
    }

    return est, se, diagnostics


def estimate_true_effect(beta_treat, beta_int=0.28, n_mc=2_000_000, seed=90421):
    rng = random.Random(seed + int((beta_treat + 2) * 10000) + int((beta_int + 2) * 10000))
    s1 = 0.0
    s0 = 0.0
    for _ in range(n_mc):
        age, comorb, sev, biom = sample_covariates(rng)
        s1 += outcome_prob(age, comorb, sev, biom, 1, beta_treat, beta_int=beta_int)
        s0 += outcome_prob(age, comorb, sev, biom, 0, beta_treat, beta_int=beta_int)
    p1 = s1 / n_mc
    p0 = s0 / n_mc
    return logit(p1) - logit(p0)


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["method"])].append(r)

    out = []
    for (_, _), rr in sorted(by.items()):
        first = rr[0]
        true = first["true_effect"]
        vals = [x["estimate"] for x in rr]
        ses = [x["se"] for x in rr]
        n = len(vals)

        errs = [v - true for v in vals]
        sq = [e * e for e in errs]

        bias = mean(errs)
        rmse = math.sqrt(mean(sq))

        mcse_bias = math.sqrt(max(sample_var(errs), 0.0) / n)
        if rmse > 0:
            mcse_rmse = math.sqrt(max(sample_var(sq), 0.0) / n) / (2.0 * rmse)
        else:
            mcse_rmse = 0.0

        cover_ind = []
        rej_ind = []
        for est, se in zip(vals, ses):
            se = max(se, 1e-12)
            lo, hi = est - 1.96 * se, est + 1.96 * se
            cover_ind.append(1 if lo <= true <= hi else 0)
            if first["null_case"]:
                rej_ind.append(1 if abs(est / se) > 1.96 else 0)

        coverage = mean(cover_ind)
        mcse_coverage = math.sqrt(max(coverage * (1 - coverage), 0.0) / n)

        type1 = None
        mcse_type1 = None
        if first["null_case"]:
            type1 = mean(rej_ind)
            mcse_type1 = math.sqrt(max(type1 * (1 - type1), 0.0) / n)

        obs_rate = mean([x["obs_rate"] for x in rr])

        # Weight diagnostics (available for IPW/AIPW rows)
        def avg_or_none(key):
            vals_k = [x[key] for x in rr if x.get(key) is not None]
            return mean(vals_k) if vals_k else None

        out.append(
            {
                "scenario": first["scenario"],
                "n_sample": first["n_sample"],
                "mechanism": first["mechanism"],
                "condition": first["condition"],
                "null_case": first["null_case"],
                "beta_treat": first["beta_treat"],
                "beta_int": first["beta_int"],
                "method": first["method"],
                "n_reps": n,
                "true_effect": true,
                "bias": bias,
                "mcse_bias": mcse_bias,
                "rmse": rmse,
                "mcse_rmse": mcse_rmse,
                "coverage": coverage,
                "mcse_coverage": mcse_coverage,
                "type1": type1,
                "mcse_type1": mcse_type1,
                "obs_rate": obs_rate,
                "p_hat_min": avg_or_none("p_hat_min"),
                "p_hat_med": avg_or_none("p_hat_med"),
                "p_hat_max": avg_or_none("p_hat_max"),
                "w_max": avg_or_none("w_max"),
                "ess": avg_or_none("ess"),
                "trunc_frac": avg_or_none("trunc_frac"),
            }
        )
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_scenarios():
    scenarios = []

    n_values = [350, 700]
    # (name, gamma for Y main effect, gamma_ay for A*Y interaction in missingness)
    # Added stronger MNAR variants to ensure scenarios where CC bias is clearly visible.
    mechanisms = [
        ("MAR", 0.0, 0.0),
        ("MNARmod", 0.6, 0.20),
        ("MNARstrong", 1.0, 0.35),
        ("MNARyOnly", 1.0, 0.0),
        ("MNARintStrong", 0.6, 0.8),
    ]
    conditions = [
        ("correct", False, False),
        ("outcome_misspec", True, False),
        ("missing_misspec", False, True),
        ("both_misspec", True, True),
    ]

    for n in n_values:
        for mech_name, gamma, gamma_ay in mechanisms:
            for cond_name, mo, mm in conditions:
                scenarios.append(
                    Scenario(
                        name=f"N{n}_{mech_name}_{cond_name}_ALT",
                        n=n,
                        mechanism=mech_name,
                        condition=cond_name,
                        null_case=False,
                        mnar_gamma=gamma,
                        mnar_ay=gamma_ay,
                        misspec_outcome=mo,
                        misspec_missing=mm,
                        beta_treat=0.45,
                        beta_int=0.28,
                    )
                )
                scenarios.append(
                    Scenario(
                        name=f"N{n}_{mech_name}_{cond_name}_NULL",
                        n=n,
                        mechanism=mech_name,
                        condition=cond_name,
                        null_case=True,
                        mnar_gamma=gamma,
                        mnar_ay=gamma_ay,
                        misspec_outcome=mo,
                        misspec_missing=mm,
                        beta_treat=0.0,
                        beta_int=0.0,
                    )
                )

    return scenarios


def scenario_seed(global_seed, scenario_name):
    h = hashlib.sha256(f"{global_seed}|{scenario_name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def run_scenario_worker(args):
    scenario, n_reps, global_seed, true_map = args
    seed0 = scenario_seed(global_seed, scenario.name)
    rng_master = random.Random(seed0)

    rows = []
    for rep in range(n_reps):
        attempt = 0
        while True:
            attempt += 1
            rng = random.Random(rng_master.randint(1, 10**12))
            data = generate_dataset(
                scenario.n,
                scenario.beta_treat,
                scenario.beta_int,
                scenario.mnar_gamma,
                scenario.mnar_ay,
                rng,
            )
            obs_rate = sum(r["R"] for r in data) / len(data)
            true_eff = true_map[(scenario.beta_treat, scenario.beta_int)]

            try:
                cc_est, cc_se = estimate_cc(data)
                om_est, om_se = estimate_om(data, scenario.misspec_outcome)
                ipw_est, ipw_se, ipw_diag = estimate_ipw(data, scenario.misspec_missing)
                aipw_est, aipw_se, aipw_diag = estimate_aipw(data, scenario.misspec_outcome, scenario.misspec_missing)
                break
            except Exception:
                if attempt >= 5:
                    raise

        common = {
            "scenario": scenario.name,
            "n_sample": scenario.n,
            "mechanism": scenario.mechanism,
            "condition": scenario.condition,
            "null_case": scenario.null_case,
            "beta_treat": scenario.beta_treat,
            "beta_int": scenario.beta_int,
            "rep": rep,
            "true_effect": true_eff,
            "obs_rate": obs_rate,
        }

        rows.append(
            {
                **common,
                "method": "CC",
                "estimate": cc_est,
                "se": cc_se,
                "p_hat_min": None,
                "p_hat_med": None,
                "p_hat_max": None,
                "w_max": None,
                "ess": None,
                "trunc_frac": None,
            }
        )
        rows.append(
            {
                **common,
                "method": "OM",
                "estimate": om_est,
                "se": om_se,
                "p_hat_min": None,
                "p_hat_med": None,
                "p_hat_max": None,
                "w_max": None,
                "ess": None,
                "trunc_frac": None,
            }
        )
        rows.append(
            {
                **common,
                "method": "IPW",
                "estimate": ipw_est,
                "se": ipw_se,
                "p_hat_min": ipw_diag["p_hat_min"],
                "p_hat_med": ipw_diag["p_hat_med"],
                "p_hat_max": ipw_diag["p_hat_max"],
                "w_max": ipw_diag["w_max"],
                "ess": ipw_diag["ess"],
                "trunc_frac": ipw_diag["trunc_frac"],
            }
        )
        rows.append(
            {
                **common,
                "method": "AIPW",
                "estimate": aipw_est,
                "se": aipw_se,
                "p_hat_min": aipw_diag["p_hat_min"],
                "p_hat_med": aipw_diag["p_hat_med"],
                "p_hat_max": aipw_diag["p_hat_max"],
                "w_max": aipw_diag["w_max"],
                "ess": aipw_diag["ess"],
                "trunc_frac": aipw_diag["trunc_frac"],
            }
        )

    return rows


def main():
    ap = argparse.ArgumentParser(description="Run extended medical-statistics missing-data simulation study")
    ap.add_argument("--n-reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260221)
    ap.add_argument("--workers", type=int, default=0, help="0 means use all available CPU cores")
    ap.add_argument("--mp-context", choices=["auto", "fork", "spawn", "forkserver"], default="auto", help="multiprocessing start method")
    ap.add_argument("--chunksize", type=int, default=0, help="imap_unordered chunksize; 0 chooses an automatic value")
    ap.add_argument("--true-mc", type=int, default=2_000_000, help="Monte Carlo draws for true-effect integration")
    ap.add_argument("--out-raw", default="results/simulation_raw.csv")
    ap.add_argument("--out-summary", default="results/summary_metrics.csv")
    ap.add_argument("--out-true", default="results/true_effects.csv")
    args = ap.parse_args()

    scenarios = build_scenarios()

    true_map = {}
    for sc in scenarios:
        key = (sc.beta_treat, sc.beta_int)
        if key not in true_map:
            true_map[key] = estimate_true_effect(sc.beta_treat, beta_int=sc.beta_int, n_mc=args.true_mc, seed=args.seed + 1)

    write_csv(
        args.out_true,
        [
            {"beta_treat": k[0], "beta_int": k[1], "true_effect": v}
            for k, v in sorted(true_map.items())
        ],
        ["beta_treat", "beta_int", "true_effect"],
    )

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(scenarios)))

    if args.mp_context == "auto":
        mp_context = "fork" if os.name == "posix" else "spawn"
    else:
        mp_context = args.mp_context

    worker_args = [(s, args.n_reps, args.seed, true_map) for s in scenarios]
    chunksize = args.chunksize if args.chunksize > 0 else max(1, len(worker_args) // max(1, workers * 4))

    print(
        f"Running {len(scenarios)} scenarios x {args.n_reps} reps using {workers} worker(s) "
        f"with context='{mp_context}', chunksize={chunksize}"
    )

    rows_out = []
    with mp.get_context(mp_context).Pool(processes=workers) as pool:
        completed = 0
        for part in pool.imap_unordered(run_scenario_worker, worker_args, chunksize=chunksize):
            rows_out.extend(part)
            completed += 1
            print(f"Completed scenarios: {completed}/{len(worker_args)}")

    method_order = {"CC": 0, "OM": 1, "IPW": 2, "AIPW": 3}
    rows_out.sort(key=lambda r: (r["scenario"], r["rep"], method_order[r["method"]]))

    write_csv(
        args.out_raw,
        rows_out,
        [
            "scenario",
            "n_sample",
            "mechanism",
            "condition",
            "null_case",
            "beta_treat",
            "beta_int",
            "rep",
            "method",
            "estimate",
            "se",
            "true_effect",
            "obs_rate",
            "p_hat_min",
            "p_hat_med",
            "p_hat_max",
            "w_max",
            "ess",
            "trunc_frac",
        ],
    )

    summary = summarize(rows_out)
    summary.sort(key=lambda r: (r["scenario"], method_order[r["method"]]))

    write_csv(
        args.out_summary,
        summary,
        [
            "scenario",
            "n_sample",
            "mechanism",
            "condition",
            "null_case",
            "beta_treat",
            "beta_int",
            "method",
            "n_reps",
            "true_effect",
            "bias",
            "mcse_bias",
            "rmse",
            "mcse_rmse",
            "coverage",
            "mcse_coverage",
            "type1",
            "mcse_type1",
            "obs_rate",
            "p_hat_min",
            "p_hat_med",
            "p_hat_max",
            "w_max",
            "ess",
            "trunc_frac",
        ],
    )

    print(f"Wrote raw results: {args.out_raw} ({len(rows_out)} rows)")
    print(f"Wrote summary: {args.out_summary} ({len(summary)} rows)")
    print(f"Wrote true effects: {args.out_true}")


if __name__ == "__main__":
    main()
