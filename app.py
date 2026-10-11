import math
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
META = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_model.joblib"))
PIPE = META["pipeline"]

PAST_DUE = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]


def add_features(d, caps):
    d = d.copy()
    for c in PAST_DUE:
        d[c] = d[c].where(d[c] < 90)   # codes 96 and 98 are data errors, treated as missing
    d["RevolvingUtilizationOfUnsecuredLines"] = d["RevolvingUtilizationOfUnsecuredLines"].clip(upper=caps["util"])
    d["DebtRatio"] = d["DebtRatio"].clip(upper=caps["debt"])
    d["MonthlyIncome"] = d["MonthlyIncome"].clip(upper=caps["income"])
    d["total_past_due"] = d[PAST_DUE].sum(axis=1, min_count=1)
    d["ever_late"] = (d["total_past_due"].fillna(0) > 0).astype(int)
    d["log_monthly_income"] = np.log1p(d["MonthlyIncome"])
    d["log_debt_ratio"] = np.log1p(d["DebtRatio"])
    d["income_per_dependent"] = d["MonthlyIncome"] / (d["NumberOfDependents"] + 1)
    return d

# Cost figures printed by the training notebook (assumed 5:1 cost of a missed defaulter vs a wrongly declined good borrower).
REPORTED_COST = {"approve_all": 0.3342, "decline_all": 0.9332, "model": 0.2107}

# key: (label, hint, kind, suffix)
SPEC = {
    "age": ("Age", "Age is a sensitive attribute. See the fairness section.", "int", "years"),
    "MonthlyIncome": ("Monthly income", "Before tax. Tick 'Not known' if it was not reported.", "money", "per month"),
    "NumberOfDependents": ("Dependents", "People who rely on the borrower's income.", "int", ""),
    "RevolvingUtilizationOfUnsecuredLines": ("Credit utilization", "Card and credit-line balances as a share of total limits.", "percent", "%"),
    "DebtRatio": ("Debt ratio", "Monthly debt payments as a share of monthly income.", "percent", "%"),
    "NumberOfOpenCreditLinesAndLoans": ("Open credit lines and loans", "Loans plus credit cards currently open.", "int", ""),
    "NumberRealEstateLoansOrLines": ("Real estate loans", "Mortgages and home equity lines.", "int", ""),
    "NumberOfTime30-59DaysPastDueNotWorse": ("Payments 30 to 59 days late", "Times in the last two years.", "int", "times"),
    "NumberOfTime60-89DaysPastDueNotWorse": ("Payments 60 to 89 days late", "Times in the last two years.", "int", "times"),
    "NumberOfTimes90DaysLate": ("Payments 90 or more days late", "Times in the last two years.", "int", "times"),
}
GROUPS = [
    ("Borrower", ["age", "MonthlyIncome", "NumberOfDependents"]),
    ("Credit usage", ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]),
    ("Payment history", ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]),
]
OPTIONAL = {"MonthlyIncome", "NumberOfDependents"}
SENSITIVE = {"age"}


def label(c):
    return SPEC[c][0]


def build_groups():
    out = []
    for title, keys in GROUPS:
        fields = []
        for k in keys:
            lab, hint, kind, suffix = SPEC[k]
            st = META["ui_stats"][k]
            med = st["median"]
            if kind == "percent":
                default, step = f"{med * 100:.0f}", "any"
            elif kind == "money":
                default, step = f"{med:.0f}", "any"
            else:
                default, step = f"{med:.0f}", "1"
            fields.append({"key": k, "label": lab, "hint": hint, "kind": kind, "suffix": suffix,
                           "default": default, "step": step, "min": "18" if k == "age" else "0",
                           "unknown": k in OPTIONAL})
        out.append({"title": title, "fields": fields})
    return out


def explain(df):
    """Swap each input for the typical (median) value, all in one batch, and measure the change in risk."""
    cols = META["raw_cols"]
    frames = [df]
    for c in cols:
        alt = df.copy()
        alt[c] = META["ui_stats"][c]["median"]
        frames.append(alt)
    big = pd.concat(frames, ignore_index=True)
    probs = PIPE.predict_proba(add_features(big, META["caps"]))[:, 1]
    return float(probs[0]), {c: float(probs[0] - probs[i + 1]) for i, c in enumerate(cols)}


def fairness_view():
    w = {r["Age group"]: r for r in META["fair_with"]}
    wo = {r["Age group"]: r for r in META["fair_without"]}
    order = ["under 30", "30 to 59", "60 and over"]
    rows = [{"group": g, "n": w[g]["n"], "actual": w[g]["Actual default rate"], "pred": w[g]["Mean predicted risk"],
             "dec_with": w[g]["Declined"], "dec_without": wo[g]["Declined"],
             "fd_with": w[g]["Declined among non-defaulters"], "fd_without": wo[g]["Declined among non-defaulters"]}
            for g in order]
    young, old = rows[0], rows[2]
    ratio = young["fd_with"] / old["fd_with"] if old["fd_with"] else None
    return rows, {"ratio": ratio, "young_fd": young["fd_with"], "old_fd": old["fd_with"],
                  "auc_drop": META["auc_with_age"] - META["auc_without_age"]}


@app.template_filter("pct")
def pct(v, digits=1):
    return f"{v * 100:.{digits}f}%"


@app.template_filter("pct0")
def pct0(v):
    return f"{v * 100:.0f}%"


@app.get("/")
def index():
    fair_rows, fair_note = fairness_view()
    t_star, t_review = META["t_star"], META["t_review"]
    cost_gain = 1 - REPORTED_COST["model"] / min(REPORTED_COST["approve_all"], REPORTED_COST["decline_all"])
    imp = sorted(META["importance"].items(), key=lambda kv: -kv[1])
    imp_max = max(v for _, v in imp) or 1.0
    config = {"t_star": float(t_star), "t_review": float(t_review), "base_rate": float(META["base_rate"]),
              "axis_max": float(max(0.4, round(t_star * 2.5, 2)))}
    return render_template(
        "index.html", groups=build_groups(), config=config, meta=META, fair_rows=fair_rows, fair_note=fair_note,
        importance=[(label(k), (0.0 if abs(v) < 0.0005 else v), max(v, 0.0) / imp_max) for k, v in imp], cost=REPORTED_COST, cost_gain=cost_gain,
        t_star=t_star, t_review=t_review)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    row = {}
    for c in META["raw_cols"]:
        v = data.get(c)
        if v is None:
            if c in OPTIONAL:
                row[c] = np.nan
                continue
            return jsonify(error=f"Please enter a value for {label(c)}."), 400
        try:
            x = float(v)
        except (TypeError, ValueError):
            return jsonify(error=f"{label(c)} must be a number."), 400
        if not math.isfinite(x) or x < 0 or x > 1e7:
            return jsonify(error=f"{label(c)} is outside the supported range."), 400
        row[c] = x
    if not 18 <= row["age"] <= 110:
        return jsonify(error="Age must be between 18 and 110."), 400

    df = pd.DataFrame([row])[META["raw_cols"]]
    p, effects = explain(df)
    t_star, t_review = META["t_star"], META["t_review"]
    band = "decline" if p >= t_star else "review" if p >= t_review else "approve"

    def items(pairs):
        return [{"label": label(c), "points": v * 100, "sensitive": c in SENSITIVE} for c, v in pairs]

    raising = sorted([kv for kv in effects.items() if kv[1] > 0.002], key=lambda kv: -kv[1])[:4]
    lowering = sorted([kv for kv in effects.items() if kv[1] < -0.002], key=lambda kv: kv[1])[:4]
    return jsonify(probability=p, band=band, base_rate=META["base_rate"],
                   raising=items(raising), lowering=items(lowering))


@app.get("/healthz")
def healthz():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
