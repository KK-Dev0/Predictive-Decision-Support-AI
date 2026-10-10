import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CreditSense AI", page_icon="💳", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 900px;}
.hero {background: linear-gradient(135deg, #0F766E 0%, #2563EB 55%, #7C3AED 100%);
       padding: 2rem 2rem 1.5rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.1rem; color: white; padding: 0;}
.hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.02rem; color: white;}
.chip {display: inline-block; padding: .18rem .75rem; border-radius: 999px;
       background: rgba(255,255,255,.2); font-size: .78rem; margin: .8rem .35rem 0 0; color: white;}
.result {border-radius: 16px; padding: 1.1rem 1.3rem; margin: .8rem 0; border: 1px solid;}
.result .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .75;}
.result .big {font-size: 2.3rem; font-weight: 800; line-height: 1.2;}
.result .band {font-size: 1.2rem; font-weight: 700;}
.ok {border-color: rgba(34,197,94,.6); background: rgba(34,197,94,.12);}
.mid {border-color: rgba(234,179,8,.6); background: rgba(234,179,8,.12);}
.bad {border-color: rgba(239,68,68,.6); background: rgba(239,68,68,.12);}
</style>
<div class="hero">
  <h1>💳 CreditSense AI</h1>
  <p>An explainable decision-support tool that estimates a borrower's chance of serious delinquency within two years, and turns it into a cost-aware recommendation.</p>
  <span class="chip">Trained on 150,000 real borrowers</span><span class="chip">Explainable</span>
  <span class="chip">Fairness checked</span><span class="chip">Decision support, not auto-approval</span>
</div>
""",
    unsafe_allow_html=True,
)

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

LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit card utilization (balance / limit)",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 days late",
    "DebtRatio": "Debt ratio (monthly debt / income)",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines and loans",
    "NumberOfTimes90DaysLate": "Times 90+ days late",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 days late",
    "NumberOfDependents": "Number of dependents",
}
SENSITIVE = {"age"}


@st.cache_resource
def load_model():
    return joblib.load("credit_model.joblib")


def pretty(c):
    return LABELS.get(c, c)


def factor_effects(row, meta, pipe):
    base_p = pipe.predict_proba(add_features(row, meta["caps"]))[0, 1]
    effects = {}
    for c in meta["raw_cols"]:
        alt = row.copy()
        alt[c] = meta["ui_stats"][c]["median"]
        effects[c] = base_p - pipe.predict_proba(add_features(alt, meta["caps"]))[0, 1]
    return pd.Series(effects).sort_values()


meta = load_model()
pipe = meta["pipeline"]
t_star, t_review = meta["t_star"], meta["t_review"]

tab1, tab2, tab3, tab4 = st.tabs(["🧮 Assess borrower", "📊 Model performance", "⚖️ Fairness and limits", "ℹ️ About"])

with tab1:
    st.caption("Educational decision-support prototype built on public data. It is not a real lending system.")
    with st.form("borrower"):
        left, right = st.columns(2)
        values = {}
        for i, c in enumerate(meta["raw_cols"]):
            box = left if i % 2 == 0 else right
            s = meta["ui_stats"][c]
            label = pretty(c) + (" (sensitive)" if c in SENSITIVE else "")
            values[c] = box.number_input(label, min_value=float(s["min"]), max_value=float(s["max"]),
                                         value=float(s["median"]), step=float(s["step"]), key="in_" + c)
        unk1, unk2 = st.columns(2)
        income_unknown = unk1.checkbox("Monthly income unknown", key="unk_income")
        dep_unknown = unk2.checkbox("Number of dependents unknown", key="unk_dep")
        go = st.form_submit_button("Assess borrower", type="primary")

    if go:
        if income_unknown:
            values["MonthlyIncome"] = np.nan
        if dep_unknown:
            values["NumberOfDependents"] = np.nan
        row = pd.DataFrame([values])[meta["raw_cols"]]
        p = float(pipe.predict_proba(add_features(row, meta["caps"]))[0, 1])
        if p >= t_star:
            css, band = "bad", "🔴 Likely decline"
        elif p >= t_review:
            css, band = "mid", "🟡 Manual review recommended"
        else:
            css, band = "ok", "✅ Likely approve"
        st.markdown(
            f'<div class="result {css}"><div class="label">Estimated probability of serious delinquency within 2 years</div>'
            f'<div class="big">{p * 100:.1f}%</div><div class="band">{band}</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(p / max(2 * t_star, 1e-9), 0.0), 1.0))
        st.caption(
            f"For context, {meta['base_rate'] * 100:.1f}% of borrowers in the training data became seriously delinquent. "
            f"Bands are illustrative: decline at {t_star * 100:.1f}% or higher, review from {t_review * 100:.1f}%. "
            "The decline threshold minimizes an assumed cost in which approving a borrower who later defaults is "
            "5 times worse than declining a good one. A real lender would set this with policy and real cost data."
        )

        eff = factor_effects(row, meta, pipe)
        raising = eff[eff > 0.002][::-1].head(4)
        lowering = eff[eff < -0.002].head(4)
        st.subheader("Why this score?")
        st.caption("Each input is swapped for a typical value (the median) to see how much it moves the estimate.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raising the risk**")
            if len(raising):
                for c, v in raising.items():
                    st.write(f"{pretty(c)}: +{v * 100:.1f} pts" + (" ⚠️" if c in SENSITIVE else ""))
            else:
                st.write("No single input stands out.")
        with c2:
            st.markdown("**Lowering the risk**")
            if len(lowering):
                for c, v in lowering.items():
                    st.write(f"{pretty(c)}: -{abs(v) * 100:.1f} pts" + (" ⚠️" if c in SENSITIVE else ""))
            else:
                st.write("No single input stands out.")
        st.caption("⚠️ marks a sensitive attribute. See the Fairness tab.")

with tab2:
    tm, pol = meta["test_metrics"], meta["policy"]
    st.caption(f"Selected model: {meta['model_name']} | trained on {meta['n_train']:,} borrowers, tested on {meta['n_test']:,}")
    a, b, c = st.columns(3)
    a.metric("ROC-AUC", f"{tm['ROC-AUC']:.3f}")
    b.metric("PR-AUC", f"{tm['PR-AUC']:.3f}")
    c.metric("Brier score", f"{tm['Brier']:.4f}")
    st.subheader("What the decision rule achieves on unseen borrowers")
    d, e, f = st.columns(3)
    d.metric("Declined", f"{pol['Decline rate'] * 100:.1f}%")
    e.metric("Default rate if approved", f"{pol['Default rate among approved'] * 100:.1f}%")
    f.metric("Defaulters caught", f"{pol['Share of all defaulters caught (recall)'] * 100:.0f}%")
    st.caption(f"Overall default rate in the test data: {meta['base_rate'] * 100:.1f}%. Accuracy is not shown because approving everyone would already be about {100 - meta['base_rate'] * 100:.0f}% accurate.")
    st.subheader("Model comparison (5-fold cross-validation)")
    st.dataframe(pd.DataFrame(meta["cv_table"]).set_index("Model"))
    st.subheader("What the model relies on most")
    st.bar_chart(pd.Series(meta["importance"]).sort_values())
    st.caption("Drop in ROC-AUC when each input is shuffled on the test set.")

with tab3:
    st.markdown(
        """
**Sensitive attribute.** Age is in the data and the model can use it. It is marked in the form and in the explanation. The data has no sex or race columns.

**What was checked.** Decline rates and error rates were compared across three age groups on unseen borrowers, with and without age in the model.
"""
    )
    st.markdown("**With age in the model**")
    st.dataframe(pd.DataFrame(meta["fair_with"]).set_index("Age group"))
    st.markdown("**With age removed**")
    st.dataframe(pd.DataFrame(meta["fair_without"]).set_index("Age group"))
    st.caption(f"Test ROC-AUC: {meta['auc_with_age']:.4f} with age, {meta['auc_without_age']:.4f} without.")
    st.markdown(
        """
**Honest limits**
- Removing age does not remove age bias, because other inputs (income, number of credit lines) can act as proxies
- The decline threshold rests on an assumed 5:1 cost ratio, not real lender data
- Real lending needs legal review, richer data and human oversight
- This tool should never make a final decision on its own
"""
    )

with tab4:
    st.markdown(
        """
**Data:** Kaggle competition *Give Me Some Credit* (`cs-training.csv`), 150,000 real borrowers, 10 inputs, target: serious delinquency (90+ days past due) within two years. Source: https://www.kaggle.com/c/GiveMeSomeCredit

**Pipeline:** cleaning of impossible values and special codes, feature engineering, imputing with missing-value flags, ANOVA feature selection, then a tuned model chosen by cross-validated ROC-AUC.

**Limitations:** the data is from the US around 2011 and is anonymized, so results will not carry over directly to other countries or time periods. Probabilities are model estimates, not guarantees.

Built by **Kaushikee Karmakar** | SparkIIT AI Internship Project 2
"""
    )
