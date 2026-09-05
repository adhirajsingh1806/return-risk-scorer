import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import json

st.set_page_config(page_title="Order Risk Assessment", page_icon=None, layout="centered")

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .risk-card {
        padding: 1.5rem;
        border-radius: 6px;
        text-align: center;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .risk-low { background-color: #f0f5f1; border-left-color: #2e7d4f; }
    .risk-medium { background-color: #f7f2e8; border-left-color: #b8860b; }
    .risk-high { background-color: #f7ecec; border-left-color: #b23b3b; }
    .risk-review { background-color: #eef1f5; border-left-color: #4a5a8a; }
    .risk-card, .risk-card * { color: #1a1a1a !important; }
    .risk-label { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.3rem; letter-spacing: 0.02em; }
    .risk-sub { font-size: 0.95rem; }
    .factor-pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .pill-up { background-color: #f7ecec; color: #a13a3a !important; }
    .pill-down { background-color: #eef4ef; color: #2e6b4a !important; }
    .computed-value {
        padding: 0.6rem 0.9rem;
        background-color: rgba(128,128,128,0.08);
        border-radius: 6px;
        font-size: 0.9rem;
        margin-top: -0.5rem;
        margin-bottom: 0.8rem;
    }
    .defense-banner {
        padding: 0.7rem 1rem;
        background-color: #f4f4f4;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        color: #333333;
    }
    .estimated-note {
        padding: 0.5rem 0.8rem;
        background-color: #fbf7ee;
        border: 1px solid #e6dcc5;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #5a4d33;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("risk_model.pkl")
        explainer = joblib.load("shap_explainer.pkl")
        with open("feature_config.json") as f:
            feature_config = json.load(f)
        with open("cat_options.json") as f:
            cat_options = json.load(f)
        with open("num_ranges.json") as f:
            num_ranges = json.load(f)
        with open("threshold_config.json") as f:
            threshold_config = json.load(f)
        with open("model_metrics.json") as f:
            model_metrics = json.load(f)
        with open("threshold_sweep.json") as f:
            threshold_sweep = json.load(f)
        with open("audit_sample.json") as f:
            audit_sample = json.load(f)
        with open("cost_scenarios.json") as f:
            cost_scenarios = json.load(f)
        return (model, explainer, feature_config, cat_options, num_ranges, threshold_config,
                model_metrics, threshold_sweep, audit_sample, cost_scenarios, None)
    except FileNotFoundError as e:
        return None, None, None, None, None, None, None, None, None, str(e)

(model, explainer, feature_config, cat_options, num_ranges, threshold_config,
 model_metrics, threshold_sweep, audit_sample, cost_scenarios, load_error) = load_artifacts()

if load_error:
    st.error(
        f"Missing required file: {load_error}. "
        "Run save_artifacts_snippet.py first, then restart this app."
    )
    st.stop()

num_features = feature_config["num_features"]
cat_features = feature_config["cat_features"]
sweep_df = pd.DataFrame(threshold_sweep)

DIRECT_NUM_FEATURES = ["price", "freight_value", "payment_installments",
                        "product_weight_g", "delay_days", "purchase_dow", "purchase_month"]
COMPUTED_FEATURES = ["freight_ratio", "freight_installment_interaction", "delay_freight_interaction"]
SELLER_FEATURES = ["seller_avg_delay", "seller_bad_rate", "seller_order_count"]

FIELD_INFO = {
    "price": {"label": "Order Price", "help": "Price of the product being purchased."},
    "freight_value": {"label": "Shipping Cost", "help": "Amount charged to the customer for shipping."},
    "payment_installments": {"label": "Number of Installments", "help": "Number of payments the order is split into."},
    "product_weight_g": {"label": "Product Weight (grams)", "help": "Heavier items carry more shipping risk."},
    "delay_days": {"label": "Delivery Delay (days)", "help": "Days late versus the promised delivery date. Negative means early."},
    "purchase_dow": {"label": "Day of Week Purchased", "help": "0 = Monday, 6 = Sunday."},
    "purchase_month": {"label": "Month Purchased", "help": "1 = January, 12 = December."},
    "payment_type": {"label": "Payment Method", "help": "How the customer is paying."},
    "product_category_name": {"label": "Product Category", "help": "Category the product belongs to."},
    "customer_state": {"label": "Customer Region", "help": "Where the customer is located."},
    "seller_avg_delay": {"label": "Seller's Average Delay (days)", "help": "This seller's typical delivery delay across their order history."},
    "seller_bad_rate": {"label": "Seller's Historical Return Rate", "help": "Share of this seller's past orders that were returned, cancelled, or led to a dissatisfied customer (0 to 1)."},
    "seller_order_count": {"label": "Seller's Order Count", "help": "How many orders this seller has fulfilled historically."},
}

FIELD_EXPLANATIONS = {
    "price": {
        "up": "Higher-value orders tend to draw more scrutiny, and buyer's remorse is more common on larger purchases.",
        "down": "Lower-value orders are lower-stakes for the customer, making a return or complaint less likely.",
    },
    "freight_value": {
        "up": "High shipping costs can make the total order feel overpriced, increasing dissatisfaction.",
        "down": "Low shipping costs are less likely to leave a customer feeling they overpaid.",
    },
    "freight_ratio": {
        "up": "When shipping cost makes up a large share of the item price, customers are more likely to feel the total wasn't worth it.",
        "down": "A small shipping-to-price ratio is a normal cost structure that doesn't tend to trigger complaints.",
    },
    "freight_installment_interaction": {
        "up": "A high shipping-cost ratio combined with many payment installments is a compounding signal — cost-sensitive customers paying extra for shipping.",
        "down": "This combination of shipping cost and payment plan doesn't show the same compounding effect.",
    },
    "delay_freight_interaction": {
        "up": "A delayed order that also had a high shipping cost tends to compound customer dissatisfaction more than either factor alone.",
        "down": "This combination of delivery timing and shipping cost isn't adding extra risk beyond each factor individually.",
    },
    "payment_installments": {
        "up": "Orders paid across many installments can signal financial strain, which correlates with higher cancellation rates.",
        "down": "Fewer installments (or full payment) doesn't carry this added financial-strain signal.",
    },
    "product_weight_g": {
        "up": "Heavier, bulkier products are more prone to shipping damage and delivery complications.",
        "down": "Lighter products ship more reliably, lowering damage-related return risk.",
    },
    "delay_days": {
        "up": "Late delivery is one of the strongest known drivers of dissatisfaction and returns.",
        "down": "On-time or early delivery is one of the strongest positive signals for a satisfied customer.",
    },
    "purchase_dow": {
        "up": "Orders placed on this day of the week have historically shown a higher return rate in this dataset.",
        "down": "Orders placed on this day of the week have historically shown a lower return rate in this dataset.",
    },
    "purchase_month": {
        "up": "This time of year has historically seen a higher return rate (seasonal demand or gifting periods).",
        "down": "This time of year has historically seen a lower return rate in this dataset.",
    },
    "payment_type": {
        "up": "This payment method has historically been associated with a higher rate of cancellations.",
        "down": "This payment method has historically been associated with more reliable transactions.",
    },
    "product_category_name": {
        "up": "This category has historically seen a higher-than-average return rate, often due to fit, expectations, or fragility.",
        "down": "This category has historically seen a lower-than-average return rate.",
    },
    "customer_state": {
        "up": "Orders shipped to this region have historically faced more delivery delays or complications.",
        "down": "Orders shipped to this region have historically had more reliable delivery.",
    },
    "seller_avg_delay": {
        "up": "A seller whose orders typically run late adds risk independent of this specific order's delay.",
        "down": "A seller with a reliable delivery history reduces risk beyond this specific order's own timing.",
    },
    "seller_bad_rate": {
        "up": "This seller has a higher-than-average historical rate of returns or dissatisfied customers.",
        "down": "This seller has a track record of low return and cancellation rates.",
    },
    "seller_order_count": {
        "up": "A seller with a shorter order history gives the model less to go on, which can push risk estimates in either direction.",
        "down": "A seller with a long order history gives the model a more reliable track record to draw on.",
    },
}

def field_label(feat):
    return FIELD_INFO.get(feat, {}).get("label", feat.replace("_", " ").title())

def field_help(feat):
    return FIELD_INFO.get(feat, {}).get("help", "")

def field_explanation(feat, direction):
    return FIELD_EXPLANATIONS.get(feat, {}).get(
        direction, "This factor shifted the score based on patterns in the historical order data."
    )

def match_source_feature(raw_name):
    cleaned = raw_name.split("__")[-1]
    for feat in num_features + cat_features:
        if cleaned.startswith(feat):
            return feat
    return None

def top_contributions(input_df, top_n=3):
    """Returns the top_n features driving this prediction using SHAP values,
    as (source_feat, display_name, direction)."""
    preprocessor = model.named_steps["prep"]
    transformed_input = preprocessor.transform(input_df)
    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    shap_values = explainer.shap_values(transformed_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    row_values = shap_values[0]

    feature_names = preprocessor.get_feature_names_out()
    contributions = pd.DataFrame({"feature": feature_names, "contribution": row_values})
    contributions = contributions.reindex(
        contributions["contribution"].abs().sort_values(ascending=False).index
    ).head(top_n)

    results = []
    for _, row in contributions.iterrows():
        source_feat = match_source_feature(row["feature"])
        display_name = field_label(source_feat) if source_feat else row["feature"].replace("_", " ").title()
        direction = "up" if row["contribution"] > 0 else "down"
        results.append((source_feat, display_name, direction))
    return results

def decide(prob, threshold, band=0.05):
    """Orders whose score sits within `band` of the decision threshold are
    routed to manual review rather than forced into an automatic decision."""
    if abs(prob - threshold) <= band:
        return "Manual Review", "risk-review", (
            "This order's score is close to the decision boundary. Rather than force an "
            "automatic call, it's routed to manual review."
        )
    if prob < 0.35:
        return "Auto-Approve", "risk-low", "This order looks safe to fulfill as-is."
    if prob < threshold:
        return "Manual Review", "risk-medium", "Consider a light touch, such as a delivery confirmation message."
    return "Auto-Flag", "risk-high", (
        "Consider proactive action: an order confirmation call, extra packaging, "
        "or a return-policy reminder."
    )

#Header
st.title("Return Risk Scorer - Order Risk Assessment")
st.write(
    "Estimates the likelihood that an order will be returned, cancelled, "
    "or leave a dissatisfied customer, based on order details."
)

st.markdown(
    '<div class="defense-banner">'
    "This tool scores and flags risk only. It does not take automated action on orders, "
    "and it is not intended or able to be used to test, game, or reverse-engineer its own "
    "detection thresholds."
    "</div>",
    unsafe_allow_html=True,
)

tab_score, tab_metrics, tab_audit, tab_architecture = st.tabs(
    ["Score an Order", "Model Performance", "Audit Trail", "Architecture"]
)

#Tab 1: Score an Order
with tab_score:
    st.subheader("Order Details")

    user_input = {}

    def number_box(feat, key_prefix="score"):
        rng = num_ranges[feat]
        if feat in ("payment_installments", "product_weight_g", "purchase_dow", "purchase_month",
                    "delay_days", "seller_order_count"):
            step, fmt = 1.0, "%.0f"
        else:
            step, fmt = 0.01, "%.2f"
        return st.number_input(
            field_label(feat),
            min_value=round(rng["min"], 2),
            max_value=round(rng["max"], 2),
            value=round(rng["mean"], 2),
            step=step,
            format=fmt,
            help=f"{field_help(feat)} Typical range: {rng['min']:.0f}-{rng['max']:.0f}.",
            key=f"{key_prefix}_{feat}",
        )

    st.markdown("**Order & Payment**")
    c1, c2 = st.columns(2)
    payment_related = ["price", "freight_value", "payment_installments", "payment_type"]
    for i, feat in enumerate(payment_related):
        col = c1 if i % 2 == 0 else c2
        with col:
            if feat in cat_features:
                user_input[feat] = st.selectbox(field_label(feat), cat_options[feat], help=field_help(feat), key=f"score_{feat}")
            else:
                user_input[feat] = number_box(feat)

    price_val = user_input.get("price", 0)
    freight_val = user_input.get("freight_value", 0)
    freight_ratio_val = freight_val / price_val if price_val else 0.0
    user_input["freight_ratio"] = freight_ratio_val

    st.markdown(
        f'<div class="computed-value">Shipping cost as a share of order price: '
        f'<b>{freight_ratio_val:.2f}</b> (calculated automatically from the two fields above)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Shipping & Product**")
    c3, c4 = st.columns(2)
    shipping_related = [f for f in DIRECT_NUM_FEATURES + cat_features if f not in payment_related]
    for i, feat in enumerate(shipping_related):
        col = c3 if i % 2 == 0 else c4
        with col:
            if feat in cat_features:
                user_input[feat] = st.selectbox(field_label(feat), cat_options[feat], help=field_help(feat), key=f"score_{feat}")
            else:
                user_input[feat] = number_box(feat)

    user_input["freight_installment_interaction"] = freight_ratio_val * user_input.get("payment_installments", 0)
    user_input["delay_freight_interaction"] = user_input.get("delay_days", 0) * freight_ratio_val

    st.markdown("**Seller History**")
    st.markdown(
        '<div class="estimated-note">'
        "These normally come from the seller's own order history and are looked up automatically, "
        "not entered by hand. Since this form has no real seller record behind it, they default to "
        "dataset averages — adjust them to see how a more or less established seller changes the score."
        "</div>",
        unsafe_allow_html=True,
    )
    c5, c6, c7 = st.columns(3)
    with c5:
        user_input["seller_avg_delay"] = number_box("seller_avg_delay")
    with c6:
        user_input["seller_bad_rate"] = st.number_input(
            field_label("seller_bad_rate"), min_value=0.0, max_value=1.0,
            value=round(num_ranges["seller_bad_rate"]["mean"], 3), step=0.01, format="%.3f",
            help=field_help("seller_bad_rate"), key="score_seller_bad_rate",
        )
    with c7:
        user_input["seller_order_count"] = number_box("seller_order_count")

    st.divider()

    if st.button("Check This Order", type="primary", use_container_width=True):

        input_df = pd.DataFrame([user_input])[num_features + cat_features]
        prob = float(model.predict_proba(input_df)[0, 1])
        threshold = threshold_config["threshold"]

        decision, css_class, message = decide(prob, threshold)

        st.subheader("Result")
        st.markdown(f"""
        <div class="risk-card {css_class}">
            <div class="risk-label">{decision}</div>
            <div class="risk-sub">Estimated probability of a return, cancellation, or dissatisfied customer: <b>{prob:.0%}</b></div>
            <div class="risk-sub" style="margin-top:0.5rem;">{message}</div>
        </div>
        """, unsafe_allow_html=True)

        gauge_color = {"Auto-Approve": "#2e7d4f", "Manual Review": "#b8860b", "Auto-Flag": "#b23b3b"}[decision]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 35], "color": "#f0f5f1"},
                    {"range": [35, threshold * 100], "color": "#f7f2e8"},
                    {"range": [threshold * 100, 100], "color": "#f7ecec"},
                ],
                "threshold": {"line": {"color": "#333333", "width": 3}, "thickness": 0.8, "value": threshold * 100},
            },
        ))
        fig.update_layout(height=230, margin=dict(t=10, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Why This Was Flagged" if decision != "Auto-Approve" else "Contributing Factors")
        try:
            for source_feat, display_name, direction in top_contributions(input_df, top_n=3):
                pill_class = "pill-up" if direction == "up" else "pill-down"
                pill_text = "Increases risk" if direction == "up" else "Decreases risk"
                with st.expander(display_name):
                    st.markdown(
                        f'{display_name} <span class="factor-pill {pill_class}">{pill_text}</span>',
                        unsafe_allow_html=True,
                    )
                    st.write(field_explanation(source_feat, direction))
                    if source_feat:
                        entered_value = user_input.get(source_feat)
                        if source_feat in num_features:
                            rng = num_ranges[source_feat]
                            st.caption(
                                f"Value: {entered_value}. "
                                f"Typical range: {rng['min']:.0f}-{rng['max']:.0f} (average {rng['mean']:.1f})."
                            )
                        else:
                            st.caption(f"Value selected: {entered_value}.")
        except Exception:
            st.caption("Factor breakdown unavailable for this input.")

        with st.expander("Model inputs used for this prediction"):
            st.write(f"Raw probability: {prob:.4f}")
            st.write(f"Decision threshold: {threshold} (selected to minimize total cost)")
            st.write(
                f"Threshold assumes a false positive (flagging a good order) costs "
                f"₹{threshold_config['cost_fp']}, and a false negative (missing a bad order) costs "
                f"₹{threshold_config['cost_fn']}."
            )
            st.json(user_input)

#Tab 2: Model Performance
with tab_metrics:
    st.subheader("Held-Out Test Set Performance")
    st.caption(
        f"All figures below are computed on a held-out test set of {model_metrics['test_set_size']:,} orders "
        f"({model_metrics['positive_rate']:.1%} positive rate) that the model never saw during training."
    )
 
    m1, m2 = st.columns(2)
    with m1:
        st.metric("AUC-ROC", f"{model_metrics['auc_roc']:.3f}")
    with m2:
        st.metric("AUC-PR", f"{model_metrics['auc_pr']:.3f}")
 
    st.divider()
    st.subheader("Cost Assumption Matters")
    st.write(
        "The true cost of a false positive (wrongly flagging a good order) versus a false "
        "negative (missing a bad one) isn't obvious. Ecommerce fraud-prevention research "
        "generally finds false declines cost merchants more than the fraud they prevent, "
        "which cuts against the intuition that missing a bad order is always worse. Rather "
        "than pick one number, three scenarios are compared below."
    )
 
    scenarios = cost_scenarios["scenarios"]
 
    def best_row_for(cost_fp, cost_fn):
        costs = sweep_df["fp"] * cost_fp + sweep_df["fn"] * cost_fn
        return sweep_df.loc[costs.idxmin()], int(costs.min())
 
    comparison_rows = []
    for key, s in scenarios.items():
        row, total_cost = best_row_for(s["cost_fp"], s["cost_fn"])
        comparison_rows.append({
            "Scenario": s["label"],
            "FP:FN cost ratio": f"{s['cost_fn']}:{s['cost_fp']}" if s["cost_fp"] > s["cost_fn"]
                                  else f"1:{round(s['cost_fn']/s['cost_fp'], 1)}",
            "Threshold": f"{row['threshold']:.2f}",
            "Precision": f"{row['precision']:.2f}",
            "Recall": f"{row['recall']:.2f}",
            "Est. cost": f"₹{total_cost:,}",
        })
    st.dataframe(pd.DataFrame(comparison_rows).set_index("Scenario"), use_container_width=True)
 
    st.caption(
        "The live scoring tab uses the industry-aligned (2:1) scenario's threshold by default. "
        "Explore any scenario in detail below."
    )
 
    scenario_key = st.selectbox(
        "Cost scenario", options=list(scenarios.keys()),
        format_func=lambda k: scenarios[k]["label"],
        index=list(scenarios.keys()).index(cost_scenarios["default_scenario"]),
    )
    selected = scenarios[scenario_key]
    cost_fp, cost_fn = selected["cost_fp"], selected["cost_fn"]
 
    slider_threshold = st.slider(
        "Decision threshold", min_value=0.05, max_value=0.95,
        value=round(selected["optimal_threshold"], 2), step=0.01,
    )
 
    closest_row = sweep_df.iloc[(sweep_df["threshold"] - slider_threshold).abs().idxmin()]
 
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Precision", f"{closest_row['precision']:.2f}")
    with s2:
        st.metric("Recall", f"{closest_row['recall']:.2f}")
    with s3:
        st.metric("F1 Score", f"{closest_row['f1']:.2f}")
 
    fp_count = int(closest_row["fp"])
    fn_count = int(closest_row["fn"])
    total_cost = fp_count * cost_fp + fn_count * cost_fn
    legit_flagged_pct = fp_count / (fp_count + int(closest_row["tn"])) if (fp_count + closest_row["tn"]) > 0 else 0
 
    st.write(
        f"At this threshold, under the **{selected['label']}** assumption: "
        f"**{legit_flagged_pct:.1%}** of legitimate orders are wrongly flagged "
        f"({fp_count} orders), costing an estimated **₹{fp_count * cost_fp:,}**. "
        f"**{fn_count}** risky orders are missed, costing an estimated **₹{fn_count * cost_fn:,}**. "
        f"Estimated total cost: **₹{total_cost:,}**."
    )
 
    st.markdown("**Confusion matrix at this threshold**")
    cm_df = pd.DataFrame(
        [[int(closest_row["tn"]), int(closest_row["fp"])],
         [int(closest_row["fn"]), int(closest_row["tp"])]],
        index=["Actual: Not Returned", "Actual: Returned"],
        columns=["Predicted: Not Returned", "Predicted: Returned"],
    )
    st.dataframe(cm_df, use_container_width=True)
 
    st.markdown("**Precision-recall curve**")
    pr_sorted = sweep_df.sort_values("recall")
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=pr_sorted["recall"], y=pr_sorted["precision"], mode="lines",
        line=dict(color="#4a5a8a", width=2), name="Model"
    ))
    fig_pr.add_trace(go.Scatter(
        x=[closest_row["recall"]], y=[closest_row["precision"]], mode="markers",
        marker=dict(color="#b23b3b", size=10), name="Current threshold"
    ))
    fig_pr.update_layout(
        xaxis_title="Recall", yaxis_title="Precision", height=350,
        margin=dict(t=20, b=20, l=40, r=20), showlegend=False,
    )
    st.plotly_chart(fig_pr, use_container_width=True)
 
    with st.expander("Classification report at the default cost-optimal threshold"):
        st.caption(f"Threshold = {model_metrics['threshold_used']}, chosen to minimize total cost.")
        report = model_metrics["classification_report"]
        report_rows = []
        for label, values in report.items():
            if isinstance(values, dict):
                report_rows.append({
                    "class": label,
                    "precision": round(values.get("precision", 0), 3),
                    "recall": round(values.get("recall", 0), 3),
                    "f1-score": round(values.get("f1-score", 0), 3),
                    "support": int(values.get("support", 0)),
                })
        st.dataframe(pd.DataFrame(report_rows).set_index("class"), use_container_width=True)

#Tab 3: Audit Trail
with tab_audit:
    st.subheader("Audit Trail")
    st.caption(
        "A sample of held-out test orders showing the score, decision, and top reason "
        "for each — the same record a reviewer would use to check the model's calls."
    )

    threshold = threshold_config["threshold"]
    audit_rows = []
    for record in audit_sample:
        prob = record["predicted_probability"]
        decision, _, _ = decide(prob, threshold)
        input_df = pd.DataFrame([{k: v for k, v in record.items()
                                   if k not in ("true_label", "predicted_probability")}])
        input_df = input_df[num_features + cat_features]
        try:
            _, top_feature_name, direction = top_contributions(input_df, top_n=1)[0]
            reason = f"{top_feature_name} ({'increases' if direction == 'up' else 'decreases'} risk)"
        except Exception:
            reason = "n/a"

        audit_rows.append({
            "Score": f"{prob:.0%}",
            "Decision": decision,
            "Top Reason": reason,
            "Actual Outcome": "Returned" if record["true_label"] == 1 else "Not Returned",
        })

    audit_df = pd.DataFrame(audit_rows)
    st.dataframe(audit_df, use_container_width=True, height=600)

#Tab 4: Architecture
with tab_architecture:
    st.subheader("How a Score Is Produced")
    st.write(
        "Data flows through the same pipeline for every order, whether scored live "
        "on this page or evaluated in bulk on the held-out test set."
    )

    boxes = ["Raw Order Data", "Feature\nEngineering", "XGBoost\nModel", "Risk Score", "Decision"]
    n = len(boxes)
    box_width = 0.15
    y0, y1 = 0.3, 0.7

    fig_arch = go.Figure()
    positions = []
    for i, label in enumerate(boxes):
        x_center = (i + 0.5) / n
        x0, x1 = x_center - box_width / 2, x_center + box_width / 2
        positions.append((x0, x1))
        fig_arch.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#555555", width=1.5), fillcolor="#f2f2f2",
        )
        fig_arch.add_annotation(
            x=x_center, y=(y0 + y1) / 2, text=label.replace("\n", "<br>"), showarrow=False,
            font=dict(size=12, color="#1a1a1a"), align="center",
        )
    for i in range(n - 1):
        fig_arch.add_annotation(
            x=positions[i + 1][0], y=(y0 + y1) / 2, ax=positions[i][1], ay=(y0 + y1) / 2,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor="#555555",
        )
    fig_arch.update_xaxes(visible=False, range=[0, 1])
    fig_arch.update_yaxes(visible=False, range=[0, 1])
    fig_arch.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    st.plotly_chart(fig_arch, use_container_width=True)

    st.markdown("""
    - **Raw order data** — price, shipping cost, payment method, product category, delivery timing, and customer region, as entered on the Score tab or pulled from an order record.
    - **Feature engineering** — the shipping-cost-to-price ratio and two interaction terms are calculated automatically; seller-history stats (average delay, historical return rate) are looked up from the seller's order record in production, shrunk toward the dataset average for sellers with limited history; categorical fields are encoded.
    - **XGBoost model** — a gradient-boosted tree ensemble trained on historical orders, outputs a probability of return, cancellation, or dissatisfaction.
    - **Risk score** — the raw probability, shown as a percentage.
    - **Decision** — the score is mapped to Auto-Approve, Manual Review, or Auto-Flag based on the cost-optimized threshold, with scores near the boundary routed to manual review rather than forced either way.
    """)

    st.divider()
    st.caption(
        "Built for the Razorpay AI Builder Hackathon 2026, AI Risk Manager track."
    )
