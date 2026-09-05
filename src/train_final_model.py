import joblib
import json
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

DATA_DIR = "data"
OUTPUT_DIR = "app"

orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
payments = pd.read_csv(f"{DATA_DIR}/olist_order_payments_dataset_en.csv")
reviews = pd.read_csv(f"{DATA_DIR}/olist_order_reviews_dataset.csv")
products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset_en.csv")
customers = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset_en.csv")

data = (orders
        .merge(items, on="order_id", how="left")
        .merge(payments, on="order_id", how="left")
        .merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(customers, on="customer_id", how="left"))

data["delay_days"] = (data["order_delivered_customer_date"] - data["order_estimated_delivery_date"]).dt.days
data["bad_order"] = (
    (data["order_status"].isin(["cancelled", "unavailable"])) |
    (data["review_score"] <= 2) |
    (data["delay_days"] > 7)
).astype(int)

data["freight_ratio"] = data["freight_value"] / data["price"].replace(0, 1)
data["purchase_dow"] = data["order_purchase_timestamp"].dt.dayofweek
data["purchase_month"] = data["order_purchase_timestamp"].dt.month

num_features = ["price", "freight_value", "freight_ratio", "payment_installments",
                 "product_weight_g", "delay_days", "purchase_dow", "purchase_month"]
cat_features = ["payment_type", "product_category_name", "customer_state"]

data = data.dropna(subset=num_features + cat_features + ["bad_order"])
x = data[num_features + cat_features]
y = data["bad_order"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#Seller history shrunk towards global average
if "seller_id" not in data.columns:
    data = data.merge(items[["order_id", "seller_id"]], on="order_id", how="left")

train_seller_lookup = data.loc[x_train.index, ["seller_id", "delay_days"]].copy()
train_seller_lookup["bad_order"] = y_train

seller_stats = train_seller_lookup.groupby("seller_id").agg(
    seller_avg_delay=("delay_days", "mean"),
    seller_bad_rate=("bad_order", "mean"),
    seller_order_count=("bad_order", "count"),
).reset_index()

global_avg_delay = train_seller_lookup["delay_days"].mean()
GLOBAL_BAD_RATE = train_seller_lookup["bad_order"].mean()
SHRINKAGE_WEIGHT = 20

seller_stats["seller_bad_rate"] = (
    (seller_stats["seller_bad_rate"] * seller_stats["seller_order_count"]) +
    (GLOBAL_BAD_RATE * SHRINKAGE_WEIGHT)
) / (seller_stats["seller_order_count"] + SHRINKAGE_WEIGHT)

def apply_seller_stats(df_slice):
    merged = data.loc[df_slice.index, ["seller_id"]].merge(seller_stats, on="seller_id", how="left")
    merged.index = df_slice.index
    merged["seller_avg_delay"] = merged["seller_avg_delay"].fillna(global_avg_delay)
    merged["seller_bad_rate"] = merged["seller_bad_rate"].fillna(GLOBAL_BAD_RATE)
    merged["seller_order_count"] = merged["seller_order_count"].fillna(0)
    return merged[["seller_avg_delay", "seller_bad_rate", "seller_order_count"]]

x_train = pd.concat([x_train, apply_seller_stats(x_train)], axis=1)
x_test = pd.concat([x_test, apply_seller_stats(x_test)], axis=1)

#Interaction features
for split_df in [x_train, x_test]:
    split_df["freight_installment_interaction"] = split_df["freight_ratio"] * split_df["payment_installments"]
    split_df["delay_freight_interaction"] = split_df["delay_days"] * split_df["freight_ratio"]

num_features_v2 = num_features + [
    "seller_avg_delay", "seller_bad_rate", "seller_order_count",
    "freight_installment_interaction", "delay_freight_interaction",
]

#Grouping rare categories into 'Others'
CATEGORY_MIN_COUNT = 200
STATE_MIN_COUNT = 150

rare_categories = set(x_train["product_category_name"].value_counts()
                       .loc[lambda s: s < CATEGORY_MIN_COUNT].index)
rare_states = set(x_train["customer_state"].value_counts()
                   .loc[lambda s: s < STATE_MIN_COUNT].index)

for split_df in [x_train, x_test]:
    split_df["product_category_name"] = split_df["product_category_name"].apply(
        lambda v: "Other" if v in rare_categories else v)
    split_df["customer_state"] = split_df["customer_state"].apply(
        lambda v: "Other" if v in rare_states else v)

# Rebuild seller stats post-grouping to keep the pipeline idempotent
x_train = x_train.drop(columns=["seller_avg_delay", "seller_bad_rate", "seller_order_count"], errors="ignore")
x_test = x_test.drop(columns=["seller_avg_delay", "seller_bad_rate", "seller_order_count"], errors="ignore")
x_train = pd.concat([x_train, apply_seller_stats(x_train)], axis=1)
x_test = pd.concat([x_test, apply_seller_stats(x_test)], axis=1)

#Training the final model
FEATURES = num_features_v2 + cat_features

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_features_v2),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = Pipeline([
    ("prep", preprocessor),
    ("clf", xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    ))
])

model.fit(x_train[FEATURES], y_train)
probs = model.predict_proba(x_test[FEATURES])[:, 1]

print("AUC-ROC:", roc_auc_score(y_test, probs))
print("AUC-PR:", average_precision_score(y_test, probs))

COST_SCENARIOS = {
    "fn_costly":  {"label": "Missed order costs more (6:1)",  "cost_fp": 50,  "cost_fn": 300},
    "balanced":   {"label": "False flag costs more (3:1)",    "cost_fp": 150, "cost_fn": 50},
    "fp_costly":  {"label": "Industry-aligned (2:1)",          "cost_fp": 100, "cost_fn": 50},
}
DEFAULT_SCENARIO = "fp_costly" 
 
sweep_thresholds = np.arange(0.1, 0.9, 0.01)
raw_sweep = []
for t in sweep_thresholds:
    preds = (probs > t).astype(int)
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tp = int(((preds == 1) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    raw_sweep.append({"threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn})
 
cost_scenario_results = {}
for key, scenario in COST_SCENARIOS.items():
    best_thresh, best_cost = None, float("inf")
    for row in raw_sweep:
        total_cost = row["fp"] * scenario["cost_fp"] + row["fn"] * scenario["cost_fn"]
        if total_cost < best_cost:
            best_cost, best_thresh = total_cost, row["threshold"]
    cost_scenario_results[key] = {
        **scenario, "optimal_threshold": best_thresh, "optimal_cost": best_cost,
    }
    print(f"[{scenario['label']}] optimal threshold: {best_thresh}, cost: {best_cost}")
 
best_thresh = cost_scenario_results[DEFAULT_SCENARIO]["optimal_threshold"]
cost_fp = COST_SCENARIOS[DEFAULT_SCENARIO]["cost_fp"]
cost_fn = COST_SCENARIOS[DEFAULT_SCENARIO]["cost_fn"]
print(f"\nDefault scenario for Score tab: {DEFAULT_SCENARIO} (threshold {best_thresh})")
print(classification_report(y_test, probs > best_thresh))

#Exporting all artifacts for app.py
joblib.dump(model, f"{OUTPUT_DIR}/risk_model.pkl")

X_train_transformed = preprocessor.transform(x_train[FEATURES])
if hasattr(X_train_transformed, "toarray"):
    X_train_transformed = X_train_transformed.toarray()
explainer = shap.TreeExplainer(model.named_steps["clf"], X_train_transformed,
                                feature_names=preprocessor.get_feature_names_out())
joblib.dump(explainer, f"{OUTPUT_DIR}/shap_explainer.pkl")

with open(f"{OUTPUT_DIR}/feature_config.json", "w") as f:
    json.dump({"num_features": num_features_v2, "cat_features": cat_features}, f, indent=2)

num_ranges = {
    feat: {"min": float(x_train[feat].min()), "max": float(x_train[feat].max()), "mean": float(x_train[feat].mean())}
    for feat in num_features_v2
}
with open(f"{OUTPUT_DIR}/num_ranges.json", "w") as f:
    json.dump(num_ranges, f, indent=2)

threshold_config = {"threshold": float(best_thresh), "cost_fp": cost_fp, "cost_fn": cost_fn}
with open(f"{OUTPUT_DIR}/threshold_config.json", "w") as f:
    json.dump(threshold_config, f, indent=2)

cat_options = {feat: sorted(x_train[feat].dropna().unique().tolist()) for feat in cat_features}
with open(f"{OUTPUT_DIR}/cat_options.json", "w") as f:
    json.dump(cat_options, f, indent=2)

preds_at_threshold = (probs > best_thresh).astype(int)
metrics = {
    "auc_roc": float(roc_auc_score(y_test, probs)),
    "auc_pr": float(average_precision_score(y_test, probs)),
    "threshold_used": float(best_thresh),
    "classification_report": classification_report(y_test, preds_at_threshold, output_dict=True),
    "test_set_size": int(len(y_test)),
    "positive_rate": float(y_test.mean()),
}
with open(f"{OUTPUT_DIR}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

y_test_arr = np.asarray(y_test)
sweep_rows = []
for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
    preds = (probs > t).astype(int)
    tp = int(((preds == 1) & (y_test_arr == 1)).sum())
    fp = int(((preds == 1) & (y_test_arr == 0)).sum())
    tn = int(((preds == 0) & (y_test_arr == 0)).sum())
    fn = int(((preds == 0) & (y_test_arr == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    sweep_rows.append({"threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                        "precision": precision, "recall": recall, "f1": f1,
                        "cost": fp * cost_fp + fn * cost_fn})
with open(f"{OUTPUT_DIR}/threshold_sweep.json", "w") as f:
    json.dump(sweep_rows, f, indent=2)

with open(f"{OUTPUT_DIR}/cost_scenarios.json", "w") as f:
    json.dump({"scenarios": cost_scenario_results, "default_scenario": DEFAULT_SCENARIO}, f, indent=2)
    
audit_base = x_test[FEATURES].reset_index(drop=True).copy()
audit_base["true_label"] = pd.Series(y_test).reset_index(drop=True)
audit_base["predicted_probability"] = probs
for col in num_features_v2:
    audit_base[col] = audit_base[col].round(2)
audit_base["predicted_probability"] = audit_base["predicted_probability"].round(4)
audit_sample = audit_base.sample(n=min(30, len(audit_base)), random_state=42).reset_index(drop=True)
with open(f"{OUTPUT_DIR}/audit_sample.json", "w") as f:
    json.dump(audit_sample.to_dict(orient="records"), f, indent=2)

print(f"\nAll artifacts written to {OUTPUT_DIR}/")
