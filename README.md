# Return Risk Scorer

A model that estimates how likely an order is to get returned, cancelled, or end up with an unhappy customer — built for Razorpay's AI Buildathon, AI Risk Manager track.

Given order details (price, shipping cost, payment method, delivery timing, product category, seller history), it scores the order and routes it to auto-approve, manual review, or auto-flag, with a plain-language explanation of what drove the score.

## Why this exists

Returns and chargebacks quietly eat into a merchant's margin, and most of that loss is preventable if you can catch the risky order *before* it ships rather than after the return request comes in. The goal here wasn't to build a black box that says "risky" or "not risky" — it was to build something a merchant ops team could actually trust and audit: a score, a reason, a cost-justified threshold, and a record of what happened.

This is strictly a scoring/flagging tool. It doesn't take any automated action on orders, and it has no way to be used to probe or reverse-engineer its own detection thresholds.

## Dataset

Built on the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — real (anonymized) order, payment, product, and review data from a Brazilian marketplace. An order is labeled "bad" if it was cancelled/unavailable, got a review score of 2 or below, or arrived more than 7 days late. Category names were translated from Portuguese for readability (`src/translate_dataset.py`).

## How scoring works

1. **Raw order data** — price, freight cost, payment method, product category, delivery delay, customer region.
2. **Feature engineering** — freight-to-price ratio and a couple of interaction terms (freight × installments, delay × freight) get computed automatically. Seller history (average delay, historical bad-order rate, order count) is pulled from that seller's past orders and shrunk toward the dataset average when a seller doesn't have much history yet, so a seller with 3 orders doesn't get an unreliable score just because their small sample happened to look bad or good.
3. **Model** — an XGBoost classifier trained on the engineered features, outputting a probability of a bad outcome.
4. **Decision** — the probability gets mapped to Auto-Approve, Manual Review, or Auto-Flag using a threshold chosen to minimize total cost (see below). Scores that land close to the threshold get routed to manual review instead of being forced either way.
5. **Explanation** — SHAP values on the transformed features surface the top 3 factors behind each score, translated into a plain-language reason (e.g. "seller's historical return rate — increases risk").

## Notebooks

The notebooks trace the actual path from a plain logistic regression baseline to the final model, in order:

1. `01_baseline_logistic_regression.ipynb` — first pass, get something working end to end
2. `02_woe_iv_and_leakage_checks.ipynb` — checking feature strength and ruling out leakage before trusting anything
3. `03_seller_features_and_shrinkage.ipynb` — adding seller history, with shrinkage for low-volume sellers
4. `04_feature_engineering_and_xgboost.ipynb` — switching to XGBoost, adding interaction features
5. `05_error_analysis_and_final_model.ipynb` — digging into what the model still gets wrong, locking the final model

Metrics reported in the app come straight from this pipeline, on a held-out test set the model never trained on.

## Model performance

On a held-out test set of 22,816 orders (15.3% actually bad):

| Metric | Value |
|---|---|
| AUC-ROC | 0.751 |
| AUC-PR | 0.532 |
| Precision @ threshold 0.51 | 0.37 |
| Recall @ threshold 0.51 | 0.54 |

AUC-PR is the more honest number here since bad orders are the minority class — ROC-AUC alone can look better than the model deserves on an imbalanced problem like this one.

**On the threshold:** it's not 0.5 by default, and it's not a single fixed
number either. The threshold is chosen to minimize estimated rupee cost —
but the ratio between a false positive (wrongly flagging a good order) and
a false negative (missing a bad order) turned out to be a real open
question rather than an obvious assumption. The original build assumed a
missed bad order costs 6x a false flag (₹50 vs. ₹300). Industry research on
ecommerce fraud/risk detection generally finds the opposite: false
positives tend to cost merchants *more* than the bad orders they prevent —
one commonly cited figure is $13 lost to false declines for every $1 of
actual fraud. Olist has no direct cost data to settle this precisely, so
rather than pick one ratio, three scenarios are compared:

| Scenario | Threshold | Cost | Precision | Recall |
|---|---|---|---|---|
| 6:1, missed order costs more (original) | 0.51 | ₹641,050 | 0.37 | 0.54 |
| 3:1, false flag costs more | 0.89 | ₹143,100 | 0.92 | 0.25 |
| 2:1, industry-aligned | 0.88 | ₹138,850 | 0.91 | 0.25 |

The app defaults to the **2:1 industry-aligned scenario** for live scoring,
but the Model Performance tab lets you compare all three and drag the
threshold within any of them — precision, recall, and cost move together
live, since chasing precision as a number in isolation ignores what it
costs in missed recall.

Notice the 3:1 and 2:1 scenarios land on nearly the same threshold and
recall despite the ratio changing — that's not a bug, it means the
marginal trade-off in this region of the precision-recall curve is steep:
past roughly 50% recall, catching each additional bad order costs a
disproportionate number of false flags. Precision at 0.37-0.92 depending
on scenario reflects a real, cost-dependent trade-off, not one settled
number to defend.

## The app

A Streamlit app with four tabs:

- **Score an Order** — enter order details, get a score, a decision, and the top 3 factors behind it
- **Model Performance** — held-out metrics, a comparison table across three cost-ratio scenarios, an adjustable threshold slider with live precision/recall/cost for whichever scenario you pick, a confusion matrix, and the precision-recall curve
- **Audit Trail** — a sample of held-out orders with their score, decision, top reason, and actual outcome, so a reviewer can check the model's calls against ground truth
- **Architecture** — the data flow above, laid out visually

### Running it locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app resolves its artifact paths relative to `app.py`'s own location, so it works correctly whether run locally with `app/` as the working directory, or deployed on Streamlit Community Cloud (which runs from the repo root regardless of where the main file lives).

## Repo structure

```
app/          Streamlit app + trained model artifacts + precomputed metrics/config
docs/         PR curve and KS curve images
notebooks/    the actual model development trail, baseline through final
src/          dataset translation script + the final training script that produces everything app/ loads
```

## What's not solved yet

- Precision is still the weak point in the FN-costly (6:1) scenario, and recall is the weak point in the FP-costly scenarios — the underlying trade-off is steep, not a tuning oversight. Better seller/behavioral features or a two-stage review process would likely shift this more than further threshold tuning at this point.
- The true cost ratio is a business decision, not something this project can settle definitively — three scenarios are presented as evidence, not one answer.
- The seller_bad_rate error-analysis finding (notebook 05) was only verified under the original 6:1 threshold (0.51); whether the same false-positive concentration pattern holds at the industry-aligned threshold (0.88) hasn't been re-checked.
- Seller history in the live scoring tab is entered manually since there's no real seller database behind this demo — in a production setting it'd be looked up automatically from order history.
- No calibration step yet, so the raw probability shouldn't be read as a literal likelihood, just a ranking signal.

Built for the Razorpay AI Buildathon 2026, AI Risk Manager track.
