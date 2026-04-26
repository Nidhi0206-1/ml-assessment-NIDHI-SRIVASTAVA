# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable:**
`items_sold` — the number of items sold in a given store during a given month under a specific promotion.

**Candidate Input Features:**

| Category | Features |
|---|---|
| Store attributes | `store_id`, `store_size`, `location_type` (urban / semi-urban / rural), `monthly_footfall`, `competition_density` |
| Promotion | `promotion_type` (Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points Bonus) |
| Temporal | `month`, `year`, `is_weekend_heavy_month`, `is_festival_month`, `season` |
| Customer demographics | `avg_customer_age_band`, `income_tier`, `loyalty_member_share` |
| Historical performance | `avg_items_sold_last_3_months`, `same_month_last_year_items_sold`, `promotion_type_historical_lift` |

**Type of ML Problem:**

This is a **supervised regression** problem. We are predicting a continuous numeric output — the expected volume of items sold — given a set of store, promotion, and contextual features.

However, because the business objective is to *choose the best promotion* rather than just predict sales, the system is best framed as a **counterfactual regression** task: for each store-month, we predict `items_sold` under *each of the five promotion options* separately, then recommend whichever promotion yields the highest predicted value.

**Justification:** The target (`items_sold`) is a continuous count variable with no natural upper bound relevant to a classifier. Regression is the appropriate choice because we need the actual magnitude of predicted sales — not just a ranked category — in order to compare promotions quantitatively and communicate expected uplift to the marketing team.

---

### B1(b) — Why Items Sold is a Better Target Than Revenue

**The case for items sold over revenue:**

Total sales revenue is the product of `items_sold × price`. Prices are not fixed — they are *directly altered by the promotion itself*. A Flat Discount mechanically reduces the unit price, which means revenue can fall even when the promotion successfully drives more customers into stores and increases volume. Using revenue as the target would penalise volume-driving promotions unfairly and reward those that preserve margin without actually increasing customer engagement.

`items_sold` captures the true behavioural outcome we care about: did the promotion attract customers and move product? It is promotion-agnostic in terms of pricing mechanics, so the model can make fair, like-for-like comparisons across promotion types.

**Broader Principle — Metric Leakage and Causal Contamination:**

This illustrates the principle that **the target variable should measure the underlying behaviour we want to change, not a downstream metric that is simultaneously manipulated by the intervention being studied.** In ML for business, targets that are co-determined by a feature (here: revenue co-determined by the promotion type through pricing) introduce a form of target contamination that makes the model's recommendations circular and unreliable. The right target isolates the causal outcome of interest.

---

### B1(c) — Beyond a Single Global Model

**The junior analyst's suggestion is problematic** because a single global model assumes that all stores respond to promotions in the same way — a homogeneity assumption that is almost certainly violated. An urban flagship with high footfall and a loyal customer base will respond very differently to a Loyalty Points Bonus than a small rural store where most customers are infrequent visitors.

**Proposed Alternative: Stratified or Hierarchical Modelling**

Two practical strategies:

1. **Segment-level models:** Group stores into clusters (e.g., by location type, size tier, and footfall band) and train a separate model per segment. This is computationally manageable and captures the most important sources of heterogeneity without requiring a fully bespoke model per store.

2. **Mixed-effects (hierarchical) model or store-level features in a single model with interaction terms:** A single model is retained, but `store_id` (or store cluster) is included as a feature with explicit interaction terms between store type and promotion type. This allows the model to learn that *the effect of BOGO depends on whether the store is urban or rural* without requiring completely separate models.

**Preferred approach:** Start with stratified models by location type (three groups: urban, semi-urban, rural) as a pragmatic first step. If data volume allows, refine to finer segments or add store-level random effects. This balances model complexity, interpretability, and data efficiency — individual store models would likely overfit given that each store only contributes ~36 monthly observations over three years.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

**Table Descriptions and Keys:**

| Table | Grain | Key Columns |
|---|---|---|
| `transactions` | One row per transaction | `store_id`, `transaction_date`, `promotion_id`, `items_sold`, `revenue` |
| `store_attributes` | One row per store | `store_id`, `store_size`, `location_type`, `footfall`, `competition_density` |
| `promotion_details` | One row per promotion | `promotion_id`, `promotion_type`, `discount_depth` |
| `calendar` | One row per date | `date`, `is_weekend`, `is_festival`, `month`, `year`, `season` |

**Join Sequence:**

```
transactions
  LEFT JOIN store_attributes   ON store_id
  LEFT JOIN promotion_details  ON promotion_id
  LEFT JOIN calendar           ON transaction_date = date
```

All joins are left joins from `transactions` as the fact table, preserving all sales records even if a store or promotion record is missing (which would itself be a data quality signal worth flagging).

**Final Modelling Grain:**

> **One row = one store × one month × one promotion type**

This is the decision unit: each month, the marketing team assigns one promotion per store. We therefore aggregate transactions up to this level before modelling.

**Aggregations performed:**

- `items_sold` → **sum** per store-month-promotion (target variable)
- `revenue` → sum (kept as a monitoring metric, not the target)
- `is_festival`, `is_weekend` → **max or mean** within the month (e.g., "did this month contain a festival?", "what fraction of days were weekends?")
- `footfall` → monthly total or average from store attributes (static unless updated)
- `transaction_count` → count of rows (a volume proxy and sanity check)

Missing promotion months (stores that ran no promotion) are retained as a sixth category — "no promotion" — since this is essential context for measuring promotional lift.

---

### B2(b) — EDA Strategy

**1. Target Distribution Analysis**

*Chart:* Histogram and boxplot of `items_sold` by promotion type.

*What to look for:* Are any promotion types associated with systematically higher sales volumes? Is the distribution skewed or bimodal (suggesting that a log transform may stabilise variance and improve regression model fit)? Outliers could indicate data entry errors or one-off events that need to be flagged.

*Modelling influence:* If the target is heavily right-skewed, consider log-transforming it. Outliers identified here may need to be capped or separately modelled.

**2. Promotion Lift by Store Segment**

*Chart:* Grouped bar chart — mean `items_sold` per promotion type, faceted by `location_type` (urban / semi-urban / rural).

*What to look for:* Does the winning promotion differ across store types? If BOGO dominates in urban stores but Flat Discount wins in rural stores, this is direct evidence for stratified modelling (reinforcing B1c) and confirms that `promotion_type × location_type` interaction terms are essential features.

*Modelling influence:* Interaction terms or segment-specific models should be included if the winning promotion varies by segment.

**3. Temporal Trends and Seasonality**

*Chart:* Line chart of monthly average `items_sold` (averaged across stores and promotions) over the full three-year period.

*What to look for:* Clear seasonal peaks (e.g., festival months, summer clearance) and year-over-year trends. If December is consistently 3× higher than February regardless of promotion, month and season are strong predictors and must be included as features.

*Modelling influence:* Extract `month`, `quarter`, `is_festival_month`, and `year` as temporal features. A model without these would have high residuals in peak months.

**4. Promotion × Month Interaction Heatmap**

*Chart:* Heatmap with months on one axis and promotion types on the other, cells showing mean `items_sold`.

*What to look for:* Are certain promotions more effective in specific months — e.g., does Loyalty Points Bonus over-index in January (post-festive loyalty rebuilding) while Free Gift with Purchase peaks in October (pre-festive gifting season)?

*Modelling influence:* If strong patterns emerge, a `promotion_type × month` interaction feature should be engineered explicitly rather than relying on the model to learn it from sparse data.

**5. Correlation and Multicollinearity Check**

*Chart:* Correlation heatmap of all numeric features — `footfall`, `competition_density`, `store_size` (encoded), `discount_depth`, historical sales features.

*What to look for:* Highly correlated feature pairs (|r| > 0.8) that could cause instability in linear models. Features with near-zero variance (e.g., `discount_depth` for non-discount promotions all coded as 0) that add noise.

*Modelling influence:* Drop or consolidate redundant features. For tree-based models, multicollinearity is less of a concern but can obscure feature importance interpretation.

---

### B2(c) — Addressing the 80% No-Promotion Imbalance

**How the imbalance affects the model:**

With 80% of records carrying no promotion, a naive model will be dominated by the "no promotion" baseline. It will learn to predict sales well for the majority condition but poorly capture the incremental lift — or drag — of each specific promotion type. When used to recommend promotions, the model may systematically underestimate promotional effects because the signal from promoted months is diluted by the volume of unpromoted observations.

Additionally, if the model is evaluated on overall RMSE, it can achieve a low error simply by predicting close to the no-promotion average for most inputs, masking poor performance precisely where recommendations matter most.

**Steps to address it:**

1. **Retain no-promotion records as a baseline category.** Do not discard them — they define the counterfactual. The model needs to know what happens without a promotion in order to estimate lift.

2. **Use a lift-focused evaluation metric.** Evaluate model performance separately on promoted vs. non-promoted records. Report RMSE and MAE stratified by promotion status so poor performance on promotional months is not hidden.

3. **Oversample or weight promoted records during training.** Use `sample_weight` in scikit-learn estimators to up-weight the 20% promoted observations, or oversample promoted months so the model gives them proportionally more influence during fitting.

4. **Engineer a lift feature explicitly.** Create a `historical_promo_lift` feature — the ratio of `items_sold` in promotion months to `items_sold` in non-promotion months for the same store and season. This gives the model a strong prior signal about how responsive each store is to promotions in general.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split, Metrics, and Evaluation Design

**Split Strategy — Temporal Hold-Out:**

With monthly data spanning 3 years across 50 stores, the data is structured in time. The correct split is:

- **Training set:** Months 1–30 (years 1 and 2.5, approximately 80% of the timeline)
- **Test set:** Months 31–36 (the final 6 months of year 3, approximately 20%)

All 50 stores are present in both sets — we are predicting future months for known stores, not unseen stores.

**Why a random split is inappropriate:**

A random split would allow the model to train on, say, December 2024 and be tested on September 2023 — training on the future and predicting the past. This creates **temporal data leakage**: the model would implicitly encode future seasonality, promotional trends, and market conditions, producing over-optimistic test performance that will not hold in production. It also means lag features (last month's sales, same-month-last-year) would be computed from test data, which the model should not have access to during training.

**Evaluation Metrics:**

| Metric | Formula | Business Interpretation |
|---|---|---|
| **RMSE** | √(mean of squared errors) | Penalises large prediction errors heavily. A recommendation that misses by 500 units is much worse than missing by 50, so this penalty is appropriate. The marketing team cares about not catastrophically over- or under-stocking. |
| **MAE** | Mean of absolute errors | The average number of items by which predictions miss. More interpretable than RMSE: "On average, our model is off by X units per store-month." |
| **Promotion Selection Accuracy** | % of months where the recommended promotion matches the empirically best-performing promotion | The most direct business metric. Did we actually recommend the right promotion? |
| **Simulated Revenue Lift** | Mean items_sold under model recommendations vs. mean under random or historical assignment | Translates model performance into business value. Even an imperfect model may deliver meaningful lift over the naive baseline. |

For all metrics, report results **stratified by location type and store size** — aggregate performance can mask poor predictions for a critical store segment.

---

### B3(b) — Investigating and Communicating Different Recommendations for the Same Store

**Context:** Store 12 receives Loyalty Points Bonus in December and Flat Discount in March.

**Step 1 — Extract SHAP values for each prediction.**

Use SHAP (SHapley Additive exPlanations) to decompose the model's predicted `items_sold` for Store 12 under each promotion type in December and March separately. SHAP assigns each feature a contribution score — positive (pushing the prediction up) or negative (pushing it down) — relative to the model's average prediction.

**Step 2 — Identify the drivers of each recommendation.**

For the December prediction, SHAP values might reveal:
- `is_festival_month = 1` contributes +80 units
- `loyalty_member_share = 0.62` (Store 12 has a high loyalty member base) contributes +60 units under Loyalty Points Bonus specifically
- `month = 12` contributes +45 units

For the March prediction:
- `is_festival_month = 0` contributes −30 units
- `competition_density = high` contributes +40 units for Flat Discount (price sensitivity is higher when competition is nearby)
- `footfall = lower than December` reduces the loyalty-programme payoff, making simple discounts more effective

**Step 3 — Communicate to the marketing team.**

Prepare a one-page brief structured around three questions:

> *Why December → Loyalty Points Bonus?* Store 12 has a high share of loyalty programme members. In December, when footfall is peak and members are spending more, a points multiplier drives disproportionately large basket sizes because frequent buyers stock up. Festival-season data historically shows that loyalty customers increase their spend by 40% more than casual shoppers under a points bonus.

> *Why March → Flat Discount?* March is a quieter month with lower footfall and higher local competition. Casual, price-sensitive shoppers dominate, and they respond to visible, immediate price reductions rather than deferred rewards. The model has learned this pattern from historical March data across similar stores.

> *What is the general principle?* The model is essentially learning that **customer type × promotion type × seasonality** is the key interaction. Loyalty-heavy, high-footfall months reward loyalty mechanics; low-footfall, price-sensitive months reward immediate discounts.

This framing keeps the explanation business-grounded and avoids overwhelming the team with model internals.

---

### B3(c) — End-to-End Deployment and Monitoring

**1. Saving the Model**

At the end of training, serialise the full scikit-learn pipeline (preprocessor + model) using `joblib`:

```python
import joblib
joblib.dump(pipeline, 'models/promo_recommender_v1.pkl')
```

Version the model file with a timestamp and git commit hash. Store it in a central model registry (e.g., MLflow Model Registry, or a versioned S3 bucket) alongside metadata: training date range, feature list, hyperparameters, and test-set RMSE.

**2. Preparing and Feeding Monthly Data**

At the start of each month, an automated pipeline runs the following steps:

1. **Extract:** Pull the latest store attributes, prior-month transaction totals, and the upcoming month's calendar flags (festivals, weekends) from the operational database.
2. **Transform:** Compute all engineered features — lag features from the previous month, rolling averages, year-on-year comparisons — using the same feature engineering code used during training (stored as a versioned script alongside the model).
3. **Score:** For each of the 50 stores, generate five rows (one per promotion type) and run all rows through the saved pipeline's `.predict()` method. Select the promotion with the highest predicted `items_sold` per store.
4. **Output:** Write the 50 recommendations to a dashboard or email report for the marketing team, with predicted uplift versus no-promotion baseline included for each recommendation.

Critical constraint: **the same preprocessing code must be used in inference as in training**. Wrapping preprocessing in a scikit-learn `Pipeline` object ensures this automatically.

**3. Monitoring for Model Degradation**

Three layers of monitoring:

| Layer | What to Monitor | Signal of Degradation |
|---|---|---|
| **Data drift** | Distribution of input features each month (mean, std, % missing) vs. training distribution. Use statistical tests (KS test, PSI) on key features like `footfall`, `competition_density`, and `promotion_type` frequency. | PSI > 0.2 or KS p-value < 0.05 for a key feature triggers an alert. |
| **Prediction drift** | Distribution of predicted `items_sold` values across stores each month. | Predictions systematically shifting up or down relative to the training-period baseline without a corresponding real-world event signals model staleness. |
| **Outcome monitoring** | After each month closes, compare actual `items_sold` for each store to the model's prediction. Track rolling MAE and RMSE over the last 3 months. | If rolling MAE exceeds 1.5× the test-set MAE from when the model was deployed, flag for retraining. |

**Retraining Trigger:**

Retrain the model quarterly as a scheduled default, incorporating the most recent 3 years of data. Additionally trigger an unscheduled retraining if:
- Outcome monitoring MAE exceeds the 1.5× threshold
- A major external event occurs (new competitor opens nearby, significant economic shift) that is flagged by the business
- A new promotion type is introduced that the model has never seen

**Retraining Process:**

Retraining follows the same pipeline as initial training, using the expanded historical dataset. The new model is validated on the most recent 3 months (held out from retraining) before replacing the production model. Both the old and new model run in shadow mode for one month so their recommendations can be compared before the switch is made permanent.
