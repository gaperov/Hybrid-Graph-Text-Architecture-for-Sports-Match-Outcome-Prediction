
# Final Report  
# Hybrid Graph-Text Architecture for Sports Match Outcome Prediction

## 1. Project Overview

The goal of this project was to predict the outcome of FIFA World Cup matches.

The target variable has three classes:

| Class | Meaning |
|---:|---|
| 0 | home_win |
| 1 | draw |
| 2 | away_win |

The project used a hybrid machine learning approach with several types of data:

- historical World Cup match data;
- FIFA ranking features;
- recent team form features;
- historical World Cup performance features;
- tournament cumulative features;
- rest days;
- match stage features;
- pre-match text data;
- text embeddings;
- graph data;
- baseline machine learning models;
- exploratory GNN models.

The main modelling dataset covers World Cup matches from 1994 to 2022.

The reason for starting from 1994 is that FIFA rankings became available from 1992, so from the 1994 World Cup onward it is possible to build a more complete and consistent feature set.

The temporal split was:

| Split | Years |
|---|---|
| Train | 1994, 1998, 2002, 2006, 2010, 2014 |
| Validation | 2018 |
| Test | 2022 |

This split was chosen to avoid tournament leakage.  
The model is trained on past tournaments, validated on 2018, and finally tested on 2022.

---

## 2. Work Completed

### 2.1. Model-ready match dataset

A model-ready dataset was prepared:

```text
data/processed/match_dataset_model_ready
```

It contains 500 World Cup matches from 1994 to 2022.

The dataset includes:

- target;
- FIFA ranking features;
- rolling form features;
- historical World Cup performance;
- tournament cumulative features;
- rest days;
- stage features;
- text availability features;
- leakage-safe pre-match text features;
- missing value handling;
- leakage-safe preprocessing.

The following leakage columns were not used as model features:

- `home_score`;
- `away_score`;
- `result`;
- `target`;
- `home_xg`;
- `away_xg`;
- `xg`;
- post-match text.

---

### 2.2. Text layer

A leakage-safe text layer was built using Guardian pre-match articles.

Only articles published at least 6 hours before the match were used:

```text
published_at <= match_date - 6 hours
```

The text embeddings were created with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding size was:

```text
384
```

For matches without available text, a zero vector was used.

Additional text metadata features were also created:

- `text_available`;
- `text_count`;
- `log_text_count`;
- `guardian_text_flag`;
- `no_text_flag`;
- timing information about available articles.

---

### 2.3. Baseline models

Several baseline models were trained and compared.

The baseline experiments included:

- majority baseline;
- stage-only models;
- text-meta-only models;
- text-only models;
- team form models;
- World Cup history models;
- ranking models;
- combined numeric models;
- numeric + text embedding models;
- simple graph feature models;
- numeric + text + simple graph models.

The following model families were tested:

- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- HistGradientBoosting;
- XGBoost;
- MLP;
- simple graph-based models.

The baseline results are stored in:

```text
data/processed/model_results_ablation.csv
```

---

### 2.4. Graph layer

A graph layer was built for World Cup entities.

The graph contains several types of nodes:

- Team;
- Match;
- Tournament;
- Stadium;
- Referee;
- TeamTournament;
- Manager, when available.

The graph contains relationships such as:

- team played match;
- team played against team;
- team participated in tournament;
- match belongs to tournament;
- match was played at stadium;
- referee officiated match;
- team has tournament instance;
- team tournament played match;
- manager managed team in tournament.

The graph files are:

```text
data/processed/graph_nodes_model_ready
data/processed/graph_edges_model_ready
```

Before building a GNN, simple graph features were tested:

- home team degree;
- away team degree;
- degree difference;
- centrality features.

These simple graph features did not produce strong results.

---

### 2.5. GNN prototype

An exploratory GNN prototype was implemented.

To keep the model simple and avoid overfitting, the first GNN used a simplified Team-Match graph:

```text
Team <-> Match
```

The reason for this simplification is that the dataset is small.  
There are only around 500 matches, so a full heterogeneous graph model could easily overfit.

The GNN architecture used:

- PyTorch Geometric;
- `HeteroData`;
- node types:
  - `team`;
  - `match`;
- edge types:
  - `team -> match`;
  - `match -> team`;
  - optional `team -> team`;
- HeteroConv;
- SAGEConv;
- hidden dimension 32 or 64;
- 1 or 2 message passing layers;
- dropout between 0.35 and 0.50;
- classifier on Match node embeddings;
- weighted CrossEntropyLoss;
- early stopping based on validation macro-F1.

The Match node features were tested in different variants:

1. numeric only;
2. numeric + text PCA;
3. numeric + text PCA + text metadata or stage features.

The original text embeddings had 384 dimensions.  
For the GNN, PCA was used to reduce them to 32 dimensions.  
PCA was fitted only on the training data.

Important note:

The current GNN is a transductive prototype.  
The graph includes nodes and edges from 1994 to 2022.  
Labels are only used through the training mask, but future graph structure may still create some leakage.

So the GNN results should be treated as exploratory, not as the final leakage-safe benchmark.

---

## 3. Evaluation Setup

Two result files were used for this report:

1. `model_results_ablation.csv`  
   This file contains results for all baseline models.

2. `gnn_vs_baseline_comparison.csv`  
   This file compares the key baseline models with the GNN models.

The main metric for model selection is:

```text
validation macro-F1
```

This metric was chosen because:

- the task has three classes;
- the classes are imbalanced;
- draws are difficult to predict;
- the test set must not be used for model selection.

Other important metrics are:

- balanced accuracy;
- weighted F1;
- draw recall;
- draw F1;
- test macro-F1 for final evaluation.

---

## 4. Main Results

The most important models are shown below.

| Model | Feature set | Val macro-F1 | Test macro-F1 | Val draw F1 | Test draw F1 |
|---|---|---:|---:|---:|---:|
| `ranking_compact_rf` | ranking_compact | **0.565** | 0.469 | **0.400** | 0.276 |
| `numeric_text_rf` | numeric + text embeddings | 0.546 | **0.554** | 0.235 | **0.462** |
| `gnn_team_match_numeric_text_meta` | numeric + text PCA + meta | 0.520 | 0.453 | 0.261 | 0.231 |
| `gnn_match_numeric_only` | numeric only | 0.505 | 0.477 | 0.250 | 0.333 |
| `combined_numeric_rf_safe` | combined_numeric | 0.490 | 0.539 | 0.194 | 0.457 |
| `gnn_team_match_numeric_text` | numeric + text PCA | 0.489 | 0.458 | 0.231 | 0.231 |
| `gnn_team_match_numeric_text_small` | numeric + text PCA | 0.483 | 0.530 | 0.276 | 0.357 |

---

## 5. Baseline Model Analysis

### 5.1. Majority baseline

The majority baseline always predicts the most common class.

Its results were:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.406 | 0.453 |
| Macro-F1 | 0.193 | 0.208 |
| Draw F1 | 0.000 | 0.000 |

This is expected.  
The majority model does not predict draws at all.

All stronger models clearly outperform this baseline.

---

### 5.2. Best validation baseline

The best model by validation macro-F1 was:

```text
ranking_compact_rf
```

Its results were:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.594 | 0.500 |
| Balanced accuracy | 0.574 | 0.478 |
| Macro-F1 | **0.565** | 0.469 |
| Weighted F1 | 0.597 | 0.498 |
| Draw recall | **0.462** | 0.267 |
| Draw F1 | **0.400** | 0.276 |

This model performed best on validation.  
It was also the best model for predicting draws on validation.

This means that compact FIFA ranking features are very strong predictors.

A likely explanation is that FIFA rankings already capture a lot of information about relative team strength.

Random Forest also works well here because:

- the dataset is small;
- the features are tabular;
- the relationships are probably nonlinear;
- Random Forest is robust in small-data settings.

However, this model performed worse on the 2022 test set than some other models.

---

### 5.3. Best hybrid text baseline

The best hybrid text model was:

```text
numeric_text_rf
```

Its results were:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | **0.641** | **0.578** |
| Balanced accuracy | 0.562 | **0.552** |
| Macro-F1 | 0.546 | **0.554** |
| Weighted F1 | **0.606** | **0.577** |
| Draw recall | 0.154 | 0.400 |
| Draw F1 | 0.235 | **0.462** |

This model had the best test macro-F1.

It also had the best test draw F1 among the main models.

This suggests that pre-match text embeddings are useful when combined with numeric features.

The text may contain information about:

- team context;
- injuries;
- expectations;
- public narratives;
- match importance;
- uncertainty before the match.

However, the validation draw recall was low:

```text
draw_recall_val = 0.154
```

So the model did not detect many draws in the 2018 validation tournament.

---

### 5.4. Combined numeric Random Forest

The model:

```text
combined_numeric_rf_safe
```

had the following results:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.531 | 0.547 |
| Balanced accuracy | 0.483 | 0.550 |
| Macro-F1 | 0.490 | 0.539 |
| Weighted F1 | 0.548 | 0.552 |
| Draw recall | 0.231 | **0.533** |
| Draw F1 | 0.194 | 0.457 |

This model was not the best on validation, but it performed very well on the test set.

It also had strong draw recall and draw F1 on the 2022 test tournament.

This shows that numeric features alone already contain a lot of predictive signal.

---

### 5.5. Boosting models

The XGBoost model on combined numeric features achieved:

| Metric | Validation | Test |
|---|---:|---:|
| Macro-F1 | 0.479 | 0.531 |
| Draw F1 | 0.188 | 0.414 |

This was a solid result, but it did not beat the best Random Forest models.

Gradient Boosting and HistGradientBoosting were also competitive, but not better than Random Forest.

A possible reason is that boosting methods are more sensitive to hyperparameters and small sample size.

---

### 5.6. Linear models

Logistic Regression models were weaker than tree-based models.

For example:

```text
ranking_full_logreg
val macro-F1  = 0.464
test macro-F1 = 0.467
```

```text
combined_numeric_logreg
val macro-F1  = 0.443
test macro-F1 = 0.390
```

This suggests that the task is not well described by simple linear decision boundaries.

The outcome probably depends on nonlinear interactions between:

- ranking differences;
- team form;
- stage;
- rest days;
- tournament context;
- historical performance.

---

### 5.7. Text-only models

Text-only models were not strong enough as standalone models.

For example:

```text
text_only_mlp_all_matches
val macro-F1  = 0.354
test macro-F1 = 0.411
test draw F1  = 0.483
```

```text
text_only_logreg_all_matches
val macro-F1  = 0.314
test macro-F1 = 0.412
test draw F1  = 0.471
```

These models had weak overall macro-F1, but some of them were surprisingly good at predicting draws on the test set.

This suggests that text may contain useful information about uncertainty or balanced matches.

However, text alone is not enough for stable prediction.

The best use of text is as an additional feature source combined with numeric match features.

---

### 5.8. Stage-only and text-meta-only models

The stage-only model had low overall performance:

```text
stage_logreg
val macro-F1  = 0.201
test macro-F1 = 0.228
```

But it had very high draw recall:

```text
val draw recall  = 0.769
test draw recall = 0.800
```

This means the model predicted many draws, but often incorrectly.

So stage features alone are not enough.

The text-meta-only model was also weak:

```text
text_meta_logreg
val macro-F1  = 0.172
test macro-F1 = 0.320
```

However, stage and text metadata can still be useful as additional features.

---

### 5.9. Simple graph features

The simple graph model:

```text
simple_graph_logreg
```

showed:

| Metric | Validation | Test |
|---|---:|---:|
| Macro-F1 | 0.365 | 0.478 |
| Draw F1 | 0.154 | 0.222 |

This is better than the majority baseline, but clearly worse than the strongest tabular and text models.

This means that simple graph features like degree and centrality are not enough.

---

## 6. GNN Model Analysis

The GNN results were:

| Model | Feature set | Val macro-F1 | Test macro-F1 | Val draw F1 | Test draw F1 |
|---|---|---:|---:|---:|---:|
| `gnn_team_match_numeric_text_meta` | numeric_text_pca32_extra | **0.520** | 0.453 | 0.261 | 0.231 |
| `gnn_match_numeric_only` | numeric_only | 0.505 | 0.477 | 0.250 | 0.333 |
| `gnn_team_match_numeric_text` | numeric_text_pca32 | 0.489 | 0.458 | 0.231 | 0.231 |
| `gnn_team_match_numeric_text_small` | numeric_text_pca32 | 0.483 | **0.530** | **0.276** | **0.357** |

---

### 6.1. Best GNN on validation

The best GNN on validation was:

```text
gnn_team_match_numeric_text_meta
```

Its results were:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.578 | 0.500 |
| Balanced accuracy | 0.522 | 0.462 |
| Macro-F1 | **0.520** | 0.453 |
| Weighted F1 | 0.571 | 0.494 |
| Draw recall | 0.231 | 0.200 |
| Draw F1 | 0.261 | 0.231 |

This model is worse than the two strongest baselines:

```text
ranking_compact_rf val macro-F1 = 0.565
numeric_text_rf val macro-F1    = 0.546
```

So the GNN did not improve validation performance.

---

### 6.2. Best GNN on test

The best GNN on the test set was:

```text
gnn_team_match_numeric_text_small
```

Its results were:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.516 | 0.563 |
| Balanced accuracy | 0.535 |
| Macro-F1 | 0.483 | **0.530** |
| Weighted F1 | 0.523 | **0.564** |
| Draw recall | 0.308 | 0.333 |
| Draw F1 | 0.276 | 0.357 |

This is interesting because the smaller GNN performed worse on validation but better on test.

This suggests that smaller and more regularized GNNs may generalize better in this small-data setting.

However, this model should not be selected as the final model based on test performance.  
The test set should only be used once for final evaluation.

---

### 6.3. Did text help the GNN?

The GNN results show that text did not clearly help the GNN.

Compare:

```text
gnn_match_numeric_only
val macro-F1 = 0.505
test macro-F1 = 0.477
```

```text
gnn_team_match_numeric_text
val macro-F1 = 0.489
test macro-F1 = 0.458
```

```text
gnn_team_match_numeric_text_meta
val macro-F1 = 0.520
test macro-F1 = 0.453
```

Adding text PCA alone did not improve the GNN.

Adding text metadata improved validation macro-F1, but not test macro-F1.

This means that the GNN did not use dense text embeddings as effectively as Random Forest.

Possible reasons:

1. PCA to 32 dimensions may lose useful semantic information.
2. The dataset is too small for a neural text branch.
3. Zero vectors for missing text are difficult for the model.
4. Guardian coverage may introduce source-specific bias.
5. Random Forest may handle mixed numeric and embedding features better in this setting.

---

### 6.4. GNN and draw prediction

Draw prediction remains difficult.

Best validation draw F1 among GNN models:

```text
gnn_team_match_numeric_text_small
draw F1 val = 0.276
```

Best validation draw F1 among all main models:

```text
ranking_compact_rf
draw F1 val = 0.400
```

Best test draw F1 among main models:

```text
numeric_text_rf
draw F1 test = 0.462
```

Best test draw F1 among GNN models:

```text
gnn_team_match_numeric_text_small
draw F1 test = 0.357
```

So the GNN did not solve the draw prediction problem better than the strongest baselines.

---

## 7. Overall Comparison

### 7.1. Best model by validation

The main model selection metric is validation macro-F1.

By this metric, the best model is:

```text
ranking_compact_rf
```

With:

```text
validation macro-F1 = 0.565
```

It also has the best validation draw F1:

```text
draw F1 val = 0.400
```

So if we choose the model strictly by validation performance, the winner is `ranking_compact_rf`.

---

### 7.2. Best model on test

The best test macro-F1 is achieved by:

```text
numeric_text_rf
```

With:

```text
test macro-F1 = 0.554
```

This model also has the best test draw F1 among the main models:

```text
draw F1 test = 0.462
```

This makes `numeric_text_rf` the strongest hybrid text model.

However, it should not be selected only because it performed best on the test set.  
The test set is for final evaluation, not model selection.

---

### 7.3. GNN compared to baselines

The GNN models were competitive but did not outperform the strongest baselines.

Best GNN by validation:

```text
gnn_team_match_numeric_text_meta
val macro-F1 = 0.520
```

This is lower than:

```text
ranking_compact_rf: 0.565
numeric_text_rf:    0.546
```

Best GNN by test:

```text
gnn_team_match_numeric_text_small
test macro-F1 = 0.530
```

This is close to:

```text
combined_numeric_rf_safe test macro-F1 = 0.539
```

but still lower than:

```text
numeric_text_rf test macro-F1 = 0.554
```

Final interpretation:

```text
The simplified Team-Match GNN did not outperform strong tabular and text baselines.
```

---

## 8. Why the GNN Did Not Beat the Baselines

### 8.1. The dataset is small

There are only about 500 matches.

This is small for a GNN because the model has to learn:

- node projections;
- message passing;
- relation aggregation;
- classification layers;
- numeric and text feature fusion.

Random Forest is often stronger and more stable on small tabular datasets.

---

### 8.2. The Team-Match graph is too simple

The current graph structure is:

```text
Team <-> Match
```

This may not add much new information.

The tabular features already include strong signals such as:

- FIFA rankings;
- recent form;
- historical performance;
- tournament context.

Also, a single Team node combines all versions of the same national team across many years.

For example:

```text
Brazil 1994, Brazil 2002, Brazil 2022
```

are very different teams, but in the current graph they are connected to the same Team node.

This may make the graph representation less useful.

---

### 8.3. The graph prototype is transductive

The current GNN uses a graph containing matches from 1994 to 2022.

Even though labels are masked, the future graph structure may still be visible.

This setup is useful for prototyping, but it is not the strict final evaluation setup.

Importantly, even with this extra structural information, the GNN still did not beat the strongest baselines.

---

### 8.4. Text embeddings worked better with Random Forest

The best text-based model was:

```text
numeric_text_rf
```

This suggests that the text embeddings are useful.

However, the GNN did not use them as effectively.

Possible reasons include:

- not enough data;
- too much noise in text embeddings;
- PCA compression;
- missing text represented by zero vectors;
- source-specific bias from Guardian articles.

---

### 8.5. Validation and test each contain only one tournament

Validation is the 2018 World Cup.  
Test is the 2022 World Cup.

One tournament is a small sample.

Metrics can change a lot because of:

- a few unexpected matches;
- number of draws;
- tournament-specific patterns;
- changes in team strength;
- random variation.

So small differences in macro-F1 should be interpreted carefully.

---

## 9. Main Conclusions

1. Strong models clearly beat the majority baseline.

2. The best validation-selected model is:

   ```text
   ranking_compact_rf
   ```

   with:

   ```text
   val macro-F1 = 0.565
   ```

3. The best test-performing model is:

   ```text
   numeric_text_rf
   ```

   with:

   ```text
   test macro-F1 = 0.554
   ```

4. Text embeddings are useful when combined with numeric features.

5. Text-only models are not strong enough by themselves.

6. Tree-based models, especially Random Forest, are the most effective models in this project.

7. Simple graph features are weak.

8. The simplified Team-Match GNN did not improve over strong baselines.

9. Smaller GNNs may generalize better than larger GNNs.

10. Draw prediction remains the hardest part of the task.

11. The current GNN should be treated as an exploratory extension, not as the final best model.

---

## 10. Recommended Final Model Interpretation

It is useful to present the models in three groups.

### 10.1. Primary validation-selected model

```text
ranking_compact_rf
```

This model should be presented as the main validation-selected model.

Reasons:

- best validation macro-F1;
- best validation draw F1;
- simple feature set;
- interpretable;
- robust for small data.

---

### 10.2. Best hybrid text model

```text
numeric_text_rf
```

This model should be presented as the best hybrid text model.

Reasons:

- best test macro-F1;
- best test weighted F1;
- best test draw F1;
- shows that leakage-safe pre-match text embeddings add value.

---

### 10.3. Exploratory graph model

```text
gnn_team_match_numeric_text_meta
```

This model should be presented as the best exploratory GNN by validation.

Reasons:

- best GNN validation macro-F1;
- uses numeric features, text PCA, and text metadata;
- proves that the GNN pipeline works;
- but does not beat strong Random Forest baselines.

---

## 11. Future Work and Improvement Hypotheses

### 11.1. Run multi-seed experiments

The current GNN results should be tested with several random seeds.

Recommended seeds:

```text
1, 2, 3, 4, 5, 42, 123, 777, 2024, 2025
```

For each model, report:

- mean validation macro-F1;
- standard deviation of validation macro-F1;
- mean test macro-F1;
- standard deviation of test macro-F1;
- mean draw F1;
- standard deviation of draw F1.

Hypothesis:

```text
GNN results may have high variance because the dataset is small.
```

---

### 11.2. Run an MLP sanity check

To understand if message passing helps, compare GNNs with MLPs using the same match features.

Suggested comparison:

| Model | Features |
|---|---|
| MLP | numeric only |
| MLP | numeric + text PCA |
| MLP | numeric + text PCA + metadata |
| GNN | numeric + text PCA + metadata |

If MLP performs as well as GNN, then the graph structure is not adding useful information.

Hypothesis:

```text
The current Team-Match message passing may not add much beyond match features.
```

---

### 11.3. Try smaller GNN models

The smaller GNN had the best test result among GNNs.

So future experiments should try even more regularized models.

Recommended grid:

```text
hidden_dim: 16, 32
num_layers: 1, 2
dropout: 0.45, 0.50, 0.60
weight_decay: 3e-4, 1e-3
text_pca_dim: 16, 32
```

Hypothesis:

```text
Smaller and more regularized GNNs will be more stable in this small-data setting.
```

---

### 11.4. Build a TeamTournament graph

This is probably the most important next graph improvement.

The current graph uses one Team node for each national team:

```text
Team -> Match
```

This is not ideal because the same national team changes a lot across tournaments.

A better graph would use TeamTournament nodes:

```text
TeamTournament -> Match
```

Examples:

```text
Brazil_1994
Brazil_1998
Brazil_2002
Argentina_2022
France_2018
```

This better represents the real football situation.

Possible TeamTournament node features:

- FIFA rank before tournament;
- average rank before tournament;
- pre-tournament form;
- historical World Cup stats before tournament;
- previous World Cup performance;
- manager information;
- squad strength, if available later;
- cumulative tournament features before each match.

Hypothesis:

```text
A TeamTournament-Match graph will be more useful than a simple Team-Match graph.
```

---

### 11.5. Build a time-aware graph benchmark

The current GNN is transductive and exploratory.

A stricter graph benchmark should avoid future structural leakage.

Recommended setup:

```text
train graph:
nodes and edges up to 2014

validation graph:
historical graph before the 2018 World Cup
plus validation match nodes without labels

test graph:
historical graph before the 2022 World Cup
plus test match nodes without labels
```

An even stricter setup:

```text
For each match, use only edges with timestamp earlier than the match time.
```

Hypothesis:

```text
A time-aware graph will provide a cleaner and more trustworthy estimate of GNN performance.
```

---

### 11.6. Do not move to full heterograph too quickly

A full heterograph could include:

- Team;
- TeamTournament;
- Match;
- Tournament;
- Stadium;
- Referee;
- Manager.

However, this should not be the immediate next step.

Recommended order:

1. Team-Match GNN — already done.
2. TeamTournament-Match GNN.
3. Add Tournament nodes.
4. Add Stadium nodes.
5. Add Referee nodes.
6. Add Manager nodes, only if coverage is good.

Each step should be tested with ablation.

Hypothesis:

```text
Adding too many node and edge types too early will increase overfitting.
```

---

### 11.7. Improve the text branch

The text embeddings were useful in Random Forest, but not clearly useful in GNN.

Possible improvements:

1. Try different PCA dimensions:

   ```text
   16, 32, 64, 128
   ```

2. Compare PCA embeddings with raw 384-dimensional embeddings.

3. Use a learned missing-text embedding instead of a zero vector.

4. Always include explicit text availability features:

   ```text
   text_available
   no_text_flag
   log_text_count
   ```

5. Check the effect of Guardian source bias.

Hypothesis:

```text
Text embeddings are useful, but GNN needs a better text fusion mechanism.
```

---

### 11.8. Improve draw prediction

Draws are the hardest class.

Possible improvements:

1. Tune class weights.
2. Try focal loss.
3. Build a two-stage model:

   ```text
   Stage 1: draw vs non-draw
   Stage 2: home_win vs away_win
   ```

4. Calibrate predicted probabilities.
5. Tune the draw probability threshold on validation.
6. Add features that describe team similarity:

   - small rank difference;
   - small form difference;
   - similar historical strength;
   - group stage vs knockout stage;
   - rest day difference;
   - market odds, if available in the future.

Hypothesis:

```text
Draws should be modelled as a special uncertainty case, not just as one of three normal classes.
```

---

## 12. Final Recommendation

The project successfully built a hybrid tabular-text-graph pipeline for World Cup match outcome prediction.

The strongest validation-selected model was:

```text
ranking_compact_rf
```

with:

```text
validation macro-F1 = 0.565
```

The best hybrid text model was:

```text
numeric_text_rf
```

with:

```text
test macro-F1 = 0.554
```

The simplified Team-Match GNN pipeline was implemented successfully, but it did not outperform the strongest Random Forest baselines.

This suggests that, in the current small World Cup dataset, a simple graph structure does not add enough predictive information beyond ranking, form, numeric, and text features.

The best next direction is:

```text
TeamTournament-Match graph
+ time-aware graph construction
+ smaller regularized GNNs
+ better text fusion
+ special handling of draws
```

In short:

```text
Use ranking_compact_rf as the primary validation-selected model.
Use numeric_text_rf as the best hybrid text model.
Treat the current GNN as an exploratory graph extension.
Continue GNN work with TeamTournament nodes and leakage-safe time-aware graphs.
```
```