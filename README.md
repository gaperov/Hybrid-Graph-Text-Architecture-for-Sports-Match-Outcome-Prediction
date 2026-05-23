# Hybrid Graph-Text Architecture for Sports Match Outcome Prediction

## 1. Introduction

Predicting sports match outcomes has long attracted interest from researchers in machine learning, statistics, operations research, and sports analytics. Traditional prediction methods mainly rely on structured numerical data such as historical match results, team rankings, player ratings, possession statistics, expected goals (xG), and bookmaker odds. While these features are informative, they often fail to capture contextual information that significantly influences match outcomes, including injuries, managerial changes, tactical adaptations, locker-room conflicts, travel fatigue, weather conditions, and psychological momentum.

Recent developments in graph-based machine learning have shown that sports competitions naturally form complex relational systems. Teams interact repeatedly over time, players transfer across clubs, coaches move between organizations, and competitions create evolving dependency structures. These relations can be represented as graphs, where nodes correspond to teams, players, or matches, and edges encode competitive or organizational relationships.

The recent study *Graph-Based Modeling and Prediction of Match Outcomes in International Football Tournaments* proposes a graph-theoretic framework using bipartite networks connecting clubs and national teams, eigenvector centrality, and random walk methods to predict football tournament outcomes, achieving approximately **60.41%** accuracy on the 2022 FIFA World Cup and **54.90%** on UEFA Euro 2020 [4]. This work demonstrates that graph structures capture useful relational information absent in standard tabular approaches.

However, graph-only models have a critical limitation: they ignore unstructured textual information. In modern sports ecosystems, a large amount of predictive information exists in text form — news articles, match previews, injury reports, tactical analyses, coach interviews, and social media discussions. For example, a late injury announcement or tactical shift may dramatically affect match probabilities but remain invisible to graph statistics.

This project proposes a **Hybrid Graph-Text Architecture** that integrates graph representations of sports ecosystems with natural language embeddings derived from sports-related textual content. The central hypothesis is that combining relational graph signals with contextual textual knowledge will improve predictive performance over standalone graph or tabular models.

The project focuses on football match outcome prediction, though the framework is generalizable to other sports such as basketball, hockey, or tennis.

### Main Research Question

Can the integration of graph neural networks and textual embeddings improve sports match outcome prediction compared with traditional graph-only or statistical baselines?

### Objectives

- Construct a dynamic sports interaction graph from historical football data.
- Extract contextual textual features from pre-match reports and news.
- Fuse graph and text modalities into a unified predictive architecture.
- Evaluate performance on multi-class match outcome prediction: win, draw, or loss.

---

## 2. Literature Review

### 2.1 Traditional Statistical Models

Early football prediction models are primarily based on probabilistic frameworks such as Poisson and bivariate Poisson models.

Dixon and Coles (1997) introduced a modified Poisson model for football score prediction, accounting for low-scoring dependence effects. Later Bayesian extensions improved uncertainty quantification and parameter regularization.

Rating systems such as Elo ratings are also widely used for team strength estimation. These systems dynamically update team ratings after each match and remain competitive baselines.

Although statistically interpretable, these methods assume relatively simple dependency structures and cannot model higher-order relationships among teams, players, and competitions.

### 2.2 Machine Learning Approaches

Machine learning methods expanded the feature space to include team statistics, player features, and bookmaker information.

Random forests, gradient boosting, logistic regression, and support vector machines have been applied successfully. Schauberger and Groll proposed random forest models for international football tournaments, showing improved predictive accuracy over classical regression models [5].

The Dolores model combines dynamic ratings with Bayesian networks and learns cross-league information, demonstrating strong generalization across football competitions [2].

However, most machine learning models treat matches as independent observations and ignore relational dependencies.

### 2.3 Graph-Based Models

Graph methods are increasingly popular in sports analytics because sports naturally involve interactions.

The graph-based football tournament paper by Osei Francis models clubs and national teams as a bipartite graph. Nodes represent clubs and national teams, while edges represent player affiliations. Eigenvector centrality identifies influential teams, and random walks estimate winning probabilities. The model achieved moderate success while maintaining interpretability [4].

Advantages of graph approaches include:

- relational reasoning;
- network influence modeling;
- representation of player mobility and organizational dependencies.

However, hand-engineered graph statistics such as centrality are limited in expressive power.

Graph Neural Networks (GNNs) provide a more flexible alternative by learning node embeddings through message passing. Xenopoulos and Silva demonstrated that graph neural networks improve sports prediction tasks by preserving player interactions and local dependencies [6].

GNNs can capture:

- team interaction histories;
- shared player dependencies;
- tournament progression effects.

### 2.4 Textual Information in Sports Prediction

Sports outcomes are influenced by latent contextual factors often described only in text.

Examples include:

- injury updates;
- tactical previews;
- coach interviews;
- transfer rumors;
- morale discussions.

Beal et al. showed that combining machine learning with expert textual information improves football prediction performance, reaching **63.18%** accuracy [1].

Modern NLP models such as BERT, RoBERTa, and Sentence-BERT generate dense embeddings from text and can encode semantic information from sports articles.

Despite progress, limited research combines graph representations and textual features in sports analytics.

This gap motivates the proposed hybrid architecture.

---

## 3. Methods

### 3.1 Dataset Collection

The project will combine structured and unstructured data.

#### Structured Data

Historical football data will include:

- match results;
- goals scored/conceded;
- team rankings;
- player rosters;
- lineups;
- venue information;
- competition metadata.

Possible sources:

- FBref;
- Kaggle football datasets;
- FIFA rankings;
- Understat.

---

### 3.2 Graph Construction

A heterogeneous temporal graph will be constructed.

#### Node Types

- Team nodes
- Player nodes
- Coach nodes
- Match nodes

#### Edge Types

- Team plays team
- Player belongs to team
- Coach manages team
- Player participated in match
- Team participated in tournament

Edges may include weights:

- recency;
- player importance;
- transfer value;
- ranking score.

This creates a dynamic sports ecosystem graph.

---

### 3.3 Graph Representation Learning

A Graph Neural Network will learn embeddings.

Candidate architectures:

- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- GraphSAGE
- Heterogeneous Graph Transformer (HGT)

Let the sports ecosystem be represented as a graph:

$$
G = (V, E)
$$

where:

- $V$ is the set of nodes;
- $E$ is the set of edges.

For a GNN layer, node representations can be updated as:

$$
h_v^{(k)} = \sigma \left( W^{(k)} \cdot \text{AGGREGATE}^{(k)} \left( \{ h_u^{(k-1)} : u \in \mathcal{N}(v) \} \right) \right)
$$

where:

- $h_v^{(k)}$ is the representation of node $v$ at layer $k$;
- $\mathcal{N}(v)$ is the neighborhood of node $v$;
- $W^{(k)}$ is a learnable weight matrix;
- $\sigma$ is a non-linear activation function.

---

### 3.4 Text Encoding

Text documents related to each team and match are encoded using transformer models.

Candidate encoders:

- BERT
- RoBERTa
- Sentence-BERT

For each match:

1. concatenate all relevant text;
2. encode it into a contextual vector.

Possible aggregation methods:

- mean pooling;
- attention pooling.

Let the set of relevant text documents for a match be:

$$
D_m = \{d_1, d_2, \dots, d_n\}
$$

A transformer encoder maps the text into an embedding:

$$
z_m^{text} = \text{Transformer}(D_m)
$$

where $z_m^{text}$ represents contextual textual information for match $m$.

---

### 3.5 Fusion Architecture

The hybrid architecture combines graph and text features.

#### Early Fusion

The graph representation and text representation are concatenated:

$$
z_m = [z_m^{graph}; z_m^{text}]
$$

The fused representation is then passed through an MLP classifier:

$$
\hat{y}_m = \text{softmax}(\text{MLP}(z_m))
$$

where $\hat{y}_m$ is the predicted probability distribution over match outcomes.

#### Cross-Attention Fusion

Cross-attention fusion can use text embeddings to attend over graph embeddings and vice versa.

This may better model interactions such as injury news affecting graph-based team strength.

For example:

$$
A = \text{Attention}(Q_{text}, K_{graph}, V_{graph})
$$

where textual queries attend to graph-based keys and values.

The final output is:

$$
\hat{y}_m = \text{softmax}(W_o z_m + b_o)
$$

where:

- $W_o$ is the output weight matrix;
- $b_o$ is the bias term;
- $\hat{y}_m$ contains probabilities for win, draw, and loss.

---

### 3.6 Training

The model is trained using cross-entropy loss:

$$
\mathcal{L} = - \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$

where:

- $N$ is the number of training examples;
- $C$ is the number of outcome classes;
- $y_{i,c}$ is the true label indicator;
- $\hat{y}_{i,c}$ is the predicted probability for class $c$.

Optimizer:

- Adam

Evaluation split:

- train/validation/test split chronologically.

Hyperparameters:

- learning rate;
- hidden dimensions;
- dropout;
- batch size.

---

### 3.7 Baselines

The proposed model will be compared against:

- logistic regression;
- random forest;
- Elo rating baseline;
- graph centrality + random walk model;
- GNN-only model;
- text-only transformer model.

Evaluation metrics:

- accuracy;
- F1-score;
- log loss;
- ROC-AUC for binary variants.

---

## 4. Anticipated Results

### 4.1 Improved Prediction Accuracy

The hybrid model is expected to outperform:

- graph-only models;
- text-only models;
- tabular baselines.

Expected accuracy improvement:

- **3–8 percentage points** over graph-only methods.

### 4.2 Better Handling of Contextual Shocks

The text component should improve sensitivity to:

- injuries;
- suspensions;
- tactical changes;
- managerial replacements.

Example:

A graph may rate Team A stronger, but injury news may shift probabilities toward Team B.

### 4.3 More Robust Generalization

Because graphs capture structural relationships and text captures contextual updates, the model should generalize across:

- leagues;
- tournaments;
- seasons.

### 4.4 Explainability Opportunities

Attention mechanisms may identify:

- influential nodes;
- influential textual phrases.

This supports interpretable predictions.

Example explanation:

> Prediction influenced by injury report and high centrality of midfield core.

---

## 5. Conclusion

This project proposes a novel **Hybrid Graph-Text Architecture for Sports Match Outcome Prediction** that integrates relational graph learning with textual contextual understanding.

The project extends prior graph-based football prediction research by moving beyond hand-crafted graph statistics such as eigenvector centrality and random walks toward learned graph embeddings via GNNs. At the same time, it incorporates rich contextual information from textual sources, addressing a major limitation of existing graph-only models.

The expected contribution is both methodological and practical:

- a multimodal framework for sports analytics;
- improved predictive performance;
- enhanced explainability.

If successful, the proposed architecture can become a general framework for prediction tasks in complex relational domains beyond sports, including finance, recommender systems, and social network forecasting.

---

## 6. References

1. Beal, R., Middleton, S., Norman, T., & Ramchurn, S. (2020). *Combining Machine Learning and Human Experts to Predict Match Outcomes in Football: A Baseline Model*.

2. Constantinou, A. C. (2019). Dolores: A model that predicts football match outcomes from all over the world. *Machine Learning, 108*, 49–75.

3. Dixon, M., & Coles, S. (1997). *Modelling association football scores and inefficiencies in the football betting market*.

4. Francis, O. (2024). *Graph-Based Modeling and Prediction of Match Outcomes in International Football Tournaments*. ResearchGate preprint.

5. Schauberger, G., & Groll, A. (2018). Predicting matches in international football tournaments with random forests. *Journal of Sports Analytics*.

6. Xenopoulos, P., & Silva, C. (2022). *Graph Neural Networks to Predict Sports Outcomes*. arXiv.

7. Groll, A., Hvattum, L. M., Ley, C., et al. (2024). *Modeling and Prediction of the UEFA EURO 2024 via Combined Statistical Learning Approaches*. arXiv.

8. Ren, Y., & Susnjak, T. (2022). *Predicting Football Match Outcomes with Explainable Machine Learning and the Kelly Index*. arXiv.

9. Ogundeji, R., et al. (2024). *A Bayesian Approach for Predicting Match Outcomes: FIFA World Cup 2026*.
