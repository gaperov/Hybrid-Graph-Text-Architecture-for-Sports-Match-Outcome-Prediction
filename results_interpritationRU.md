
# Итоговый отчёт  
# Hybrid Graph-Text Architecture for Sports Match Outcome Prediction

## 1. Краткое описание проекта

Цель проекта — предсказание исхода матча FIFA World Cup в постановке многоклассовой классификации:

| Класс | Значение |
|---:|---|
| 0 | home_win |
| 1 | draw |
| 2 | away_win |

Проект был построен как hybrid sports analytics pipeline, объединяющий несколько типов признаков:

- исторические матчевые данные World Cup;
- FIFA ranking features;
- rolling form features;
- historical World Cup performance;
- tournament cumulative features;
- rest days;
- stage features;
- leakage-safe pre-match text features;
- sentence-transformer text embeddings;
- graph representation World Cup entities;
- baseline ML models;
- exploratory GNN models.

Основной modeling subset охватывает чемпионаты мира с 1994 по 2022 год.  
Причина выбора 1994 года как стартовой точки — доступность FIFA rankings с 1992 года и наличие полного набора ranking/form/text features для турниров 1994–2022.

Использованный temporal split:

| Split | Годы |
|---|---|
| Train | 1994, 1998, 2002, 2006, 2010, 2014 |
| Validation | 2018 |
| Test | 2022 |

Такой split выбран для предотвращения tournament leakage и имитирует реальный сценарий прогнозирования будущего турнира на основе прошлых.

---

## 2. Проделанная работа

В рамках проекта были выполнены следующие этапы.

### 2.1. Подготовка model-ready датасета

Был подготовлен основной датасет:


data/processed/match_dataset_model_ready


Он содержит 500 матчей World Cup за период 1994–2022.

В датасете были подготовлены:

- target;
- FIFA ranking features;
- rolling form features;
- historical World Cup form;
- tournament cumulative features;
- rest days;
- stage features;
- text availability features;
- leakage-safe pre-match text;
- missing value handling;
- leakage-safe preparation.

Из признаков были исключены потенциально leakage-признаки:

- `home_score`;
- `away_score`;
- `result`;
- `target`;
- `home_xg`;
- `away_xg`;
- `xg`;
- post-match text.

---

### 2.2. Подготовка текстового слоя

Был построен leakage-safe text layer на основе Guardian pre-match articles.

Использовалось правило:

```text
published_at <= match_date - 6 hours
```

То есть в признаки попадали только тексты, опубликованные минимум за 6 часов до матча.

Для текстовых embedding использовалась модель:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Размерность embedding:

```text
384
```

Для матчей без текста использовался zero vector.

Дополнительно в табличный датасет были добавлены text meta features:

- `text_available`;
- `text_count`;
- `log_text_count`;
- `guardian_text_flag`;
- `no_text_flag`;
- временные признаки доступности текста.

---

### 2.3. Baseline-модели

Был проведён ablation-анализ baseline-моделей на разных группах признаков:

- majority baseline;
- stage-only models;
- text-meta-only models;
- text-only models;
- form models;
- World Cup history models;
- ranking models;
- combined numeric models;
- numeric + text embeddings models;
- simple graph feature models;
- numeric + text + simple graph models.

Использовались разные семейства моделей:

- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- HistGradientBoosting;
- XGBoost;
- MLP;
- simple graph feature models.

Основной файл результатов:

```text
data/processed/model_results_ablation.csv
```

---

### 2.4. Graph layer

Был подготовлен graph layer с сущностями:

- Team;
- Match;
- Tournament;
- Stadium;
- Referee;
- TeamTournament;
- Manager, если доступен.

И связями:

- `team_played_match`;
- `team_played_against_team`;
- `team_participated_in_tournament`;
- `match_belongs_to_tournament`;
- `match_played_at_stadium`;
- `referee_officiated_match`;
- `team_has_tournament_instance`;
- `team_tournament_played_match`;
- `manager_managed_team_in_tournament`.

Были подготовлены файлы:

```text
data/processed/graph_nodes_model_ready
data/processed/graph_edges_model_ready
```

Сначала были протестированы простые graph features:

- home team degree;
- away team degree;
- degree difference;
- centrality features.

Эти признаки сами по себе не дали сильного результата.

---

### 2.5. GNN prototype

На GNN-этапе был реализован exploratory transductive Team-Match GNN prototype.

Был выбран упрощённый граф:

```text
Team <-> Match
```

Основная причина — small-data setting.  
Полный heterograph с большим числом типов узлов и рёбер мог бы быстро переобучиться на 500 матчах.

Использованная архитектура:

- PyTorch Geometric `HeteroData`;
- node types:
  - `team`;
  - `match`;
- edge types:
  - `team -> match`;
  - `match -> team`;
  - optional `team -> team`;
- HeteroConv;
- SAGEConv;
- hidden dimension 32 или 64;
- 1–2 message passing layers;
- dropout 0.35–0.50;
- classifier на Match node embeddings;
- weighted CrossEntropyLoss;
- early stopping по validation macro-F1.

Для Match nodes использовались варианты признаков:

1. numeric only;
2. numeric + text PCA;
3. numeric + text PCA + text meta/stage features.

Text embeddings были снижены через PCA до 32 измерений. PCA обучался только на train split.

Важно: текущая GNN была transductive prototype, то есть граф включал узлы/рёбра 1994–2022. Labels использовались только через train mask, однако структурная информация будущих матчей потенциально может создавать leakage. Поэтому результаты GNN следует интерпретировать как exploratory, а не как финальный leakage-safe benchmark.

---

## 3. Сравнительный анализ результатов

Для анализа использовались два файла:

1. `model_results_ablation.csv` — результаты всех baseline-моделей.
2. `gnn_vs_baseline_comparison.csv` — сравнение ключевых baseline-моделей и GNN.

Ключевая метрика выбора модели:

```text
validation macro-F1
```

Причина:

- задача многоклассовая;
- классы несбалансированы;
- draw является сложным и миноритарным классом;
- test 2022 не должен использоваться для выбора модели.

Дополнительные важные метрики:

- balanced accuracy;
- weighted F1;
- draw recall;
- draw F1;
- test macro-F1 как финальная контрольная оценка.

---

## 4. Общая таблица сильнейших моделей

Ниже приведены ключевые модели из baseline и GNN comparison.

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

## 5. Интерпретация baseline-результатов

### 5.1. Majority baseline

Majority baseline показал:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.406 | 0.453 |
| Macro-F1 | 0.193 | 0.208 |
| Draw F1 | 0.000 | 0.000 |

Это ожидаемо: majority classifier почти всегда предсказывает наиболее частый класс и полностью игнорирует ничьи.

Все сильные модели существенно превосходят majority baseline, особенно по macro-F1.

---

### 5.2. Лучший baseline по validation

Лучшей моделью по validation macro-F1 стала:

```text
ranking_compact_rf
```

Результаты:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.594 | 0.500 |
| Balanced accuracy | 0.574 | 0.478 |
| Macro-F1 | **0.565** | 0.469 |
| Weighted F1 | 0.597 | 0.498 |
| Draw recall | **0.462** | 0.267 |
| Draw F1 | **0.400** | 0.276 |

Эта модель особенно хорошо выглядит на validation, в том числе по draw class.

Интерпретация:

- compact ranking features являются очень сильным сигналом;
- FIFA ranking отражает относительную силу команд;
- Random Forest хорошо работает на малых табличных данных;
- компактный набор признаков снижает риск переобучения.

Однако test macro-F1 у этой модели ниже, чем у некоторых других моделей. Это говорит о том, что модель хорошо подошла под турнир 2018, но хуже перенеслась на 2022.

---

### 5.3. Лучший hybrid text baseline

Лучшей hybrid text baseline стала:

```text
numeric_text_rf
```

Результаты:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | **0.641** | **0.578** |
| Balanced accuracy | 0.562 | **0.552** |
| Macro-F1 | 0.546 | **0.554** |
| Weighted F1 | **0.606** | **0.577** |
| Draw recall | 0.154 | 0.400 |
| Draw F1 | 0.235 | **0.462** |

Интерпретация:

- добавление text embeddings к numeric features улучшило test performance;
- текстовые признаки, вероятно, содержат информацию о контексте матча, ожиданиях, травмах, форме команд, настроениях и narrative вокруг матча;
- Random Forest смог использовать текстовые embedding-признаки лучше, чем линейные и нейросетевые модели;
- на test 2022 эта модель дала лучший overall result.

Однако validation draw recall у неё низкий:

```text
draw_recall_val = 0.154
```

Это говорит о том, что модель на validation плохо находила ничьи, несмотря на хороший общий macro-F1.

---

### 5.4. Combined numeric baseline

Модель:

```text
combined_numeric_rf_safe
```

Результаты:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.531 | 0.547 |
| Balanced accuracy | 0.483 | 0.550 |
| Macro-F1 | 0.490 | 0.539 |
| Weighted F1 | 0.548 | 0.552 |
| Draw recall | 0.231 | **0.533** |
| Draw F1 | 0.194 | 0.457 |

Интерпретация:

- модель не является лучшей на validation;
- однако показывает сильный test macro-F1;
- особенно хорошо ловит ничьи на test;
- numeric признаки без текста уже содержат значительный предиктивный сигнал.

Это важный результат: даже без текстового слоя combined numeric features являются сильной основой.

---

### 5.5. XGBoost и boosting-модели

Модель:

```text
xgboost_combined_numeric
```

показала:

| Metric | Validation | Test |
|---|---:|---:|
| Macro-F1 | 0.479 | 0.531 |
| Draw F1 | 0.188 | 0.414 |

Модель не стала лучшей, но дала достаточно сильный test result.

Gradient Boosting и HistGradientBoosting также показывали конкурентные результаты, но не превзошли Random Forest baselines.

Интерпретация:

- tree-based methods в целом хорошо подходят для данной задачи;
- Random Forest оказался более стабильным на validation;
- boosting-модели могут быть чувствительны к малому размеру выборки и настройкам гиперпараметров.

---

### 5.6. Линейные модели

Логистические регрессии на ranking и combined numeric features дали средние результаты:

Например:

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

Интерпретация:

- задача явно нелинейная;
- взаимодействия между ranking, form, stage, rest и tournament context важны;
- линейные модели плохо справляются с draw class.

---

### 5.7. Text-only models

Text-only модели дали неоднозначные результаты.

Например:

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

Text-only модели слабее hybrid numeric-text моделей по overall macro-F1, но в некоторых случаях достаточно хорошо ловят ничьи на test.

Интерпретация:

- текст сам по себе недостаточен для стабильного прогноза исхода;
- но текст может содержать специфический сигнал для ничьих или неопределённых матчей;
- zero vectors для матчей без текста и source-specific Guardian bias ограничивают качество text-only подхода;
- текст лучше использовать как дополнительный слой поверх numeric features.

---

### 5.8. Stage-only и text-meta-only models

Модель:

```text
stage_logreg
```

показала низкий overall macro-F1:

```text
val macro-F1  = 0.201
test macro-F1 = 0.228
```

Но при этом высокий draw recall:

```text
val draw recall  = 0.769
test draw recall = 0.800
```

Это значит, что stage-признаки склоняют модель к предсказанию ничьих в определённых стадиях, но overall качество низкое.

Модель:

```text
text_meta_logreg
```

также не дала хорошего overall качества:

```text
val macro-F1  = 0.172
test macro-F1 = 0.320
```

Интерпретация:

- stage и text meta не являются самостоятельными сильными predictors;
- но могут быть полезны как дополнительные признаки;
- высокий draw recall при низком macro-F1 указывает на плохую калибровку и перекос в сторону draw.

---

### 5.9. Simple graph features

Simple graph baseline:

```text
simple_graph_logreg
```

показал:

| Metric | Validation | Test |
|---|---:|---:|
| Macro-F1 | 0.365 | 0.478 |
| Draw F1 | 0.154 | 0.222 |

Интерпретация:

- простые degree/centrality features недостаточны;
- графовая структура в простом агрегированном виде не раскрывает полезный сигнал;
- при этом test macro-F1 0.478 выше majority baseline, значит некоторый structural signal есть, но он слабый.

---

## 6. Интерпретация GNN-результатов

В GNN comparison были получены следующие результаты.

| Model | Feature set | Val macro-F1 | Test macro-F1 | Val draw F1 | Test draw F1 |
|---|---|---:|---:|---:|---:|
| `gnn_team_match_numeric_text_meta` | numeric_text_pca32_extra | **0.520** | 0.453 | 0.261 | 0.231 |
| `gnn_match_numeric_only` | numeric_only | 0.505 | 0.477 | 0.250 | 0.333 |
| `gnn_team_match_numeric_text` | numeric_text_pca32 | 0.489 | 0.458 | 0.231 | 0.231 |
| `gnn_team_match_numeric_text_small` | numeric_text_pca32 | 0.483 | **0.530** | **0.276** | **0.357** |

---

### 6.1. Лучший GNN по validation

Лучшей GNN-моделью по validation стала:

```text
gnn_team_match_numeric_text_meta
```

Результаты:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.578 | 0.500 |
| Balanced accuracy | 0.522 | 0.462 |
| Macro-F1 | **0.520** | 0.453 |
| Weighted F1 | 0.571 | 0.494 |
| Draw recall | 0.231 | 0.200 |
| Draw F1 | 0.261 | 0.231 |

Эта модель уступает двум основным baseline:

```text
ranking_compact_rf val macro-F1 = 0.565
numeric_text_rf val macro-F1    = 0.546
```

То есть GNN не улучшила validation performance относительно сильных baseline.

---

### 6.2. Лучший GNN по test

Лучшей GNN по test стала:

```text
gnn_team_match_numeric_text_small
```

Результаты:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.516 | 0.563 |
| Balanced accuracy | 0.482 | 0.535 |
| Macro-F1 | 0.483 | **0.530** |
| Weighted F1 | 0.523 | **0.564** |
| Draw recall | 0.308 | 0.333 |
| Draw F1 | 0.276 | 0.357 |

Этот результат интересен: более маленькая GNN хуже на validation, но лучше на test.

Интерпретация:

- меньшая модель, вероятно, меньше переобучается;
- hidden_dim=32 и высокий dropout лучше соответствуют размеру данных;
- сложные GNN-варианты могут переобучаться на один validation tournament.

Но выбирать эту модель как финальную по test нельзя, так как test не должен использоваться для model selection.

---

### 6.3. Вклад текста в GNN

Сравнение GNN-вариантов:

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

Вывод:

- добавление text PCA само по себе не улучшило GNN;
- добавление text meta features улучшило validation macro-F1;
- dense text embeddings в GNN используются хуже, чем в Random Forest;
- text availability и metadata могут быть более полезны для GNN, чем сами PCA embeddings.

Возможные причины:

1. 32-dimensional PCA теряет часть семантической информации.
2. Нейросетевой text branch переобучается на малом количестве матчей.
3. Zero-vector для отсутствующего текста сложно интерпретировать без explicit missing indicator.
4. Guardian coverage может создавать source-specific bias.
5. Tree-based model лучше извлекает нелинейные паттерны из tabular + embedding features.

---

### 6.4. GNN и draw class

Draw class остаётся сложным.

Лучший GNN по validation draw F1:

```text
gnn_team_match_numeric_text_small
draw F1 val = 0.276
```

Лучший baseline по validation draw F1:

```text
ranking_compact_rf
draw F1 val = 0.400
```

Лучший baseline по test draw F1:

```text
numeric_text_rf
draw F1 test = 0.462
```

Лучший GNN по test draw F1:

```text
gnn_team_match_numeric_text_small
draw F1 test = 0.357
```

Вывод:

```text
GNN пока не решает проблему draw prediction лучше сильных baseline.
```

---

## 7. Общий сравнительный вывод

### 7.1. Лучшая модель по validation

Основной критерий выбора модели — validation macro-F1.

По этому критерию лучшая модель:

```text
ranking_compact_rf
```

С результатом:

```text
validation macro-F1 = 0.565
```

Эта модель также имеет лучший validation draw F1:

```text
draw F1 val = 0.400
```

Следовательно, если выбирать модель строго по validation, то финальной model-selected baseline является `ranking_compact_rf`.

---

### 7.2. Лучшая модель по test

Лучший test macro-F1 показывает:

```text
numeric_text_rf
```

С результатом:

```text
test macro-F1 = 0.554
```

Также эта модель показывает лучший test draw F1 среди ключевых моделей:

```text
draw F1 test = 0.462
```

Однако test используется только для финальной оценки, а не для выбора модели. Поэтому `numeric_text_rf` следует интерпретировать как лучший hybrid text baseline и как наиболее успешную модель на holdout 2022, но не как validation-selected winner.

---

### 7.3. Положение GNN относительно baseline

GNN-модели дали конкурентное, но не превосходящее качество.

Лучшая GNN по validation:

```text
gnn_team_match_numeric_text_meta
val macro-F1 = 0.520
```

Она уступает:

```text
ranking_compact_rf: 0.565
numeric_text_rf:    0.546
```

Лучшая GNN по test:

```text
gnn_team_match_numeric_text_small
test macro-F1 = 0.530
```

Она близка к:

```text
combined_numeric_rf_safe test macro-F1 = 0.539
```

но уступает:

```text
numeric_text_rf test macro-F1 = 0.554
```

Итог:

```text
Simplified Team-Match GNN did not outperform strong tabular/text baselines.
```

---

## 8. Почему GNN не улучшила baseline

Наиболее вероятные причины.

### 8.1. Малый размер данных

500 матчей — очень мало для GNN.

GNN учит:

- node projections;
- message passing layers;
- relation aggregation;
- classifier;
- fusion numeric/text features.

Random Forest на малых табличных данных обычно более устойчив.

---

### 8.2. Простая Team-Match topology добавляет мало нового сигнала

Сильные ranking/form features уже кодируют большую часть информации о силе команд.

Team-Match graph в World Cup имеет ограниченную плотность:

- команды играют мало матчей;
- турниры редкие;
- между чемпионатами составы команд сильно меняются;
- static Team node смешивает разные поколения одной команды.

Например, один и тот же Team node для Brazil объединяет Brazil 1994, Brazil 2002, Brazil 2022, хотя это разные команды по составу и силе.

---

### 8.3. Transductive graph prototype не является финальным leakage-safe benchmark

Текущая GNN использовала transductive graph 1994–2022.  
Даже с этим потенциальным преимуществом она не превзошла baseline.

Это говорит о том, что сама topology в текущем виде не даёт сильного дополнительного сигнала.

---

### 8.4. Text embeddings лучше раскрываются в Random Forest

`numeric_text_rf` показал лучший test result.

Значит, текстовые признаки полезны, но текущая GNN не смогла использовать их лучше tree-based модели.

---

### 8.5. Validation и test — по одному турниру

Validation = World Cup 2018.  
Test = World Cup 2022.

Каждый split содержит один турнир, поэтому метрики чувствительны к:

- малому числу матчей;
- распределению ничьих;
- особенностям конкретного турнира;
- неожиданным результатам;
- изменению силы команд.

Разница в 0.02–0.05 macro-F1 может быть нестабильной.

---

## 9. Основные выводы

1. **Все сильные модели существенно превосходят majority baseline.**

2. **Лучшей моделью по validation macro-F1 является `ranking_compact_rf`.**

   ```text
   val macro-F1 = 0.565
   ```

3. **Лучшей моделью по test macro-F1 является `numeric_text_rf`.**

   ```text
   test macro-F1 = 0.554
   ```

4. **Text embeddings полезны в hybrid setting.**

   Text-only модели недостаточно сильные, но numeric + text модель показывает лучший test result.

5. **Tree-based models оказались наиболее эффективными.**

   Random Forest стабильно превосходит линейные модели, MLP и текущие GNN-варианты.

6. **Simple graph features не раскрывают потенциал графа.**

   Degree/centrality признаки дают слабый сигнал.

7. **Simplified Team-Match GNN не улучшила сильные baseline.**

   Лучшая GNN по validation:

   ```text
   gnn_team_match_numeric_text_meta
   val macro-F1 = 0.520
   ```

   Это ниже, чем у `ranking_compact_rf` и `numeric_text_rf`.

8. **Меньшая GNN выглядит более устойчивой на test.**

   ```text
   gnn_team_match_numeric_text_small
   test macro-F1 = 0.530
   ```

   Это указывает, что для текущего датасета нужны более регуляризованные neural models.

9. **Draw prediction остаётся главной сложностью.**

   Лучшие draw F1 достигаются разными моделями на validation и test, что говорит о нестабильности draw class.

10. **Текущий GNN-этап следует рассматривать как exploratory extension, а не как финальную production-модель.**

---

## 10. Рекомендованная финальная интерпретация моделей

Для отчёта разумно выделить три категории моделей.

### 10.1. Primary validation-selected model

```text
ranking_compact_rf
```

Почему:

- лучший validation macro-F1;
- лучший validation draw F1;
- компактный набор признаков;
- высокая интерпретируемость;
- низкая сложность.

---

### 10.2. Best hybrid text model

```text
numeric_text_rf
```

Почему:

- лучший test macro-F1;
- лучший test weighted F1;
- лучший test draw F1 среди ключевых моделей;
- демонстрирует пользу leakage-safe pre-match text embeddings.

---

### 10.3. Exploratory graph model

```text
gnn_team_match_numeric_text_meta
```

Почему:

- лучшая GNN по validation;
- использует numeric, text PCA и meta features;
- демонстрирует работоспособность GNN pipeline;
- но не превосходит сильные baseline.

---

## 11. Дальнейшие пути развития

### 11.1. Multi-seed evaluation

Текущие GNN-результаты нужно проверить на устойчивость.

Рекомендуется запустить GNN с несколькими seed:

```text
1, 2, 3, 4, 5, 42, 123, 777, 2024, 2025
```

И считать:

- mean validation macro-F1;
- std validation macro-F1;
- mean test macro-F1;
- std test macro-F1;
- mean draw F1;
- std draw F1.

Если стандартное отклонение велико, отдельные отличия между моделями неустойчивы.

---

### 11.2. MLP sanity check

Нужно понять, даёт ли message passing реальную пользу.

Следует сравнить:

| Model | Features |
|---|---|
| MLP | numeric only |
| MLP | numeric + text PCA |
| MLP | numeric + text PCA + meta |
| GNN | numeric + text PCA + meta |

Если MLP показывает качество не хуже GNN, значит graph topology в текущем виде не добавляет полезного сигнала.

---

### 11.3. Более маленькие GNN

Так как `gnn_team_match_numeric_text_small` показала лучший test result среди GNN, стоит продолжить эксперименты с более регуляризованными моделями.

Рекомендуемый grid:

```text
hidden_dim: 16, 32
num_layers: 1, 2
dropout: 0.45, 0.50, 0.60
weight_decay: 3e-4, 1e-3
text_pca_dim: 16, 32
```

Основная гипотеза:

```text
меньшие и более регуляризованные GNN будут стабильнее на small-data setting.
```

---

### 11.4. Переход к TeamTournament graph

Наиболее перспективное graph-направление — заменить static Team nodes на TeamTournament nodes.

Текущая структура:

```text
Team -> Match
```

Проблема:

```text
один Team node смешивает разные поколения команды across tournaments
```

Лучше:

```text
TeamTournament -> Match
```

Примеры узлов:

```text
Brazil_1994
Brazil_1998
Brazil_2002
Argentina_2022
France_2018
```

Такой graph лучше отражает реальность, потому что команда в конкретном турнире — более корректная единица анализа, чем национальная команда за 30 лет.

Возможные признаки TeamTournament node:

- FIFA rank before tournament;
- average rank before tournament;
- pre-tournament form;
- historical World Cup stats before tournament;
- previous World Cup performance;
- manager features;
- squad strength, если будет добавлена позже;
- tournament cumulative features до конкретного матча.

Гипотеза:

```text
TeamTournament-Match graph даст более полезный graph signal, чем Team-Match graph.
```

---

### 11.5. Time-aware graph benchmark

Для финального graph benchmark нужно уйти от transductive prototype.

Рекомендуемый leakage-safer вариант:

```text
train graph: edges/nodes до 2014 включительно
val graph: historical graph до начала World Cup 2018 + val match nodes без labels
test graph: historical graph до начала World Cup 2022 + test match nodes без labels
```

Ещё более строгий вариант:

```text
для каждого матча использовать только edges с timestamp < match_time
```

Это позволит избежать future structural leakage.

---

### 11.6. Full heterograph только после TeamTournament

Full heterograph может включать:

- Team;
- TeamTournament;
- Match;
- Tournament;
- Stadium;
- Referee;
- Manager.

Но не стоит сразу переходить к full heterograph.

Рекомендуемый порядок:

1. Team-Match GNN — уже реализовано.
2. TeamTournament-Match GNN.
3. Добавить Tournament node.
4. Добавить Stadium node.
5. Добавить Referee node.
6. Добавить Manager node, если покрытие достаточно хорошее.

Каждый шаг должен сопровождаться ablation.

---

### 11.7. Улучшение text branch

Возможные направления:

1. Использовать разные PCA dimensions:

   ```text
   16, 32, 64, 128
   ```

2. Сравнить PCA embeddings с raw 384-dimensional embeddings.

3. Добавить learned missing text embedding вместо zero vector.

4. Обязательно сохранять explicit indicator:

   ```text
   text_available
   no_text_flag
   log_text_count
   ```

5. Проверить robustness к Guardian source bias.

Гипотеза:

```text
text embeddings полезны, но для GNN нужен более осторожный fusion mechanism.
```

---

### 11.8. Улучшение draw prediction

Draw class является самым сложным.

Возможные идеи:

1. Подбор class weights.
2. Focal loss вместо CrossEntropyLoss.
3. Two-stage model:

   ```text
   stage 1: non-draw vs draw
   stage 2: home_win vs away_win
   ```

4. Калибровка вероятностей.
5. Оптимизация threshold для draw на validation.
6. Добавление признаков равенства команд:

   - rank difference close to zero;
   - form difference close to zero;
   - similar historical strength;
   - knockout/group stage;
   - rest difference;
   - market odds, если когда-нибудь будут доступны.

Гипотеза:

```text
draw лучше моделировать как отдельный режим неопределённости, а не просто как один из трёх классов.
```

---

## 12. Итоговая рекомендация

На текущем этапе финальная оценка проекта должна быть сформулирована следующим образом:

```text
The project successfully built a hybrid tabular-text-graph pipeline for FIFA World Cup outcome prediction.
Strong tabular and hybrid text baselines were established.
The best validation-selected model was ranking_compact_rf with validation macro-F1 of 0.565.
The best hybrid text model, numeric_text_rf, achieved the highest test macro-F1 of 0.554.
The simplified Team-Match GNN prototype was successfully implemented but did not outperform strong Random Forest baselines.
This suggests that, in the current small-data World Cup setting, simple graph topology does not add enough predictive signal beyond ranking, form and text features.
The most promising next direction is a leakage-safe TeamTournament-Match graph with time-aware snapshots and stronger regularization.
```

Кратко на русском:

```text
Проект успешно реализовал hybrid tabular-text-graph pipeline для прогнозирования исходов матчей World Cup.
Сильнейшей моделью по validation стала ranking_compact_rf.
Лучшей hybrid text моделью по test стала numeric_text_rf.
GNN pipeline был реализован и протестирован, но не превзошёл сильные baseline.
В текущем виде простая Team-Match graph topology не добавляет достаточно информации сверх ranking/form/text features.
Наиболее перспективное развитие — TeamTournament-Match graph, time-aware graph construction и более регуляризованные GNN-модели.
```
```