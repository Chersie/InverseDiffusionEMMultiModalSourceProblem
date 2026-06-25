# Research Manifest — Chapter 6 «Эксперименты и результаты»

**Дата**: 2026-06-10
**Цель**: Полное наполнение Главы 6 ВКР числовыми результатами экспериментов из `multirun/`.

## Источники

Все факты Главы 6 трассируются до артефактов в `multirun/`. Использованы валидационные метрики `report/val/*` из файлов `multirun/<series>/<run>/<idx>/metrics.json`. Конфигурация каждого индексированного запуска — `multirun/<series>/<run>/<idx>/.hydra/{config.yaml, overrides.yaml}`.

## R1 — Общая конфигурация для всех серий Step 0–3

Источник: `multirun/final_step1_pure/2026-06-09_07-13-02/0/.hydra/config.yaml`.

| Параметр | Значение |
|---|---|
| `data.l_max` | 5 |
| `data.n_source` | 200 |
| `data.n_train_sources` | 180 |
| `data.n_augmented` | 10 000 |
| `data.n_holdout_samples` | 100 |
| `data.batch_size` | 64 |
| `data.dropout_prob` (mode dropout) | 0.1 |
| `data.field_sigma` (additive field noise) | $10^{-8}$ |
| `data.scale_factor` | $10^{6}$ |
| `model.hidden_size` | 60 |
| `model.n_hidden_layers` | 4 |
| `model.architecture` | flat |
| `model.activation` | ELU |
| `model.dropout` | 0.001 |
| `optimiser.name` | AdamW |
| `optimiser.lr` | $10^{-3}$ |
| `optimiser.weight_decay` | 0.0 |
| `scheduler.name` | none |
| `trainer.max_epochs` | 30 |
| `callbacks.grad_clip_max_norm` | 1.0 |
| `callbacks.early_stop_patience` | 10 |

Архитектура модели — `mlp_4x60` (4 скрытых слоя ширины 60), что является компактным вариантом магистрали из § 3.1, выбранным для масштабного экспериментального сравнения.

## R2 — Step 0 baseline (single-loss baseline на коэффициентах)

Источник: `multirun/final_step1_pure/2026-06-09_07-13-02/0/metrics.json` (overrides: `loss=coef_mse`).

| Метрика | Значение |
|---|---|
| `val/coef_mse` | 0.6236 |
| `val/coef_r2` | -0.3021 |
| `val/coef_mse_amb_aware` | 0.5023 |
| `val/field_mse_w` | 41.6982 |
| `val/field_nrmse_w` | 0.9067 |
| `val/spearman_rho_P` | 0.2553 |
| `val/bin_accuracy_P` | 0.1620 |

## R3 — Step 1: чистые потери

Источник: `multirun/final_step1_pure/2026-06-09_07-13-02/{0,1}/metrics.json`.

| Конфигурация | val/field_nrmse_w | val/spearman_rho_P | val/coef_mse | val/coef_mse_amb_aware | val/bin_accuracy_P |
|---|---|---|---|---|---|
| `loss=coef_mse` (CoefMSE) | 0.9067 | 0.2553 | 0.6236 | 0.5023 | 0.1620 |
| `loss=physics_power` (PhysicsPowerLoss) | 0.1300 | 0.9625 | 0.9422 | 0.9422 | 0.5294 |

## R4 — Step 1: смешанная потеря PhysicsPowerMixed

Источник: `multirun/final_step1_mixed/2026-06-09_07-13-05/{0..4}/metrics.json` (overrides: `loss=physics_power_mixed loss.coef_aux_weight=<w>`).

| `coef_aux_weight` | val/field_nrmse_w | val/spearman_rho_P | val/coef_mse | val/coef_mse_amb_aware | val/bin_accuracy_P |
|---|---|---|---|---|---|
| 0.001 | 0.1344 | 0.9635 | 0.9603 | 0.9367 | 0.5114 |
| 0.01 | 0.1223 | 0.9631 | 0.9721 | 0.9365 | 0.5303 |
| 0.1 | 0.1267 | 0.9624 | 0.9290 | 0.9204 | 0.5017 |
| 1.0 | 0.1207 | 0.9572 | 0.9811 | 0.9383 | 0.5215 |
| 10.0 | 0.1659 | 0.9396 | 1.0013 | 0.9203 | 0.4335 |

## R5 — Step 1: ранжирующая композитная потеря PhysicsPowerRank

Источник: `multirun/final_step1_rank/2026-06-09_07-13-09/{0..4}/metrics.json` (overrides: `loss=physics_power_rank loss.rank_bin_weight=<w>`).

| `rank_bin_weight` | val/field_nrmse_w | val/spearman_rho_P | val/coef_mse | val/coef_mse_amb_aware | val/bin_accuracy_P |
|---|---|---|---|---|---|
| 0.001 | 0.1190 | 0.9626 | 0.9881 | 0.9429 | 0.5257 |
| 0.01 | 0.1318 | 0.9647 | 0.9557 | 0.8941 | 0.5163 |
| 0.1 | 0.1323 | 0.9596 | 0.9776 | 0.9113 | 0.5165 |
| 1.0 | 0.1355 | 0.9681 | 0.9424 | 0.9424 | 0.5196 |
| 10.0 | 0.1892 | 0.9605 | 0.9188 | 0.9188 | 0.4987 |

## R6 — Step 1: полная композитная потеря PhysicsPowerCombined

Источник: `multirun/final_step1_combined/2026-06-09_07-13-13/{0..8}/metrics.json` (overrides: `loss=physics_power_combined loss.coef_aux_weight=<wc> loss.rank_bin_weight=<wr>`).

| `coef_aux` | `rank_bin` | val/field_nrmse_w | val/spearman_rho_P | val/coef_mse | val/coef_mse_amb_aware | val/bin_accuracy_P |
|---|---|---|---|---|---|---|
| 0.01 | 0.01 | **0.1134** | 0.9697 | 0.9624 | 0.9350 | 0.5597 |
| 0.01 | 0.1 | 0.1238 | 0.9654 | 0.9989 | 0.9374 | 0.5187 |
| 0.01 | 1.0 | 0.1284 | 0.9726 | 0.9622 | 0.9307 | 0.5501 |
| 0.1 | 0.01 | 0.1306 | 0.9594 | 0.9884 | 0.9205 | 0.4898 |
| 0.1 | 0.1 | 0.1331 | 0.9633 | 0.9532 | 0.9436 | 0.4999 |
| 0.1 | 1.0 | 0.1284 | 0.9680 | 0.9715 | 0.9319 | 0.5426 |
| 1.0 | 0.01 | 0.1275 | 0.9625 | 0.9850 | 0.9288 | 0.4993 |
| 1.0 | 0.1 | 0.1335 | 0.9675 | 0.9645 | 0.9135 | 0.5433 |
| 1.0 | 1.0 | 0.1334 | 0.9693 | 0.9513 | 0.9513 | 0.5259 |

Лучшая конфигурация по композитной метрике $C = \text{nrmse} - 0{,}5\,\rho_P$: `coef_aux=0.01, rank_bin=0.01`, $C = -0{,}3715$.

## R7 — Step 2: признаковые представления

Источник: `multirun/final_step2_features/2026-06-09_19-16-24/{0..6}/metrics.json` (overrides: `features=<name>`). Все запуски используют лучшую конфигурацию потери из Step 1.

| Признаки | val/field_nrmse_w | val/spearman_rho_P | val/coef_mse | val/coef_mse_amb_aware | val/bin_accuracy_P |
|---|---|---|---|---|---|
| raw_flat | 0.1134 | 0.9697 | 0.9624 | 0.9350 | 0.5597 |
| raw_plus_sh | 0.1329 | 0.9643 | 0.9687 | 0.9679 | 0.5083 |
| power_pca | 0.0862 | 0.9846 | 0.9562 | 0.9386 | 0.6512 |
| power_pca_small | 0.0793 | 0.9884 | 0.9570 | 0.9542 | 0.6894 |
| pca_cv | **0.0769** | 0.9878 | 0.9588 | 0.9588 | 0.6800 |
| cv_only | 0.5672 | 0.3891 | 0.9325 | 0.9325 | 0.1315 |
| subsample_stride4 | 0.1016 | 0.9769 | 0.9516 | 0.9231 | 0.5848 |

Композитная метрика $C = \text{nrmse} - 0{,}5\,\rho_P$:
- pca_cv: $C = -0{,}4170$ — лучший
- power_pca_small: $C = -0{,}4149$
- power_pca: $C = -0{,}4061$
- subsample_stride4: $C = -0{,}3869$
- raw_flat: $C = -0{,}3715$
- raw_plus_sh: $C = -0{,}3493$
- cv_only: $C = +0{,}3727$ — провал (одни SH-проекции теряют слишком много информации)

## R8 — Step 3: расписание обучения многоголовой модели

Источник: `multirun/final_step3_scheduling/2026-06-10_00-23-36/{0..5}/metrics.json` (overrides: `training.backbone_policy=<p> training.truncate_target_to_active_band=<t>`). Все запуски используют лучшие конфигурации потери (PhysicsPowerCombined) и признаков (pca_cv) из предыдущих шагов; модель — многоголовая `MultiHeadMLP` (§ 3.2).

| backbone_policy | truncate_target | val/field_nrmse_w | val/spearman_rho_P | val/coef_mse | val/coef_mse_amb_aware | val/bin_accuracy_P |
|---|---|---|---|---|---|---|
| freeze_after_stage1 | true | 0.3861 | 0.7653 | 1.0192 | 0.9407 | 0.2508 |
| freeze_after_stage1 | false | 0.2518 | 0.8831 | 1.0657 | 0.9763 | 0.3386 |
| trainable_always | true | **0.0784** | 0.9880 | 0.9852 | 0.9745 | 0.6861 |
| trainable_always | false | 0.0819 | 0.9879 | 0.9736 | 0.9446 | 0.6897 |
| all_trainable_active_boost | true | 0.1094 | 0.9805 | 0.9771 | 0.9689 | 0.6273 |
| all_trainable_active_boost | false | 0.0968 | 0.9824 | 0.9650 | 0.9556 | 0.6420 |

Композитная метрика $C = \text{nrmse} - 0{,}5\,\rho_P$:
- trainable_always × truncate=true: $C = -0{,}4156$ — лучший
- trainable_always × truncate=false: $C = -0{,}4121$
- all_trainable_active_boost × truncate=false: $C = -0{,}3944$
- all_trainable_active_boost × truncate=true: $C = -0{,}3808$
- freeze_after_stage1 × truncate=false: $C = -0{,}1898$
- freeze_after_stage1 × truncate=true: $C = +0{,}0035$ — провал

## R9 — Итоговая лучшая конфигурация (по композитной метрике на val)

| Слой | Выбор | Значение метрики $C$ на val |
|---|---|---|
| Потеря | PhysicsPowerCombined, coef_aux=0.01, rank_bin=0.01 | $C = -0{,}3715$ (с raw_flat признаками) |
| Признаки | pca_cv | $C = -0{,}4170$ (с одноголовой моделью) |
| Расписание | многоголовая модель `MultiHeadMLP`, backbone_policy=trainable_always, truncate_target_to_active_band=true | $C = -0{,}4156$ |

Лучший val/field_nrmse_w = 0.0769 (Step 2: одноголовая MLP с pca_cv).
Лучший val/spearman_rho_P = 0.9884 (Step 2: одноголовая MLP с power_pca_small).
В многоголовой постановке (Step 3) с trainable_always + truncate=true: val/field_nrmse_w = 0.0784, val/spearman_rho_P = 0.9880 — то же качество, что и одноголовая модель Step 2, при разрешённости вверх по $L$ (§ 3.2).

## Источники, не переиспользуемые в тексте

- `final_step1_rank/2026-06-09_04-24-28` и `final_step1_rank/2026-06-09_04-25-16` — smoke-тесты на 1 эпохе с 20 источниками и `data.smoke_test=True`. Не входят в выводы.
- `final_step1_pure/2026-06-09_05-26-42`, `final_step1_mixed/2026-06-09_05-26-49`, `final_step1_rank/2026-06-09_05-27-13`, `final_step1_combined/2026-06-09_05-27-25` — ранние запуски с теми же параметрами; на их месте использованы последующие полные сеточные запуски (`07-13-*`).
- `final_step2_features/2026-06-09_19-09-17` — частичный запуск с одной конфигурацией; на его месте использован полный (`19-16-24`).
- В тексте Главы 6 отчёт ведётся только по val-метрикам (как требует постановка задачи).
