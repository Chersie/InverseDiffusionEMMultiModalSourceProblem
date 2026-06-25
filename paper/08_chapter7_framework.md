# Глава 7. Программный фреймворк `mpinv`

> Статус главы: **DRAFT**. Программный фреймворк ещё может быть видоизменён в последующих итерациях работы (рефакторинг отдельных модулей, перераспределение ответственности между слоями `cli` и `training`, доукомплектование реестров). В настоящей редакции зафиксированы пакетная архитектура, маппинг «глава работы — модуль фреймворка» и архитектурные инварианты в том виде, в котором они на момент защиты обеспечивают воспроизводимость результатов Главы 6.

Программный фреймворк, реализующий описанные в Главах 2–5 пайплайн данных, архитектуры моделей, функции потерь, метрики и наборы оценки, оформлен как самостоятельный Python-пакет `mpinv` с собственными точками входа командной строки и единой системой композиции конфигураций. Фреймворк решает три практические задачи: (а) служит воспроизводимой кодовой базой, по которой Глава 6 заявлена как бенчмарк для последующих исследований; (б) обеспечивает один-к-одному соответствие между разделами текста работы и местом в коде, где соответствующее методическое решение реализовано; (в) фиксирует архитектурные инварианты — угловую сетку, упаковку коэффициентов и реестры рабочих имён — в одном источнике истины, исключая дублирование констант между модулями. § 7.1 описывает пакетную архитектуру; § 7.2 содержит таблицу маппинга «секция работы — модуль `mpinv`»; § 7.3 фиксирует ключевые архитектурные инварианты.

## 7.1. Архитектура пакета `mpinv`

Пакет `mpinv` версии 0.1.0 (определена в [src/mpinv/__init__.py](src/mpinv/__init__.py)) разделён на десять подпакетов, каждый из которых отвечает за свой слой обратной задачи. Слои упорядочены от низкоуровневых констант к высокоуровневым точкам входа.

**`mpinv.core`** содержит источник истины по угловой сетке, упаковке коэффициентов, типам и инициализации псевдослучайных генераторов: модули `grid.py` (класс `GridSpec` и константа `GRID_DEFAULT` § 1.2), `packing.py` (функции `pack_coefficients`, `unpack_coefficients` и константы `L_MAX`, `K_MODES`, `PACKED_DIM` § 1.1), `area_weights.py` (весовая функция площади $\mu(\theta)$ из § 1.3), `seeds.py`, `shapes.py`, `types.py`.

**`mpinv.data`** реализует пайплайн генерации обучающих выборок Главы 2: `synthetic_generator.py` — четыре режима синтетики § 2.1; `augment.py` — пять примитивов аугментации § 2.1; `real_antenna_loader.py` и `real_augmented_pipeline.py` — обработка реальных антенн § 2.2; `basis_decomposer.py` — аналитический декомпозитор $\mathbf B$ из § 1.3, используемый и при проекции реальных данных на $L = 5$; `splits.py` — разбиение по идентификатору источника § 2.2; `memmap_dataset.py` — поддержка ленивой загрузки больших выборок; `dummy_probe.py` — генерация dummy-данных по § 2.1 (используются как набор оценки § 5.5.3).

**`mpinv.features`** реализует признаковые пайплайны § 2.3–2.5: `raw_flat.py`, `subsample.py`, `pca.py`, `fft_radial.py`, `sh_power.py`, `composite.py` (составные стеки `pca_cv`, `raw_plus_sh`), `normalisers.py` (стандартизатор § 2.6), `power_pipeline.py` (конвейер преобразований от $P$ к финальному признаковому вектору), `registry.py` (реестр `FEATURE_EXTRACTORS`).

**`mpinv.models`** реализует архитектуры § 3.1–3.2: `mlp.py` (магистраль `flat`, конфигурация `mlp_5x200`), `multi_head_mlp.py` (многоголовая модель с Матрёшкой), `base.py` (общий базовый класс), `linear_baselines.py` (служебные линейные модели для проверочных и отладочных запусков; в основной экспериментальной серии работы они не задействованы, поскольку методически в архитектурном перечне Главы 3 не значатся), `registry.py` (реестр `MODELS`).

**`mpinv.losses`** реализует функции потерь § 4.1–4.4 и дифференцируемый декодер § 3.3: `coef_mse.py` (`CoefMSE`), `physics_power.py` (`PhysicsPowerLoss` и композиты), `rank_bin.py` (`RankBinPLoss`), `differentiable_field.py` (`DifferentiableMultipoleField`), `registry.py` (реестр `LOSSES`).

**`mpinv.training`** реализует тренировочный цикл § 4.5–4.6: `trainer.py` (`Trainer`), `staged.py` (`StagedTrainer` для Матрёшки), `optim.py` (фабрики оптимизатора и шедулера § 4.7), `amp.py` (поддержка обучения со смешанной точностью), `sanity.py` (предобучаемые проверки корректности конфигурации перед запуском).

**`mpinv.callbacks`** содержит общий базовый класс `Callback` и семь конкретных колбэков, через которые тренер реализует управление обучением (§ 4.5): `CheckpointCallback`, `EarlyStoppingCallback`, `GradClipCallback`, `LoggingCallback`, `MemoryWatchdogCallback`, `TimingCallback`, `ValidationCallback`. Все колбэки экспортируются через [src/mpinv/callbacks/__init__.py](src/mpinv/callbacks/__init__.py).

**`mpinv.analysis`** реализует метрики Главы 5, а также визуализации и сводные отчёты Главы 6: подпакет `metrics/` содержит `coefficient_metrics.py`, `field_metrics.py`, `mode_metrics.py`; подпакет `plots/` — построители графиков (кривые потерь, гистограммы коэффициентов, диаграммы сравнения полей, разбиение ошибки по порядкам $l$, распределения коэффициента детерминации и т. д.); подпакет `reports/` — сборка сводного отчёта по результатам экспериментальной серии.

**`mpinv.tracking`** обеспечивает интеграцию с системой учёта экспериментов MLflow [11] (версия не ниже 3.10): `mlflow_sink.py` (`MLflowSink` и `MLflowSinkConfig`), `dataset_logger.py` (`DatasetSpec`, `log_numpy_dataset` — фиксация состава обучающей и валидационной выборок), `params.py` (приведение сложенных конфигурационных словарей к плоскому виду для MLflow).

**`mpinv.cli`** содержит точки входа командной строки, перечисленные в файле проекта [pyproject.toml](pyproject.toml): `mpinv-train` ([cli/train.py](src/mpinv/cli/train.py)), `mpinv-evaluate`, `mpinv-sweep`, `mpinv-generate-data`, `mpinv-validate-physics`, `mpinv-report`, `mpinv-data-stats`. Точки входа единообразно построены на базе Hydra [10] (версия 1.3.2): сценарий запуска задаётся композицией конфигов из директории [configs/](configs/) с возможностью переопределения отдельных параметров из командной строки. Файл `_builders.py` инкапсулирует общие сборщики (загрузчиков, признаков, моделей, потерь), `_configstore.py` — регистрацию типизированных конфигов в `ConfigStore` Hydra.

Внешние ключевые зависимости пакета: `torch` [9] (нейросетевая среда обучения), `torch-harmonics` [12] (преобразования сферических гармоник, используются в дифференцируемом декодере § 3.3), `scikit-learn` [8] (рандомизированный SVD § 2.4); вспомогательные библиотеки `numpy`, `scipy`, `matplotlib` входят в стандартный научный Python-стек и применяются для матрично-векторных операций, численных подпрограмм и визуализаций подпакета `analysis/plots/` соответственно. Конкретные диапазоны версий зависимостей фиксированы в файле проекта `pyproject.toml` воспроизводимого репозитория фреймворка.

## 7.2. Маппинг «секция работы — модуль фреймворка»

Таблица 7.1 сопоставляет каждую методически значимую секцию основного текста работы с модулем (или классом/функцией) фреймворка `mpinv`, в котором соответствующее решение реализовано. Маппинг служит навигационной картой: читателю достаточно открыть текст работы и таблицу, чтобы по любой секции восстановить точку в коде, а от точки в коде — обратиться к методическому обоснованию.

**Таблица 7.1 – Соответствие «секция работы — модуль `mpinv`»**

| Секция | Содержание | Модуль фреймворка |
|---|---|---|
| § 1.1 | Упаковка коэффициентов, порядок усечения | [src/mpinv/core/packing.py](src/mpinv/core/packing.py) |
| § 1.2 | Угловая сетка $360 \times 179$ | [src/mpinv/core/grid.py](src/mpinv/core/grid.py) |
| § 1.3 | Аналитический декомпозитор $\mathbf B$, весовая функция $\mu(\theta)$ | [src/mpinv/data/basis_decomposer.py](src/mpinv/data/basis_decomposer.py), [src/mpinv/core/area_weights.py](src/mpinv/core/area_weights.py) |
| § 2.1 | Синтетический генератор, четыре режима | [src/mpinv/data/synthetic_generator.py](src/mpinv/data/synthetic_generator.py) |
| § 2.1 | Примитивы аугментации | [src/mpinv/data/augment.py](src/mpinv/data/augment.py) |
| § 2.1 | Dummy-датасет (латинский квадрат) | [src/mpinv/data/dummy_probe.py](src/mpinv/data/dummy_probe.py) |
| § 2.2 | Реальные антенны, разбиение по источнику | [src/mpinv/data/real_antenna_loader.py](src/mpinv/data/real_antenna_loader.py), [src/mpinv/data/real_augmented_pipeline.py](src/mpinv/data/real_augmented_pipeline.py), [src/mpinv/data/splits.py](src/mpinv/data/splits.py) |
| § 2.3 | Входные режимы `power`/`magnitude`/`complex` | [src/mpinv/features/raw_flat.py](src/mpinv/features/raw_flat.py), [src/mpinv/features/power_pipeline.py](src/mpinv/features/power_pipeline.py) |
| § 2.4 | Рандомизированный SVD-PCA | [src/mpinv/features/pca.py](src/mpinv/features/pca.py) |
| § 2.5 | Радиальный спектр FFT, спектральная мощность SH, составные стеки | [src/mpinv/features/fft_radial.py](src/mpinv/features/fft_radial.py), [src/mpinv/features/sh_power.py](src/mpinv/features/sh_power.py), [src/mpinv/features/composite.py](src/mpinv/features/composite.py) |
| § 2.6 | Стандартизация признаков | [src/mpinv/features/normalisers.py](src/mpinv/features/normalisers.py) |
| § 3.1 | MLP-магистраль `mlp_5x200` | [src/mpinv/models/mlp.py](src/mpinv/models/mlp.py) |
| § 3.2 | Многоголовая модель и Матрёшка-trick | [src/mpinv/models/multi_head_mlp.py](src/mpinv/models/multi_head_mlp.py) |
| § 3.3 | Дифференцируемый VSH-декодер | [src/mpinv/losses/differentiable_field.py](src/mpinv/losses/differentiable_field.py) |
| § 4.1 | `CoefMSE` | [src/mpinv/losses/coef_mse.py](src/mpinv/losses/coef_mse.py) |
| § 4.2 | `PhysicsPowerLoss` и опциональный режим `log_ratio` | [src/mpinv/losses/physics_power.py](src/mpinv/losses/physics_power.py) |
| § 4.3 | `RankBinPLoss` | [src/mpinv/losses/rank_bin.py](src/mpinv/losses/rank_bin.py) |
| § 4.4 | Композитные потери `physics_power_mixed`, `physics_power_rank` | [src/mpinv/losses/physics_power.py](src/mpinv/losses/physics_power.py) |
| § 4.5 | Тренировочный цикл и колбэки | [src/mpinv/training/trainer.py](src/mpinv/training/trainer.py), [src/mpinv/callbacks/](src/mpinv/callbacks/) |
| § 4.6 | Пошаговый тренер для Матрёшки | [src/mpinv/training/staged.py](src/mpinv/training/staged.py) |
| § 4.7 | Сборка оптимизатора и шедулера | [src/mpinv/training/optim.py](src/mpinv/training/optim.py) |
| § 5.1.1 | Регрессионные метрики на поле | [src/mpinv/analysis/metrics/field_metrics.py](src/mpinv/analysis/metrics/field_metrics.py) |
| § 5.1.2 | Регрессионные метрики на коэффициентах | [src/mpinv/analysis/metrics/coefficient_metrics.py](src/mpinv/analysis/metrics/coefficient_metrics.py) |
| § 5.2 | Ранжирующие метрики на $P$ | [src/mpinv/analysis/metrics/field_metrics.py](src/mpinv/analysis/metrics/field_metrics.py) |
| § 5.3 | Метрика, инвариантная к отражённо-сопряжённой симметрии | [src/mpinv/analysis/metrics/mode_metrics.py](src/mpinv/analysis/metrics/mode_metrics.py) |
| § 5.5 | Наборы оценки и теги артефактов | [src/mpinv/data/splits.py](src/mpinv/data/splits.py), [src/mpinv/data/dummy_probe.py](src/mpinv/data/dummy_probe.py) |
| Гл. 6 | Точка входа обучения | [src/mpinv/cli/train.py](src/mpinv/cli/train.py) |
| Гл. 6 | Точки входа оценки, перебора, сводного отчёта | [src/mpinv/cli/evaluate.py](src/mpinv/cli/evaluate.py), [src/mpinv/cli/sweep.py](src/mpinv/cli/sweep.py), [src/mpinv/cli/report.py](src/mpinv/cli/report.py), [src/mpinv/analysis/reports/run_report.py](src/mpinv/analysis/reports/run_report.py) |
| Гл. 6 | Учёт экспериментов, фиксация состава выборок | [src/mpinv/tracking/](src/mpinv/tracking/) |

Расширенный маппинг с указанием конкретных классов и функций на каждый параграф приводится в Приложении Г и собирается автоматически по факту стабилизации фреймворка.

## 7.3. Архитектурные инварианты

В коде фреймворка фиксированы несколько архитектурных инвариантов, обеспечивающих внутреннюю согласованность реализации и единообразие именования между секциями работы и кодом.

**Угловая сетка — один источник истины.** Класс `GridSpec` и константа `GRID_DEFAULT = GridSpec(n_phi=360, n_theta=179, theta_start_deg=1.0, theta_end_deg=179.0)` объявлены в [src/mpinv/core/grid.py](src/mpinv/core/grid.py). Все модули фреймворка импортируют `GRID_DEFAULT` и производные величины ($n_\theta$, $n_\varphi$, $\mathrm{d}\theta$, $\mathrm{d}\varphi$, элемент площади $\mu(\theta)$) из неё, а не объявляют сетку повторно: соответствующий контракт зафиксирован в документационном комментарии модуля. Так исключается рассогласование между генератором данных § 2.1, признаковым пайплайном § 2.3, дифференцируемым декодером § 3.3 и метриками на поле § 5.1.1, которые все ссылаются на одну и ту же дискретизацию.

**Упаковка коэффициентов — один источник истины.** Константы `L_MAX = 15`, `K_MODES = L_MAX * (L_MAX + 2) = 255`, `PACKED_DIM = 4 * K_MODES = 1020` и функции `pack_coefficients`/`unpack_coefficients` объявлены в [src/mpinv/core/packing.py](src/mpinv/core/packing.py). Сама конвенция упаковки — `[\mathrm{Re}\,a^E, \mathrm{Im}\,a^E, \mathrm{Re}\,a^M, \mathrm{Im}\,a^M]` с обходом мод $l$ по возрастанию и $m$ от $-l$ до $+l$ — соответствует § 1.1; константы фреймворка поддерживают порядок усечения вплоть до $L = 15$ (что обеспечивает наращивание $L = 5 \to L = 15$ через Матрёшка-trick § 3.2), при этом в основной экспериментальной серии работы используется $L = 5$ с $K = 35$ и $4K = 140$. Биекция между упакованным вектором и сеткой коэффициентов, ожидаемой библиотекой `torch-harmonics`, реализована в той же [packing.py](src/mpinv/core/packing.py) (функции `pack_to_sht_grid`, `unpack_from_sht_grid`), что обеспечивает совместимость аналитической инверсии § 1.3 и градиентного синтеза § 3.3.

**Именованные реестры рабочих сущностей.** Каждое из трёх семейств рабочих сущностей — модели, потери, признаковые экстракторы — имеет собственный единый реестр: словарь `MODELS` в [src/mpinv/models/registry.py](src/mpinv/models/registry.py), словарь `LOSSES` в [src/mpinv/losses/registry.py](src/mpinv/losses/registry.py), словарь `FEATURE_EXTRACTORS` в [src/mpinv/features/registry.py](src/mpinv/features/registry.py). Регистрация выполняется декораторами `@register_model`, `@register_loss`, `@register_feature` с проверкой на повторное переопределение под тем же именем; рабочее имя — например, `mlp_5x200`, `physics_power_rank`, `raw_plus_sh` — служит ключом, по которому конфигурационная подсистема Hydra (§ 7.1) собирает запуск. Так исключается возможность того, чтобы два независимых модуля содержали одно и то же имя с расходящейся реализацией.

**Композиция конфигов через Hydra.** Точка входа `mpinv-train` ([src/mpinv/cli/train.py](src/mpinv/cli/train.py)) собирает конфигурацию запуска из девяти групп: `data`, `features`, `model`, `loss`, `optimiser`, `scheduler`, `trainer`, `callbacks`, `tracking`. Группы задаются в [configs/train.yaml](configs/train.yaml) через стандартный механизм `defaults` Hydra и допускают переопределение любой ветви из командной строки. Каждое имя в конфиге соответствует ключу в одном из реестров; благодаря этому переход от написанного в работе методического решения к воспроизводимому запуску эксперимента сводится к выбору соответствующих рабочих имён в командной строке.

**Учёт состава выборок.** Состав каждой выборки (обучающей, валидационной, отложенной, dummy) фиксируется через `DatasetSpec` и записывается в MLflow вспомогательной функцией `log_numpy_dataset` из [src/mpinv/tracking/dataset_logger.py](src/mpinv/tracking/dataset_logger.py). Это обеспечивает возможность точной репликации экспериментов Главы 6 по запущенному ранее identifier-у — без полного сохранения исходных массивов в репозитории.

Совокупность перечисленных инвариантов фиксирует именно ту минимальную структуру, при которой описание работы в текстовой форме (Главы 1–6) и описание в коде согласованы по обозначениям, по дискретизации, по упаковке и по рабочим именам; за её пределами фреймворк остаётся открытым к доработке в последующих итерациях.
