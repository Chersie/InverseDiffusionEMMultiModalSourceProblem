# Pre-defense Q&A — likely questions and draft answers

Use this sheet as a reference, not a script. Listen to the full question, paraphrase it back to the audience, then answer in 30–60 seconds. If you do not know, say so honestly: «Хороший вопрос — я это пока не исследовал; вот как бы я подошёл …».

Material backing every answer:
- `presentation/ch1_full.md` (theoretical sections §1.1–§1.8)
- `experiments/baseline/baseline_experiments_report.md` (S0–S3)
- `experiments/baseline/S1_table.md`
- `experiments/baseline/S5_real_augmented_results_best.json` (the headline run)
- `defense.pdf` (project-level overview)

---

## Q1. «Почему MLP, а не CNN на сфере / Transformer / SE(3)-equivariant network?»

**Black-box claim**: the simplest baseline that respects the data shape is the right first answer to an ill-posed problem; architecture is, empirically, not the dominant lever.

**Draft answer (≈ 45 s)**

«Это сознательный выбор baseline. Во-первых, в S3 мы показали, что главный рычаг — это распределение обучающих данных, а не архитектура: 115-кратное улучшение при одной только смене режима генератора, при фиксированных модели, лоссе и аугментациях. Когда сама задача недоопределена, сложная архитектура не починит проблему. Во-вторых, MLP-baseline — необходимая отправная точка по дисциплине Arridge et al. 2019 §5.1.3 «fully learned»; без него мы не сможем оценить, что именно даёт CNN/Transformer на сфере. План на недели 6–7 — добавить literature-comparison и опционально сферическую CNN, после того как мы закроем главу 7.»

---

## Q2. «Почему `holdout/field_NRMSE = 3.07`, а `val/field_NRMSE = 0.31`? Это десятикратный gap.»

**Black-box claim**: distribution shift inside the real-antenna pool. The 200 источников, на которых мы обучаемся, и 196 источников, на которых валидируемся, геометрически отличаются. Аугментация (phi_roll, field_additive_noise, mode_dropout) обогащает выборку внутри источника, но не учит модель обобщаться между источниками.

**Draft answer (≈ 50 s)**

«Этот gap — главная честная проблема работы, и я её на 12-м слайде специально вынесла. val_real — это другие аугментированные семплы тех же 200 источников; модель видела их геометрию. holdout_real — 196 источников, которые модель не видела ни в каком виде. Аугментации, которые я применяю, физически согласованы — например, field_phi_roll имеет регрессионный тест на consistency с поворотом коэффициентов, — но они меняют поле в пределах одной геометрии радиатора. Они не моделируют переход между разными антеннами. Поэтому решение в плане — это leave-one-source-out протокол, плюс совместное обучение с синтетикой, чтобы покрыть пространство геометрий шире.»

---

## Q3. «Что вы делаете с тривиальной неоднозначностью §1.7 в обучении и валидации?»

**Black-box claim**: in evaluation — we use `coef_mse_amb_aware = min(MSE(pred, target), MSE(pred, target_reflected))`. In training — currently nothing explicit; on the §1.7 (i) global-phase orbit, augmentation `coef_phase_rotation` rotates the *target*, telling the network «any global phase is equally good», but does not symmetrise on (ii) reflected-conjugate. That is a planned addition.

**Draft answer (≈ 50 s)**

«В метрике — да, есть `coef_mse_amb_aware`, который берёт минимум MSE между предсказанием и таргетом, и между предсказанием и его reflected-conjugate. В обучении — на (i), глобальную фазу U(1), мы используем аугментацию `coef_phase_rotation`, которая поворачивает таргет на случайный угол: сетка таким образом видит, что любой глобальный фазовый сдвиг одинаково валиден. На (ii), reflected-conjugate — пока ничего явного. В S1 мы проверили: gap между обычным `coef_mse` и `coef_mse_amb_aware` < 0.001, то есть модель сейчас одинаково далека от обоих таргетов, ни одного не «выучивает». Явная симметризация лосса по reflected-conjugate — конкретный пункт плана на недели 4–5.»

---

## Q4. «Где аналитический baseline? Почему вы сравниваетесь только с линейной регрессией?»

**Black-box claim**: the analytic VSH decomposition of §1.3 requires the **complex** field; phaseless data does not admit an analytic inverse. Linear regression is the strongest classical baseline that takes the same input the MLP takes — that is why it is the benchmark.

**Draft answer (≈ 45 s)**

«Аналитического baseline для нашей задачи не существует. Аналитическая разложение из 1.3 требует *комплексного* поля E(θ,φ), а у нас на входе скалярная мощность P — фаза уничтожена. Это не лимит метода, а физический факт, который мы доказываем в 1.4 и 1.5. Самый сильный классический baseline, который работает от того же входа P — это линейная регрессия. В S1 на gaussian-режиме линейная регрессия с raw_flat признаками даёт val/coef_mse = 11.7 — катастрофа из-за 64 440 входных фичей на 4096 обучающих сэмплах. На power_pca-фичах — 0.51, что уже сопоставимо с MLP. То есть наш MLP на main-эксперименте (val/field_NRMSE = 0.31) бьёт линейный baseline (≈ 1.0) более чем в 3 раза, а на raw_flat — в 80 раз.»

---

## Q5. «Что если комитет на защите попросит результат на L = 15, а не L = 5?»

**Black-box claim**: the infrastructure runs L = 15 unchanged. The baseline experiments report runs at L = 5 for budget; the architectural invariants (`(B, 179, 360)` grid, `4K = 1020` packed coefficient layout) are exactly the project default. Estimated turnaround ≈ 6 hours per L per cell.

**Draft answer (≈ 40 s)**

«Инфраструктура спроектирована под L = 15 — это и есть проектный default. В baseline-эксперименте мы намеренно работаем на L = 5, чтобы прогнать матрицу 60×3 за день и понять, какие оси действительно двигают метрику. Все физические инварианты — сетка (179, 360), упаковка коэффициентов длины 4K = 1020 — фиксированы один раз в `core/packing.py` и работают одинаково на L = 5 и L = 15. План: бандлимит-аблейшн на неделе 3, ожидаемый турнэраунд — 6 часов на L на cell. Если комитет хочет L = 15 — это конкретный run, который я могу показать к финальной защите.»

---

## Bonus questions to keep ready (≤ 20 s each)

| #  | Вопрос                                                       | Ответ-«hook»                                                              |
|----|--------------------------------------------------------------|---------------------------------------------------------------------------|
| B1 | Почему именно `physics_power_rank`, а не `physics_power`?    | Rank-based лосс устойчив к outlier-пикам в P; см. §S1 ranking ablation. |
| B2 | Что значит `field_r2_w = −6×10⁸` на synthetic_test?          | Не баг; scale-factor 1e6 не транслируется на синтетическую тестовую распред. |
| B3 | Сколько параметров в финальной модели?                       | MLP 5×200 на raw_plus_sh ≈ 64 445→200→…→200→1020 ≈ 13 М параметров.   |
| B4 | Почему именно `raw_plus_sh`, а не `power_pca`?               | На S1 на gaussian power_pca уступает; на реальных данных — see Wk 1–2 HPO. |
| B5 | Использовали ли вы Lightning?                                | Нет — custom Trainer per AGENTS.md; rationale в practice.pdf §3.        |
| B6 | Где код экспериментов?                                       | `scripts/run_real_augmented.py`, `scripts/run_baseline_*.py`; результаты в `experiments/baseline/`. |
| B7 | Что такое `coef_mse_amb_aware`?                              | `min(MSE(pred, target), MSE(pred, reflected_target))`; адресует §1.7 (ii).|
| B8 | Сколько времени уходит на одну тренировку?                   | 90–120 секунд на L = 5 / 10k augmented / mlp_5x200 / 300 эпох, MPS.    |
| B9 | На каком железе обучали?                                     | Apple Silicon M-series, PyTorch MPS backend; CUDA-совместимо без правок. |
| B10| Что не вошло в работу?                                       | Сферические CNN, plug-and-play, диффузионные prior, classification §1.7(iv).|
