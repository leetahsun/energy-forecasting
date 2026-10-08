# energy-forecasting
# Energy Forecasting for the German Electricity Market

Machine-learning and physics-based forecasting experiments for German electricity-market and renewable-energy data.

This project explores how **data-driven models** and **physical solar models** can be used together for short-term energy forecasting.

The repository is a follow-up to my [`electricitymarketreport`](https://github.com/leetahsun/electricitymarketreport) project, which focuses on automated collection and reporting of German electricity-market KPIs.

The goal here is different:

> move from describing the electricity system to forecasting its behaviour.

---

## Project goals

This repository is designed to explore four questions:

1. How strong are simple time-series baselines for German energy forecasting?
2. How much improvement can classical machine-learning models provide?
3. Can physical knowledge improve purely data-driven predictions?
4. How should energy forecasting models be evaluated without introducing temporal leakage?

The project currently contains two complementary forecasting approaches:

- **Machine-learning forecasting** using historical energy and temporal features
- **Physics-based solar forecasting** using physical relationships relevant to solar generation

The long-term goal is to combine both approaches into a **hybrid physics + machine-learning forecasting system**.

---

## Why this project?

Renewable-energy forecasting is difficult because electricity generation depends on both:

- predictable temporal structure
- changing physical conditions

Solar generation, for example, depends strongly on:

- time of day
- season
- solar geometry
- irradiance
- cloud cover
- atmospheric conditions

Pure machine-learning models can learn historical relationships, while physical models encode known behaviour of the underlying system.

This project investigates how both approaches can complement each other.

---

## Forecasting workflow

```text
German electricity-market data
            +
     weather / solar data
            ↓
      data validation
            ↓
     feature engineering
            ↓
  ┌─────────┴─────────┐
  │                   │
baseline models   physical model
  │                   │
  │               solar estimate
  │                   │
  └─────────┬─────────┘
            ↓
        XGBoost
            ↓
     model evaluation
            ↓
 forecasts + diagnostics
```

---

## Machine-learning forecasting

The machine-learning component uses structured historical data to create forecasting features such as:

### Temporal features

- hour of day
- day of week
- month
- weekend / weekday information

### Lag features

Examples include:

- previous hour
- previous day
- same hour on the previous day
- same hour during the previous week

### Rolling statistics

Examples include:

- rolling mean
- rolling standard deviation
- recent generation trends

These features are used with classical machine-learning models such as:

- linear / simple statistical baselines
- scikit-learn models
- XGBoost

---

## Baselines

Forecasting models should not only be compared with each other.

They should first beat simple forecasting rules.

The project therefore uses seasonal persistence baselines such as:

```text
prediction(t) = observation(t - 24 hours)
```

and:

```text
prediction(t) = observation(t - 168 hours)
```

These correspond approximately to:

- same hour yesterday
- same hour last week

A machine-learning model is useful only if it can consistently improve on these simple alternatives.

---

## Physics-based solar forecasting

The repository also contains a separate solar-forecasting component.

Rather than learning every relationship entirely from data, this approach incorporates physical information related to solar-energy production.

The broader idea is:

```text
solar geometry
      +
physical solar model
      ↓
expected generation
```

This estimate can then be used directly or supplied as an input to a machine-learning model.

---

## Hybrid physics + machine learning

A major direction of this project is **residual learning**.

Instead of asking a machine-learning model to predict solar generation from scratch, a physical model can first produce an estimate:

\[
\hat{y}_{physics}
\]

The residual is then:

\[
r = y_{actual} - \hat{y}_{physics}
\]

A machine-learning model learns to predict this residual:

\[
\hat{r}_{ML}
\]

The final forecast becomes:

\[
\hat{y}_{final}
=
\hat{y}_{physics}
+
\hat{r}_{ML}
\]

This creates a hybrid model that combines:

- known physical behaviour
- patterns learned from data

One of the main objectives of this repository is to determine whether this approach improves robustness compared with either method individually.

---

## Time-series evaluation

Time-series forecasting requires special care during evaluation.

A random train/test split can leak information from the future into the training data and produce overly optimistic results.

This project therefore uses **chronological evaluation**.

Conceptually:

```text
past                         future

|-------- training --------|--- validation ---|--- test ---|
```

Future development will also include rolling / walk-forward evaluation:

```text
train ───────────> test

train ─────────────────> test

train ───────────────────────> test
```

This better represents how forecasting systems are used in practice.

---

## Evaluation metrics

Models are evaluated using standard regression metrics including:

### Mean Absolute Error

\[
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
\]

### Root Mean Squared Error

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

MAE provides an intuitive measure of typical forecast error, while RMSE penalizes large errors more strongly.

---

## Results

Quantitative benchmark results are currently being consolidated into a reproducible evaluation pipeline.

The final comparison will follow this structure:

| Model | MAE | RMSE | Improvement vs baseline |
|---|---:|---:|---:|
| 24-hour persistence | TBD | TBD | — |
| 168-hour seasonal persistence | TBD | TBD | TBD |
| Classical ML baseline | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| Physics-based model | TBD | TBD | TBD |
| Hybrid physics + ML | TBD | TBD | TBD |

Results will only be reported after evaluation on a strictly chronological test period.

---

## Repository structure

```text
energy-forecasting/
│
├── ml_forecasting/
│   └── machine-learning forecasting pipeline
│
├── solar_forecasting/
│   └── physics-based solar forecasting
│
├── shared/
│   └── functionality shared across forecasting approaches
│
├── tests/
│   └── automated tests
│
├── .github/
│   └── CI workflows
│
├── pyproject.toml
└── README.md
```

---

## Tech stack

The project currently uses:

- Python 3.12
- pandas
- NumPy
- scikit-learn
- XGBoost
- Plotly
- PyArrow
- Pydantic
- Joblib
- Pytest
- GitHub Actions

---

## Installation

Clone the repository:

```bash
git clone https://github.com/leetahsun/energy-forecasting.git
cd energy-forecasting
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install the package:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Testing

Run the test suite with:

```bash
pytest
```

Tests are intended to cover important parts of the forecasting pipeline, including data processing and model-related functionality.

---

## Current development priorities

The next improvements are focused on making the forecasting comparison more rigorous and reproducible:

- [ ] Define a single primary forecasting target and forecast horizon
- [ ] Add 24-hour persistence baseline
- [ ] Add 168-hour seasonal baseline
- [ ] Implement chronological train / validation / test split
- [ ] Add rolling backtesting
- [ ] Benchmark XGBoost against simple baselines
- [ ] Add MAE and RMSE reporting
- [ ] Add forecast-vs-actual visualizations
- [ ] Add feature-importance / SHAP analysis
- [ ] Combine physical solar forecasts with ML residual correction
- [ ] Document final benchmark results


## Related project

### Electricity Market Report

The forecasting work builds on:

[`electricitymarketreport`](https://github.com/leetahsun/electricitymarketreport)

That project focuses on automated German electricity-market data collection, KPI analysis, and reporting.

This repository extends the workflow from:

```text
What happened?
```

toward:

```text
What happens next?
```

---

## License

This repository is intended as an educational and research portfolio project.

A formal open-source license can be added as the project matures.
