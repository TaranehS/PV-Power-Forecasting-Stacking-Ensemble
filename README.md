# 🌞 Photovoltaic Power Forecasting using Deep Stacking Ensemble (LSTM–GRU)

## 📌 Overview

Accurate short-term forecasting of photovoltaic (PV) power generation is essential for efficient grid integration, energy trading, and renewable energy management. However, the nonlinear and highly dynamic nature of solar energy production makes this task challenging.

This repository presents a comprehensive machine learning and deep learning framework for PV power forecasting, culminating in a **novel deep stacking ensemble model** that integrates **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** architectures.

The proposed model achieves state-of-the-art performance and demonstrates the effectiveness of combining complementary deep learning models within a streamlined ensemble structure.

---

## 🚀 Key Contributions

* Development of a **deep stacking ensemble meta-model (LSTM + GRU)**
* Systematic comparison of:

  * Machine Learning models (FFNN, ELM)
  * Deep Learning models (LSTM, GRU)
  * Ensemble Learning models (Random Forest, XGBoost)
* Implementation of **Bayesian hyperparameter optimization (TPE)**
* Integration of **Model Output Statistics (MOS)** for bias correction
* Demonstration that **fewer, high-quality base models outperform large ensembles**

---

## 📊 Performance Highlights

The proposed stacking model achieves:

* **R² = 0.9805**
* **MAPE = 0.1146**
* **RMSE = 6.32**

This outperforms:

* All individual models
* Conventional stacking approaches with more base learners

---

## 🧠 Methodology

The workflow includes:

1. Data preprocessing and feature engineering
2. Training of multiple base models:

   * FFNN, ELM
   * LSTM, GRU
   * Random Forest, XGBoost
3. Construction of stacking ensemble models:

   * 7-base model stacking
   * 5-base model stacking
   * ✅ Proposed 2-base model stacking (LSTM + GRU)
4. Hyperparameter optimization using **Tree-structured Parzen Estimator (TPE)**
5. Post-processing using **Model Output Statistics (MOS)**

---

## 🗂️ Repository Structure

```
models/
  ├── machine_learning/
  ├── deep_learning/
  └── ensemble_learning/

stacking/
  ├── stacking_7_models/
  ├── stacking_5_models/
  └── proposed_stacking_LSTM_GRU/

utils/
data/
results/
```

---

## 📊 Data and Full Research Materials

The complete dataset and all supporting materials are publicly available at:

**Saadati, T., & Barutcu, B. (2025)**
*Solar Energy Production Time Series Data*
Mendeley Data, V3
DOI: https://doi.org/10.17632/2tpv28kr83.3

The Mendeley repository includes:

* Raw Data
* Final Dataset used in the study
* Exploratory Data Analysis
* Evaluation Metrics Comparison
* Chaotic Analysis (False Nearest Neighbors)
* Chaotic Model (Echo State Network - ESN)
* Machine Learning Models (FFNN, ELM)
* Deep Learning Models (LSTM, GRU)
* Ensemble Learning Models (RF, XGBoost)
* Stacking Ensemble Models

This GitHub repository provides a structured and reusable implementation of the **core forecasting models**, focusing on reproducibility and usability.

---

## 📄 Related Publications

This repository supports the following publications:

1. Saadati, T., & Barutcu, B. (2025a)
   *Forecasting Solar Energy: Leveraging Artificial Intelligence and Machine Learning for Sustainable Energy Solutions*
   Journal of Economic Surveys

2. Saadati, T., & Barutcu, B. (2025b)
   *Optimized Solar Energy Forecasting for Sustainable Development Using Machine Learning, Deep Learning, and Chaotic Models*
   International Journal of Energy Economics and Policy, 15(1), 110–120

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/TaranehS/PV-Power-Forecasting-Stacking-Ensemble.git
cd PV-Power-Forecasting-Stacking-Ensemble
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Download the dataset from Mendeley Data (link above)
2. Place data in the `/data` folder
3. Run base models from `/models/`
4. Run stacking models from `/stacking/`

---

## 🎯 Key Insight

A major finding of this research is that:

> A **carefully selected small set of complementary models** (LSTM + GRU) can outperform larger and more complex ensembles by reducing noise, redundancy, and overfitting.

---

## 🔮 Future Work

* Model interpretability (Explainable AI)
* Longer forecasting horizons
* Hierarchical forecasting frameworks

---

## 🙏 Acknowledgment

This work is part of a PhD research on solar energy forecasting and contributes to advancing AI-driven solutions for sustainable energy systems.
This research was supported by the Scientific Research Projects (BAP) Coordination Unit of Istanbul Technical University  
(Project ID: 44133, Project Code: MDK-2022-44133).
