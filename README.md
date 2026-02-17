# ESG-Classification-Challenge 🌍⚖️

# 🏆 2nd Place Solution - Go Data Science 5.0

This project details the solution that secured **2nd Place** in the **Go Data Science 5.0** competition. We developed a robust Deep Learning pipeline to classify corporate text into **Environmental**, **Social**, and **Governance** (ESG) categories.

The work was carried out collaboratively by **[Dalia Riahi]**, **[Asser Saadaoui]**, and **[Firas Jebeniani]**.

## 📚 Project Overview

In this notebook, we analyze unstructured corporate text data and build a sophisticated multi-label classification pipeline using advanced NLP techniques. The workflow includes:

* **Data Integrity Audit:** Regex-based noise profiling and removal of conflicting labels.
* **Preprocessing:** Text normalization and tokenization using Hugging Face tools.
* **Handling Imbalance:** Custom `WeightedTrainer` to address rare classes.
* **Ensemble Learning:** 5-Fold Cross-Validation with DistilBERT.
* **Stacking:** A Logistic Regression Meta-Learner to optimize final predictions.

## 🛠️ Technologies Used

* Python
* Jupyter Notebook
* PyTorch & Transformers (Hugging Face)
* pandas, numpy
* scikit-learn
* matplotlib, seaborn

## 📊 Models Used

* **Base Model:** DistilBERT (`distilbert-base-uncased`)
* **Meta-Learner:** Logistic Regression (for Stacking)
* **Architecture:** Stacking Ensemble with 5-Fold CV

## 🧪 Evaluation Metrics

We used standard multi-label classification metrics such as:

* F1-Score (Macro & Weighted)
* Accuracy
* Calibration Score (for the Meta-Learner)

## 📁 Project Structure

```text
📦 ESG Classification
 ┣ 📓 esg-notebook.ipynb      # Main Jupyter notebook with training pipeline
 ┣ 📄 submission.csv          # Final ensemble predictions
 ┣ 📂 data/                   # Dataset folder (train.csv, test.csv)
 ┗ 📄 README.md

```

## 🤝 Team Members

* [Firas Jebeniani]
* [Asser Saadaoui]
* [Dalia Riahi]

## 📌 Notes

* This project addresses a **Multi-Label Classification** problem where a single text can belong to multiple ESG pillars.
* We implemented a **Stacking Strategy** to combine predictions from 5 different model splits, significantly reducing variance and improving generalization on the test set.
