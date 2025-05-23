# Network-Intrusion-Detection-Project

Using the UNSW-NB15 dataset, a NIDS can be developed and evaluated by training classifier models on it to predict future attacks. However, the development of an effective NIDS is challenging as the network traffic dataset is noisy, imbalanced, and has complex relations, especially for predicting the attack category.

# Network Intrusion Detection System (NIDS) - Project

## Overview

This project explores the implementation and comparison of multiple classification algorithms to develop a Network Intrusion Detection System (NIDS) using the UNSW-NB15 dataset. It focuses on predicting whether a network record is an attack, and identifying the attack category.

## Models Implemented

- **K-Nearest Neighbors (KNN):** Non-parametric model using distance metrics and neighbor voting.
- **Random Forest:** Ensemble model using multiple decision trees with majority voting.
- **Multi-Layer Perceptron (MLP):** Feedforward neural network using backpropagation.

## Feature Selection Techniques

- **Principal Component Analysis (PCA):** Unsupervised dimensionality reduction capturing maximum variance.
- **Recursive Feature Elimination (RFE):** Wrapper-based feature selection using model feature importances.
- **Correlation-based Selection (SelectKBest + ANOVA F-test):** Univariate filtering to choose top correlated features.

## Dataset

- **Source:** [UNSW-NB15 Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- Simulated network traffic data labeled with normal/attack and attack categories.
- **Split:** Train (70%), Validation (15%), Test (15%)

## Evaluation Metrics

- **Accuracy**
- **Micro-F1 Score** (for label classification)
- **Macro-F1 Score** (for attack category classification)

## How to Run

### Prerequisites

Install required packages:

```bash
pip install argparse scikit-learn pandas numpy matplotlib
```

### Running the script

```bash
python NIDS.py -t <testset.csv> -c <classifier> -t <task> [-m <model.pkl>]
```

- `-c`: Classifier (`knn`, `rf`, `mlp`)
- `-t`: Task (`label`, `attackcat`)
- `-m`: Optional pre-trained model file

## Results Summary

- **Best Label Classifier:** KNN with 99.3% accuracy after feature selection.
- **Best Attack Category Classifier:** Random Forest with Macro-F1 ~0.69 and Micro-F1 ~0.95.
- **Feature selection significantly improved KNN’s performance** by reducing dimensionality.

## Contributions

- **Kris Chan:** RFE, KNN classifier, test/train setup
- **Angus Lin:** PCA, Random Forest, analysis tools
- **Aadi Bisht:** Correlation analysis, MLP classifier, hyperparameter tuning

All members contributed to shared infrastructure and reporting.

## License

MIT License
