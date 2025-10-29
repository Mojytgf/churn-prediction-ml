# 📊 Prédiction du Churn Client - Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Classification-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![XGBoost](https://img.shields.io/badge/XGBoost-81%25%20Accuracy-red)

## 📖 À propos du Projet

Ce projet vise à prédire le **churn client** (départ des clients) en utilisant des algorithmes de Machine Learning. L'objectif est d'identifier les clients susceptibles de quitter une entreprise pour permettre la mise en place de stratégies de rétention proactives.

## 📊 Résultats Clés

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 81% |
| **Précision** | 84% |
| **Rappel** | 78% |
| **F1-Score** | 81% |

**Meilleur modèle :** 🏆 XGBoost

## 🗂 Structure du projet

```text
churn-prediction-ml/
├── 📄 codeSource.py         # Script principal Python
├── 📓 codeSource.ipynb      # Notebook Jupyter interactif
├── 📋 rapport.pdf           # Rapport détaillé du projet
├── 📊 churn_dataset.csv     # Jeu de données clients
├── 📝 README.md             # Ce fichier
└── ⚙️ requirements.txt      # Dépendances Python


🎯 Objectifs du projet

Comprendre les comportements menant au churn

Développer un modèle prédictif robuste

Identifier des actions concrètes pour retenir les clients

📊 Résultats

Meilleur modèle : XGBoost

Accuracy : 81%

Précision : 84%

Rappel : 78%

F1-Score : 81%

Les résultats montrent que notre modèle peut prédire avec assez de précision les clients susceptibles de partir, ce qui permet de cibler les actions de rétention.

🛠️ Technologies et librairies

Python 3.8+

Pandas & NumPy pour la manipulation des données

Scikit-learn & XGBoost pour le Machine Learning

Matplotlib & Seaborn pour les visualisations

Imbalanced-learn pour gérer les classes déséquilibrées

🚀 Comment utiliser ce projet

Cloner le dépôt

git clone https://github.com/Mojytgf/churn-prediction-ml.git
cd churn-prediction-ml

    Installer les dépendances

pip install -r requirements.txt

    Exécuter le script Python

python codeSource.py

    Ou explorer le Notebook Jupyter

jupyter notebook codeSource.ipynb

    Le Notebook est parfait pour suivre l’analyse étape par étape et voir les visualisations.

💡 Remarques

    Gardez le fichier churn_dataset.csv dans le même dossier que les scripts pour que tout fonctionne correctement.

    Ce projet est idéal pour se familiariser avec la fouille de données, le Machine Learning et l’analyse prédictive sur un cas réel.

Merci d’avoir consulté ce projet ! 🎉
N’hésitez pas à cloner le dépôt et à expérimenter avec vos propres idées pour améliorer la prédiction du churn.
