# Prédiction du Churn Client – Projet de Fouille de Données

Salut ! 👋  
Ce projet a pour objectif de comprendre pourquoi certains clients quittent une entreprise (le fameux **churn**) et de prédire quels clients sont les plus à risque. L’idée est de fournir des insights utiles pour améliorer la rétention client.

---

## 📖 À propos du projet
Nous utilisons des **données clients** comprenant des informations démographiques et comportementales pour :  
- Identifier les facteurs qui influencent le départ des clients  
- Construire un modèle de prédiction efficace  
- Fournir des recommandations pour réduire le churn

---

## 🗂 Structure du projet

/churn-prediction-ml
│
├── codeSource.py # Script Python pour analyser les données et créer les modèles
├── codeSource.ipynb # Notebook Jupyter interactif avec visualisations et explications
├── rapport.pdf # Rapport détaillant les résultats et conclusions
├── churn_dataset.csv # Jeu de données des clients


---

## 🎯 Objectifs du projet
- Comprendre les comportements menant au churn  
- Développer un modèle prédictif robuste  
- Identifier des actions concrètes pour retenir les clients

---

## 📊 Résultats
- **Meilleur modèle** : XGBoost  
- **Accuracy** : 81%  
- **Précision** : 84%  
- **Rappel** : 78%  
- **F1-Score** : 81%  

> Les résultats montrent que notre modèle peut prédire avec assez de précision les clients susceptibles de partir, ce qui permet de cibler les actions de rétention.

---

## 🛠️ Technologies et librairies
- Python 3.8+  
- Pandas & NumPy pour la manipulation des données  
- Scikit-learn & XGBoost pour le Machine Learning  
- Matplotlib & Seaborn pour les visualisations  
- Imbalanced-learn pour gérer les classes déséquilibrées

---

## 🚀 Comment utiliser ce projet

1. **Cloner le dépôt**
```bash
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
