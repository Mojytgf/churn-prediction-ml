import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import seaborn as sns

# ===============================
# Étape 1 : Compréhension de la problématique
# ===============================
# Objectif : Prédire si un client quittera la banque (variable cible 'Exited').

# ===============================
# Étape 2 : Exploratory Data Analysis (EDA)
# ===============================
# Charger les données
df = pd.read_csv('churn_dataset.csv')

# Afficher le nombre de lignes et de colonnes
print("Nombre de lignes et colonnes :", df.shape)

# Définir les colonnes pertinentes
cont_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']  # Variables continues
cat_columns = ['Geography', 'Gender']  # Variables catégoriques
binary_columns = ['HasCrCard', 'IsActiveMember', 'NumOfProducts']  # Variables binaires

# Définir la variable cible
target = 'Exited'

# Distribution de la variable cible
print("\nDistribution de la variable cible :")
print(df[target].value_counts())

# Distribution des variables continues
for var in cont_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[var], kde=True, bins=30)
    plt.title(f"Distribution de {var}")
    plt.show()

# Relation entre les variables continues et la cible
for var in cont_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target, y=var, data=df)
    plt.title(f"{var} en fonction de {target}")
    plt.show()

# Matrice de corrélation
plt.figure(figsize=(10, 6))
sns.heatmap(df[cont_columns + [target]].corr(), annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

# Distribution des variables catégoriques
for var in cat_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=var, data=df)
    plt.title(f"Distribution de {var}")
    plt.show()

# Relation entre les variables catégoriques et la cible
for var in cat_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=var, hue=target, data=df)
    plt.title(f"Relation entre {var} et {target}")
    plt.show()

# Distribution des variables binaires
for var in binary_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=var, data=df)
    plt.title(f"Distribution de {var}")
    plt.show()

# Relation entre les variables binaires et la cible
for var in binary_columns:
    cross_tab = pd.crosstab(df[var], df[target], normalize='index')
    print(f"\nRelation entre {var} et {target} :\n{cross_tab}")

# Vérification des valeurs manquantes
print("\nValeurs manquantes :")
print(df.isnull().sum())

# ===============================
# Étape 3 : Split des données
# ===============================
X = df[cat_columns + cont_columns + binary_columns]  # Variables explicatives
y = df[target]  # Variable cible

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\nTaille des ensembles :")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# ===============================
# Étape 4 : Data Preprocessing
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_columns),
        ('binary', 'passthrough', binary_columns),
        ('num', StandardScaler(), cont_columns)
    ]
)

# Configurer SMOTE et le pipeline
smote = SMOTE(random_state=42)
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=200, random_state=42))
])

# ===============================
# Étape 5 : Model Training
# ===============================
model_pipeline.fit(X_train, y_train)

# ===============================
# Étape 6 : Model Testing
# ===============================
# Prédictions
y_pred = model_pipeline.predict(X_test)

# ===============================
# Étape 7 : Analyse des résultats
# ===============================
# Évaluation des performances
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))

# Calcul de l'AUC-ROC
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]  # Probabilité pour la classe Exited=1
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC: {auc}")

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ===============================
# Étape 8 : Optimisation des hyperparamètres
# ===============================
param_grid_xgb = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 6],
    'classifier__learning_rate': [0.01, 0.1]
}

grid_search_xgb = GridSearchCV(model_pipeline, param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)

# Meilleurs paramètres pour XGBoost
print("\nMeilleurs paramètres pour XGBoost :", grid_search_xgb.best_params_)
print("Meilleure précision pour XGBoost :", grid_search_xgb.best_score_)

# ===============================
# Étape 9 : Comparaison avec d'autres modèles
# ===============================
# Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
logreg_predictions = logreg.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, logreg_predictions))
print("\nRapport de classification pour Logistic Regression :\n", classification_report(y_test, logreg_predictions))

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
dt_predictions = decision_tree.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_predictions))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_predictions))

# Comparer les performances des modèles
print("\nRapport de classification pour Random Forest :\n", classification_report(y_test, rf_predictions))
