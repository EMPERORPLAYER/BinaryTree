
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBRegressor


# =====================================================================
# Завдання 1: Дерево рішень (регресія)
# =====================================================================
print("=" * 60)
print("Завдання 1: DecisionTreeRegressor")
print("=" * 60)

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = np.array(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
r2_tree = r2_score(y_test, tree.predict(X_test))
print(f"DecisionTreeRegressor  Тест R²: {r2_tree:.4f}")

# Важливість ознак (горизонтальна діаграма)
importances = tree.feature_importances_
order = np.argsort(importances)

plt.figure(figsize=(8, 5))
plt.barh(feature_names[order], importances[order])
plt.xlabel("Важливість")
plt.title("Feature importances (DecisionTreeRegressor)")
plt.tight_layout()
plt.show()


# =====================================================================
# Завдання 2: Дерево рішень vs Випадковий ліс
# =====================================================================
print("\n" + "=" * 60)
print("Завдання 2: DecisionTree vs RandomForest")
print("=" * 60)

forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
r2_forest = r2_score(y_test, forest.predict(X_test))

table_2 = pd.DataFrame({
    "Модель": ["DecisionTreeRegressor", "RandomForestRegressor"],
    "Тест R²": [round(r2_tree, 4), round(r2_forest, 4)],
})
print(table_2.to_string(index=False))


# =====================================================================
# Завдання 3: Градієнтний бустинг (XGBoost)
# =====================================================================
print("\n" + "=" * 60)
print("Завдання 3: XGBRegressor vs RandomForest")
print("=" * 60)

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
)
xgb.fit(X_train, y_train)
r2_xgb = r2_score(y_test, xgb.predict(X_test))

table_3 = pd.DataFrame({
    "Модель": ["RandomForestRegressor", "XGBRegressor"],
    "Тест R²": [round(r2_forest, 4), round(r2_xgb, 4)],
})
print(table_3.to_string(index=False))

best = "XGBRegressor" if r2_xgb > r2_forest else "RandomForestRegressor"
print(f"\nВисновок: на цих даних краще працює {best}.")


# =====================================================================
# Завдання 4: Логістична регресія (Breast Cancer)
# =====================================================================
print("\n" + "=" * 60)
print("Завдання 4: LogisticRegression")
print("=" * 60)

cancer = load_breast_cancer()
Xc, yc = cancer.data, cancer.target

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xc_train, yc_train)
y_pred = clf.predict(Xc_test)

print(f"Accuracy:  {accuracy_score(yc_test, y_pred):.4f}")
print(f"Precision: {precision_score(yc_test, y_pred):.4f}")
print(f"Recall:    {recall_score(yc_test, y_pred):.4f}")
print(f"F1:        {f1_score(yc_test, y_pred):.4f}")

cm = confusion_matrix(yc_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=cancer.target_names).plot()
plt.title("Confusion matrix (LogisticRegression)")
plt.show()