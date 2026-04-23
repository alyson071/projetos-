import pandas as pd
import sklearn.model_selection as ms
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Carregamento (
dataset = pd.read_csv("archive/doencas_cardiacas.csv")
dataset = dataset.dropna()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2. Divisão PRIMEIRO
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Normalização DEPOIS (Essencial para evitar Data Leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # Apenas transform aqui!

# 4. Treinando o modelo
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 5. Valor Específico (Agora com o scaler aplicado)
exemplo = np.array([[1, 30, 4, 0, 0, 0, 0, 0, 0, 195, 130, 70, 80, 20, 56]])
exemplo_normalizado = scaler.transform(exemplo) # Normaliza antes de prever
print("Previsão para exemplo específico:", classifier.predict(exemplo_normalizado))

# 6. Previsões e Métricas
y_pred = classifier.predict(X_test)

print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório:\n", classification_report(y_test, y_pred))
