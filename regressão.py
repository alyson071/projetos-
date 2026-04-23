import numpy as np
import matplotlib.pyplot as plt

mean = 7  # media
std = 2  # desvio padrão
n_alunos = 30

# Gerando notas aleatorias
x = np.random.normal(loc=mean, scale=std, size=(n_alunos, 2))
x = np.clip(x, 0, 10)
y = (np.average(x, axis=1) >= mean) + 0.0

# Visualização dos dados
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

# Função sigmoid
def sigmoid(z):
    return 1 / (1 + np.e**-z)

# Regressão logística
def regressão_logistica(x, w, b):
    x_transposed = x.T
    return sigmoid(np.dot(w, x_transposed) + b)

# Inicialização dos parâmetros
w = np.random.rand(1, 2)
b = 0.0

def gradients(x, y, y_hat):
    n = x.shape[0]  # Número de amostras
    dw = np.zeros_like(w)  # Inicializa dw com o mesmo formato de w
    db = 0.0

    for i in range(n):
        x_i = x[i]  # x_i é um array de formato (2,)
        y_i = y[i]
        y_hat_i = y_hat[i]
        
        # Atualiza dw corretamente
        dw += (y_hat_i - y_i) * x_i
        db += (y_hat_i - y_i)

    dw /= n
    db /= n

    return dw, db

# Função de perda
def binary_cross_entropy(x, y, y_hat):
    n = x.shape[1]
    loss = 0
    epsilon = 1e-10
    for i in range(n):
        x_i, y_i, y_hat_i = x[:, i], y[i], y_hat[i]
        loss += -(y_i * np.log(y_hat_i + epsilon) + (1 - y_i) * np.log((1 - y_hat_i) + epsilon))
    return loss / n

# Loop de treinamento
learning_rate = 0.01
n_iterations = 1000

for _ in range(n_iterations):
    y_hat = regressão_logistica(x, w, b).squeeze()
    dw, db = gradients(x, y, y_hat)
    w -= learning_rate * dw
    b -= learning_rate * db



novo_exemplo = np.array([[6, 7]])  # Notas de um novo aluno
probabilidade = regressão_logistica(novo_exemplo, w, b)
classe = 1 if probabilidade >= 0.5 else 0
print("Classe prevista:", classe)

