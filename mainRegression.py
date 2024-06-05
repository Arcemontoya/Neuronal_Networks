import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from Layer_Dense import Layer_Dense
from ActivationFunctions import *
from Layer_Dropout import Layer_Dropout
from Optimizer import *
from Loss import *
from Normalizacion import *
from Metricas_desempeño import *

#TEMPORAL
from sklearn.preprocessing import MinMaxScaler
import sys

# Cargar el archivo .mat
#data = loadmat('engine_dataset.mat')
#X = data['engineInputs']
#y = data['engineTargets']

concrete_data = pd.read_csv('concrete_dataset.csv')
# Asumiendo que la última columna es el target
X_raw = concrete_data.iloc[:, :-1].values
y_raw = concrete_data.iloc[:, -1].values.reshape(-1, 1)

# Normalización
X = min_max_normalize(X_raw)
y = min_max_normalize(y_raw)

# Inicializar capas de la red
dense1 = Layer_Dense(X.shape[1], 60, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(60, 30)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(30, 1)
activation3 = Activation_Linear()

# Inicializar función de pérdida y optimizador
loss_funcion = Loss_MeanSquaredError()
optimizer = GDX_Optimizer(initial_learning_rate=1.6, decay=1e-3)
#optimizer = Adadelta_Optimizer(learning_rate=1.6, decay=0.9)

loss_list = []

# Entrenar la red
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    data_loss = loss_funcion.calculate(activation3.output, y)
    regularization_loss = loss_funcion.regularization_loss(dense1) + loss_funcion.regularization_loss(dense2) + loss_funcion.regularization_loss(dense3)
    loss = data_loss + regularization_loss
    loss_list.append(loss)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_funcion.backward(activation3.output, y)
    activation3.backward(loss_funcion.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

# Graficar la pérdida a lo largo de las épocas
plt.plot(loss_list)
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.show()

# Calcular R^2 para el target
r2 = calculate_r2(y, activation3.output)

plt.figure(figsize=(8, 6))
plt.scatter(y, activation3.output, alpha=0.5)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Línea de referencia para valores reales vs predichos
plt.title('Valores Reales vs Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.text(0.05, 0.9, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
plt.show()