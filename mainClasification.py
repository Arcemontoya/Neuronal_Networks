import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Layer_Dense import Layer_Dense
#from ActivationFunctions import *
from Layer_Dropout import Layer_Dropout
from Optimizer import *
from Loss import *
from Arquitecture import *
import seaborn as sns
from sklearn.metrics import confusion_matrix

#TEMPORAL
from sklearn.preprocessing import MinMaxScaler


# Importar módulos definidos previamente (clases de capas, activaciones, pérdidas, optimizadores)
# Se asume que estos módulos están definidos correctamente.

# Cargar el dataset .mat
#wine_data = loadmat('wine_dataset.mat')
#X = wine_data['wineInputs'].T  # Entradas
#y = wine_data['wineTargets'].T  # Etiquetas

# Carga el dataser .csv
wine_data = pd.read_csv('wine_dataset.csv')
wine_data_shuffled = wine_data.sample(frac=1).reset_index(drop=True)
X = wine_data.iloc[:, :-3].values  # Todas las columnas excepto las últimas 3
y = wine_data.iloc[:, -3:].values  # Las últimas 3 columnas

print(X.shape)

#Normalizacion con sklearn TEMPORAL
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Definir las capas de la red neuronal
dense1 = Layer_Dense(13, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.8)
#dropout2 = Layer_Dropout(0.5)
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Adadelta_Optimizer(learning_rate=1., decay=0.9, epsilon=1e-7)
#optimizer = GDX_Optimizer(initial_learning_rate=2, decay=1e-3)

accuracy_list = []
loss_list = []

# Entrenamiento de la red neuronal
for epoch in range(10000):
    dense1.forward(X_normalized)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
#    dropout2.forward(dense2.output)

    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    loss_list.append(loss)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y_true = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y_true)
    accuracy_list.append(accuracy)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate:.3f}, ')

    loss_activation.bakcward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Graficar la precisión y la pérdida
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(accuracy_list)
plt.title("Precisión")
plt.xlabel("Epoca")
plt.ylabel("Valor")

plt.subplot(1, 2, 2)
plt.plot(loss_list)
plt.title("Pérdida")
plt.xlabel("Epoca")
plt.ylabel("Valor")

plt.show()

# Obtener predicciones finales
final_predictions = np.argmax(loss_activation.output, axis=1)
final_true_labels = np.argmax(y, axis=1)

#matriz de confucion
conf_matrix = confusion_matrix(final_true_labels, final_predictions)
class_names = ['Clase 1', 'Clase 2', 'Clase 3']

print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()