import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from Layer_Dense import Layer_Dense
from ActivationFunctions import *
from Layer_Dropout import Layer_Dropout
from Optimizer import *
from Loss import *

# Importar módulos definidos previamente (clases de capas, activaciones, pérdidas, optimizadores)
# Se asume que estos módulos están definidos correctamente.

# Cargar el dataset
#"C:/Users/a1271/Downloads/wine_dataset.mat"
wine_data = loadmat('C:/Users/a1271/Downloads/wine_dataset.mat')
X = wine_data['wineInputs'].T  # Entradas
y = wine_data['wineTargets'].T  # Etiquetas

# Definir las capas de la red neuronal
dense1 = Layer_Dense(13, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.7)
#dropout2 = Layer_Dropout(0.5)
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#optimizer = Adadelta_Optimizer(learning_rate=1., decay=0.9, epsilon=1e-7)
optimizer = GDX_Optimizer(initial_learning_rate=1, decay=0.9)

accuracy_list = []
loss_list = []

# Entrenamiento de la red neuronal
for epoch in range(10001):
    dense1.forward(X)
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
plt.plot(accuracy_list)
plt.title("Precisión")
plt.xlabel("Iteraciones")
plt.ylabel("Valor")
plt.show()

plt.plot(loss_list)
plt.title("Pérdida")
plt.xlabel("Iteraciones")
plt.ylabel("Valor")
plt.show()
