import numpy as np
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


# Cargar el archivo .mat
data = loadmat('C:/Users/a1271/Downloads/engine_dataset.mat')

# Imprimir las llaves del archivo para entender su estructura
print(data.keys())

# Asumir que los datos están en las llaves 'X' y 'y' (esto puede variar)
X = data['engineInputs']
y = data['engineTargets']

# Imprimir las formas de los datos
print('X shape:', X.shape)
print('y shape:', y.shape)

# Normalizar los datos si es necesario
X = (X - np.min(X)) / (np.max(X) - np.min(X))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# Inicializar capas de la red
dense1 = Layer_Dense(X.shape[1], 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()

# Inicializar función de pérdida y optimizador
loss_funcion = Loss_MeanSquaredError()
optimizer = Adadelta_Optimizer(learning_rate=1., decay=0.9, epsilon=1e-7)

accuracy_precision = np.std(y) / 250

# Entrenar la red
accuracy_list = []
loss_list = []

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

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    accuracy_list.append(accuracy)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
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

# Graficar la precisión y la pérdida a lo largo de las épocas
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(accuracy_list)
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')

plt.subplot(1, 2, 2)
plt.plot(loss_list)
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')

plt.show()