import numpy as np
from ActivationFunctions import *
from Layer_Dense import *
from Loss import *
import nnfs
from Optimizer import *
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)
#optimizer = Adadelta_Optimizer(learning_rate=1, decay=0.9, epsilon=1e-7) # Corregir
#optimizer = GDX_Optimizer(initial_learning_rate=0.01) # Corregir

accuracy_list = []
loss_list = []

for epoch in range(1000):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

    loss = data_loss + regularization_loss
    loss_list.append(loss)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    accuracy_list.append(accuracy)

    if not epoch%100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'data_loss: {data_loss:.3f}, '+
              f'reg_loss: {regularization_loss:.3f}, '+
              f'lr: {optimizer.current_learning_rate:.3f}, ')

    loss_activation.bakcward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.pre_update_params()

# Validar modelo

#X_test, y_test = spiral_data(samples=100, classes=3)

#dense1.forward(X_test)
#activation1.forward(dense1.output)
#dense2.forward(activation1.output)

#loss = loss_activation.forward(dense2.output, y_test)

#predictions = np.argmax(loss_activation.output, axis=1)
#if len(y_test.shape == 2):
#    y_test = np.argmax(y_test, axis=1)
#accuracy = np.mean(predictions == y_test)

#print(f'validation, acc: {accuracy: .3f}, loss: {loss:.3f}')

plt.plot(accuracy_list)
plt.title("Precision")
plt.xlabel("Iteraciones")
plt.ylabel("Valor")
plt.show()

plt.plot(loss_list)
plt.title("Perdida")
plt.xlabel("Iteraciones")
plt.ylabel("Valor")
plt.show()