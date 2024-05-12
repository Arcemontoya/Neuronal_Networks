import nnfs

from Layer_Dense import *
from ActivationFunctions import *
from Loss import Loss_CategoricalCrossentropy, Activation_Softmax_Loss_CategoricalCrossentropy
from Optimizer import *
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)

#239
# Crea una capa con 25 entradas y 15 valores de salida
ld = Layer_Dense(2,3)

# Función de activación para capa oculta 1
activation1 = Activation_ReLU()

# Crea una segunda capa con 3 entradas (aquí va como entrada
# la salida de la primera capa) y 2 valores de salida
ld2 = Layer_Dense(3, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Función de activación para la segunda capa oculta
activation2 = Activation_Softmax()

loss_function = Activation_Softmax_Loss_CategoricalCrossentropy()

#Crear optimizador
optimizer = Optimizer_SGD(learning_rate=1, decay=0., momentum=0.)

for epoch in range(1001):

    ld.forward(X)

    activation1.forward(ld.output)

    ld2.forward(activation1.output)

    loss = loss_activation.forward(ld2.output, y)

    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch%100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    ld2.backward(loss_activation.dinputs)
    activation1.backward(ld2.dinputs)
    ld.backward(activation1.dinputs)

    # Actualizar pesos y sesgos
    optimizer.pre_update_params()
    optimizer.update_params(ld)
    optimizer.update_params(ld2)
    optimizer.post_update_params()



