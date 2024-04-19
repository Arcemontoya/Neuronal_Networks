from Layer_Dense import Layer_Dense
from ActivationFunctions import *
from Loss import Loss_CategoricalCrossentropy, Activation_Softmax_Loss_CategoricalCrossentropy

# Crea una capa con 25 entradas y 15 valores de salida
ld = Layer_Dense(25,15)

# Función de activación para capa oculta 1
activation1 = Activation_ReLU()

# Crea una segunda capa con 3 entradas (aquí va como entrada
# la salida de la primera capa) y 2 valores de salida
ld2 = Layer_Dense(15, 2)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Función de activación para la segunda capa oculta
activation2 = Activation_Softmax()

loss_function = Activation_Softmax_Loss_CategoricalCrossentropy()

#dense1.forward(X)
#activation.forward(dense1.output)
#dense2.forward(X)
loss = loss_activation.forward(activation1.output)
#print(loss_activation.output[:5])
#print('loss:', loss)

#predictions = np.argmax(loss_activation.output, axis = 1)
#if len(y.shape) == 2:
#    y = np.argmax(y, axis = 1)
#accuracy = np.mean(predictions == y)

#print ('acc:', accuracy)

loss_activation.backward(loss_activation.output, y)
ld2.backward(loss_activation.dinputs)
activation1.backward(ld2.dinputs)
ld.backward(activation1.dinputs)

#Imprimir gradientes
#print(ld.dweights)
#print(ld.biases)
#print(ld2.dweights)
#print(ld2.biases)