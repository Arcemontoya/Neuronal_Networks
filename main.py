from Layer_Dense import Layer_Dense
from ActivationFunctions import Activation_ReLU
from ActivationFunctions import Activation_Sigmoid
from Loss import Loss_CategoricalCrossentropy

# Crea una capa con 25 entradas y 15 valores de salida
ld = Layer_Dense(25,15)

# Función de activación para capa oculta 1
activation1 = Activation_ReLU()

# Crea una segunda capa con 3 entradas (aquí va como entrada
# la salida de la primera capa) y 2 valores de salida
ld2 = Layer_Dense(15, 2)

# Función de activación para la segunda capa oculta
activation2 = Activation_Sigmoid()

loss_function = Loss_CategoricalCrossentropy()

#dense1.forward(X)
#activation.forward(dense1.output)
#dense2.forward(X)
#activation2.forward(dense2.output)
#print(activation2.output[:5])
# loss = loss_function.calculate(activation2.output, y)