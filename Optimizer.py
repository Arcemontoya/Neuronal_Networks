import numpy as np

# Todavía en pruebas
class GDX_Optimizer:
    def __init__(self, initial_learning_rate=1, decay=0.01):
        self.initial_learning_rate = initial_learning_rate
        self.decay = decay
        self.iterations = 0
        self.current_learning_rate = initial_learning_rate

    def pre_update_params(self):
        # Actualiza la tasa de aprendizaje antes de cada iteración
        self.current_learning_rate = self.initial_learning_rate * (1 - self.decay) ** self.iterations

    def update_params(self, layer):
        # Aplanar los pesos para simplificar la actualización
        weights = layer.weights.flatten()
        # Obtener el gradiente de los pesos
        df_val = layer.dweights.flatten()
        # Actualizar los pesos
        weights -= self.current_learning_rate * df_val
        # Volver a dar forma a los pesos al tamaño original
        layer.weights = weights.reshape(layer.weights.shape)

    def post_update_params(self):
        # Incrementa el contador de iteraciones
        self.iterations += 1


class Adadelta_Optimizer:
    def __init__(self, learning_rate=1., decay=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.accumulated_gradients_w = None
        self.accumulated_gradients_b = None
        self.accumulated_updates_w = None
        self.accumulated_updates_b = None

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        if not (hasattr(layer, 'weight_cache')):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.delta_weights = np.zeros_like(layer.weights)
            layer.delta_biases = np.zeros_like(layer.biases)

            # Actualiza cache
        layer.weight_cache = self.decay * layer.weight_cache + (1 - self.decay) * (layer.dweights ** 2)
        layer.bias_cache = self.decay * layer.bias_cache + (1 - self.decay) * (layer.dbiases ** 2)

            # Calcula la actualización de pesos y sesgos
        weight_update = -(np.sqrt(layer.delta_weights + self.epsilon) / np.sqrt(
            layer.weight_cache + self.epsilon)) * layer.dweights
        bias_update = -(np.sqrt(layer.delta_biases + self.epsilon) / np.sqrt(
            layer.bias_cache + self.epsilon)) * layer.dbiases

            # Actualiza los pesos y sesgos
        layer.weights += weight_update
        layer.biases += bias_update

            # Actualiza delta
        layer.delta_weights = self.decay * layer.delta_weights + (1 - self.decay) * (weight_update ** 2)
        layer.delta_biases = self.decay * layer.delta_biases + (1 - self.decay) * (bias_update ** 2)
    def post_update_params(self):
        self.iterations += 1

#No necesarios para el proyecto
class Optimizer_SGD:

    # Inicializar optimizador
    # Learning rate de 1.
    def __init__(self, learning_rate=1, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Se llama antes de que se actualice cualquier parametro
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1.+self.decay * self.iterations))

    # Actualizar parametros
    def update_params(self, layer):
        # Si usamos momentum
        if self.momentum:
            #Si la capa no contiene arreglos momentum, crearlas llenas con ceros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # Si no hay arreglo de momentum para pesos
                # No existen tampoco para sesgos
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Crear weight updates con momentum - toma las prevas
            # Las actualizaciones se multiplican por el retain factor
            # Y actualiza las gradientes actuales
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights

            layer.weight_momentums = weight_updates

            # Crea las actualizaciones de sesgos
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
                # Vanilla SDG updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Actualizar pesos y biases usando cualquiera de vanilla o momentum update
        layer.weights += weight_updates
        layer.biases += bias_updates

        # Se llama una vez después se actualice cualquier parametro
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:

    def __init__(self, learning_rate = 1., decay = 0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):

        if not (hasattr(layer, 'weight_cache')):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate + layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    def __init__(self, learning_rate = 0.001, decay = 0, epsilon= 1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1.+self.decay * self.iterations))
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2

        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        #Alomejor
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter (Idk posiblemente mal pipipipi)
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
                    np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
                    np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1