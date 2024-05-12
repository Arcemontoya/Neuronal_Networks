import numpy as np

# Todavía en prueba
class Adadelta_Optimizer:

    def __init__(self, model, decay_factor=0.9, eps=0.0000001):
        self.model = model
        self.decay_factor = decay_factor
        self.eps = eps
        self.cache_weights = [np.zeros_like(layer.weights) for layer in model]
        self.cache_biases = [np.zeros_like(layer.biases) for layer in model]
        self.delta_weights = [np.zeros_like(layer.weights) for layer in model]
        self.delta_biases = [np.zeros_like(layer.biases) for layer in model]

    def update_params(self, xs, ys):
        inputs = xs
        for layer in self.model:
            layer.forward(inputs)
            inputs = layer.output


        # Backward pass
        dvalues = None
        for i, layer in enumerate(reversed(self.model)):
            if i == 0:
                loss = layer.backward(ys)
                dvalues = loss.dinputs
            else:
                layer.backward(dvalues)
                dvalues = layer.dinputs
            dweights = layer.dweights
            dbiases = layer.dbiases

            # Update cache
            self.cache_weights[i] = self.decay_factor * self.cache_weights[i] + (1-self.decay_factor) * (dweights **2)
            self.cache_biases[i] = self.decay_factor * self.cache_biases[i] + (1-self.decay_factor) * (dbiases**2)

            #Compute update
            weight_update = np.sqrt(self.delta_weights[i] + self.eps) / np.sqrt(self.cache_weights[i] + self.eps) * dweights
            bias_update = np.sqrt(self.delta_biases[i] + self.eps) / np.sqrt(self.cache_biases[i] + self.eps) * dbiases

            # Update parameters
            layer.weights -= weight_update
            layer.biases -= bias_update

            #Update delta
            self.delta_weights[i] = self.decay_factor*self.cache_weights[i]+(1-self.decay_factor)*(weight_update**2)
            self.delta_biases[i] = self.decay_factor * self.delta_biases[i] + (1-self.decay_factor)*(bias_update**2)

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

    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta_1=0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1.+self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        # Corregir momentum
        # self.iteration es 0 a la primera ejecucion
        # y se comienza por 1
        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations+1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations+1))
        #Actualizar cache con los gradientes cuadrados actuales
        layer.weight_cache = self.beta2 * layer.weight_cache + \
                             (1 - self.beta_2)*layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1-self.beta) * layer.dbiases ** 2

        # Cache corregido
        weight_cache_corrected = layer.weight_cache / \
                                 (1-self.beta_2 ** (self.iterations+1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations+1))

        # Vanilla SGD parameter update + normalizacion con squared rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate + bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1