import numpy as np

class Activation_ReLU:

    def forward(self, inputs):
        #Calcular valores de input
        self.output = np.maximum(0, inputs)

class Activation_Sigmoid:

    def forward(self, inputs):
        #Calcular valores de input
        self.output = 1.0 / (1.0 + np.exp(-inputs))