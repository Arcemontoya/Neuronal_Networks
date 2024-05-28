import numpy as np
import pandas as pd
from Layer_Dense import Layer_Dense
from ActivationFunctions import *
from Layer_Dropout import Layer_Dropout
from Optimizer import *
from Loss import *

class Network:
    def __init__(self, list_neurons, list_activation_f, list_dropout, w_reg_l2, b_reg_l2, loss_f, optimizer):
        self.layer = []
        self.num_hidden_layers = 0
        self.activation = []
        self.dropout = []
        
        for index, (neuron1, neuron2) in enumerate(zip(list_neurons, list_neurons[1:])):
            if(w_reg_l2 and b_reg_l2 and (index == 0)):
                self.layer.append(Layer_Dense(neuron1, neuron2, weight_regularizer_l2=w_reg_l2, bias_regularizer_l2=b_reg_l2))
            else:    
                self.layer.append(Layer_Dense(neuron1, neuron2))
            self.num_hidden_layers += 1
        
        for activation in list_activation_f:
            self.activation.append(activation)

        for drop in list_dropout:
            self.dropout.append(Layer_Dropout(drop))

        self.loss_function = loss_f

        self.optimizer = optimizer

'''
red_clasificacion = Network([13, 60, 3],
                            [Activation_ReLU],
                            [0.7],
                            5e-4, 5e-4,
                            Activation_Softmax_Loss_CategoricalCrossentropy,
                            GDX_Optimizer(initial_learning_rate=2, decay=.9))

print(red_clasificacion.num_hidden_layers)  
'''
