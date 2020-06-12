import matplotlib.pyplot as plt 
import numpy as np 
from plot_commons import colors, set_params

set_params()

''' Creates the activation function plot used in neural network theory of master's'''

def leaky_relu(z): return (0.0*z if z < 0 else z)
def tanh(z): return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)
def sigmoid(z): return 1 / (1 + np.exp(-z))

vals = np.linspace(-4,4,101)
relus = [leaky_relu(val) for val in vals]
tanhs = [tanh(val) for val in vals]
sigms = [sigmoid(val) for val in vals]

f = plt.figure(figsize=(6,4.5))
plt.axvline(0,linestyle='--',color='black')
plt.plot(vals,[0 for i in range(len(vals))],'--',color='black')
plt.plot(vals,tanhs,color=colors[0],label='Tanh')
plt.plot(vals,sigms,color=colors[1],label='Sigmoid')
plt.plot(vals,relus,color=colors[2],label='Relu')
plt.xlabel('z')
plt.ylabel('f(z)')
plt.legend(loc='best').set_draggable(True)
plt.show()