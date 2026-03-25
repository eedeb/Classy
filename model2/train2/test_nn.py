N1=[0.5,-0.5]
N1_bias=0.0

N2=[0.3,0.8]
N2_bias=0.0

O1=[0.7,-0.6]
O1_bias=0.0

lr=0.01

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

def forward_pass(I1,I2,expected):
    global N1,N1_bias,N2,N2_bias,O1,O1_bias
    #hidden layer
    N1_output=sigmoid(I1*N1[0] + I2*N1[1] + N1_bias)
    N2_output=sigmoid(I1*N2[0] + I2*N2[1] + N2_bias)
    O1_output=sigmoid(N1_output*O1[0] + N2_output*O1[1] + O1_bias)
    #backprop
    error=expected - O1_output
    delta_out=error * sigmoid_derivative(O1_output)
    #Adjust Neuron 2
    delta_out_N2=delta_out*sigmoid_derivative(N2_output)*O1[1]
    N2[0] += lr*delta_out_N2*I1
    N2[1] += lr*delta_out_N2*I2
    N2_bias += lr*delta_out_N2
    #Adjust Neuron 1
    delta_out_N1=delta_out*sigmoid_derivative(N1_output)*O1[0]
    N1[0] += lr*delta_out_N1*I1
    N1[1] += lr*delta_out_N1*I2
    N1_bias += lr*delta_out_N1
    #adjust output Neuron
    O1[0] += lr*delta_out*N1_output
    O1[1] += lr*delta_out*N2_output
    O1_bias += lr*delta_out
def test(I1,I2):
    N1_output=sigmoid(I1*N1[0] + I2*N1[1] + N1_bias)
    N2_output=sigmoid(I1*N2[0] + I2*N2[1] + N2_bias)
    O1_output=sigmoid(N1_output*O1[0] + N2_output*O1[1] + O1_bias)
    return O1_output
import random
def train(epochs):
    for epoch in range(epochs):
        combinations=[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
        random.shuffle(combinations)
        for combo in combinations:
            forward_pass(combo[0],combo[1],combo[2])
train(100000)

print(test(0,0),
test(0,1),
test(1,0),
test(1,1))
# 0, 1, 1, 0