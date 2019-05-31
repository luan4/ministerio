import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from time import time
import pickle

class Sample:
    def __init__(self, entrada, target=None):
        self.entrada = np.array([entrada]).T
        if target:
            self.target = np.array([target]).T

class Layer:
    def __init__(self, entradas, salidas, l=1):
        self.weights = np.random.rand(salidas, entradas)*0.001
        self.bias = np.random.rand(salidas,1)*0.001
        self.l = l

    def sigmoid(self, X):
        return 1/(1+np.exp(-self.l*X))

    def __call__(self, entrada):
        self.output = self.sigmoid(self.weights @ entrada + self.bias)
        self.dv_matrix = np.diag([self.l*i*(1-i) for i in np.nditer(self.output)])
        return self.output

class Perceptron:
    def __init__(self, shape):
        self.layers = [Layer(shape[i], shape[i+1]) for i in range(len(shape)-1)]
        self.shape = shape

    def __call__(self, sample):
        activation = sample.entrada
        for layer in self.layers:
            activation = layer(activation)
        return activation

    def train(self, training_list, validation_list=None, max_epochs=100, learning_rate=0.5, mezclar=False):
        error_list=[]
        len_training_list = len(training_list)
        if validation_list:
            validation_error_list = []
            len_validation_list = len(validation_list)
        for epoch in range(max_epochs):
            progreso = 100*epoch/max_epochs
            if progreso.is_integer():
                print("{:.0f}%".format(progreso))
            if mezclar:
                shuffle(training_list)
            ECM = 0
            for sample in training_list:
                diff_vector = sample.target - self(sample)
                # calcular sensitivities
                sensitivities = [(-2)*self.layers[-1].dv_matrix @ diff_vector]
                for i in range(len(self.layers)-1):
                    sensitivities += [self.layers[-2-i].dv_matrix @ self.layers[-1-i].weights.T @ sensitivities[i]]
                sensitivities = sensitivities[::-1]
                #ajustar
                self.layers[0].weights -= learning_rate*sensitivities[0] @ sample.entrada.T
                self.layers[0].bias -= learning_rate*sensitivities[0]
                contador=1
                for layer in self.layers[1:]:
                    layer.weights -= learning_rate*sensitivities[contador] @ self.layers[contador-1].output.T
                    layer.bias -= learning_rate*sensitivities[contador]
                    contador+=1
                ECM += float(diff_vector.T @ diff_vector)
            error_list += [ECM/len_training_list]
            if validation_list:
                validation_ECM = 0
                for val_sample in validation_list:
                    val_diff_vector = val_sample.target - self(val_sample)
                    validation_ECM += float(val_diff_vector.T @ val_diff_vector)
                validation_error_list += [validation_ECM/len_validation_list]

        #plt.ion()
        plt.plot(error_list, label="training error")
        if validation_list:
            plt.plot(validation_error_list, label="validation error")
        plt.legend()
        plt.show()

# time1 = time()
# perceptron1 = Perceptron([2,5,1])
# train_list1 = [Sample([0,0], [0.3]), Sample([0,1],[0.7]), Sample([1,0],[0.7]), Sample([1,1],[0.3])]
# for i in train_list1:
#     print(i.entrada, perceptron1(i))
#     print()
# perceptron1.train(train_list1,10000,0.5)
# time2 = time()
# print('######')
# for i in train_list1:
#     print(i.entrada, perceptron1(i))
#     print()
# print(time2 - time1)
# salir = input("...")


with open("./pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)
train_imgs = data[0]
train_labels = data[1]
train_labels_one_hot = data[2]
train_sample_list = data[3]

perceptron_MNIST = Perceptron([784,16,16,10])
misteaks = 0
for i in train_sample_list[:1000]:
    a = list(perceptron_MNIST(i))
    b = list(i.target)
    if a.index(max(a))!=b.index(max(b)):
        misteaks += 1
print(str(100*(1-misteaks/1000))+"%")

perceptron_MNIST.train(train_sample_list[:1000], train_sample_list[1000:2000], max_epochs=110, learning_rate=0.5, mezclar=True)
print()
print()
misteaks = 0
for i in train_sample_list[1000:2000]:
    a = list(perceptron_MNIST(i))
    b = list(i.target)
    if a.index(max(a))!=b.index(max(b)):
        misteaks += 1
        if misteaks%10 == 0 and misteaks%2 == 0 and misteaks%3 == 0:
            img = i.entrada.reshape((28,28))
            plt.imshow(img, cmap="Greys")
            plt.xlabel(str(b.index(max(b))))
            plt.ylabel(str(a.index(max(a))))
            plt.show()
print(str(100 - 100*misteaks/1000)+"%")
salir = input("...")
