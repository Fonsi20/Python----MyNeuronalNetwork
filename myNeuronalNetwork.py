import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.datasets import make_circles
from IPython.display import clear_output

#   CREATE DATASET
n = 500
p = 2
X,Y=make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

#   SHOW DATA
#   Circle 2
#plt.scatter(X[Y[:,0]==0,0],X[Y[:,0]==0,1], c="skyblue")
#   Circle 1
#plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1], c="salmon")
#plt.axis("equal")
#plt.show()


#  ACTIVATION FUNTIONS
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))

relu = lambda x: np.maximum(0, x)

#   SHOW ACTIVATIONS
_x = np.linspace(-5, 5, 100)
#plt.plot(_x, sigm[0](_x))
#plt.show()
#plt.plot(_x, sigm[1](_x))
#plt.show()
#plt.plot(_x, relu(_x))
#plt.show()

#   CLASS OF THE LAYER OF THE NETWORK
class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur)      * 2 - 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1

#   CREATE MY NEURONAL NETWORK
#   MANUAL
#l0 = neural_layer(p, 4, sigm)
#l1 = neural_layer(4, 8, sigm)
#l2 = neural_layer(8, 12, sigm)
#l3 = neural_layer(12, 16, sigm)
#   ...

#   AUTOMATIC
def create_nn(topology, act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l],topology[l+1],act_f))
    return nn

topology = [p,4,8,1]

#   Train our neuronal network
#   Error cuadratico medio
l2_cost = (lambda Yp, Yr: np.mean((Yp- Yr) ** 2),
           lambda Yp, Yr: (Yp- Yr))

neural_net = create_nn(topology,sigm)

def train(neuronal_net,X,Y,l2_cost,lr,train):
    #   FORWAR PASS
    #   MANUAL
    #z = X @ neural_layer[0].W + neuronal_net[0].b
    #a = neuronal_net[0].act_f(z)

    #   AUTOMATIC
    out = [(None,X)]
    for l, layer in enumerate(neuronal_net):
        z = out[-1][1] @ neuronal_net[l].W + neuronal_net[l].b
        a = neuronal_net[l].act_f[0](z)
        out.append((z,a))

    #ERROR
    #print(l2_cost[0](out[-1][1], Y))

    if train:
        #BACKWARD PASS
        deltas = []

        for l in reversed(range(0,len(neuronal_net))):
            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(neuronal_net) - 1:
                #CALCUALTE DELTA LAST LAYER
                deltas.insert(0, l2_cost[1](a, Y) * neuronal_net[l].act_f[1](a))
            else:
                #CALCULATE DELTA RESTECT LAYER PREVIEW
                deltas.insert(0, deltas[0] @ _W.T * neuronal_net[l].act_f[1](a))
            _W = neuronal_net[l].W

            #GRADIENT DESCENT
            neuronal_net[l].b = neuronal_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neuronal_net[l].W = neuronal_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]

#   TEST NEURONAL NETWORK
loss= []
for i in range(2500):
    #   START
    pY = train(neural_net, X, Y, l2_cost, 0.05, True)
    if i % 25 == 0:
        loss.append(l2_cost[0](pY, Y))
        res = 50
        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)

        _Y = np.zeros((res,res))

        for i0,x0 in enumerate(_x0):
            for i1,x1 in enumerate(_x1):
                _Y[i0, i1] = train(neural_net, np.array([[x0, x1]]), Y, l2_cost, 0.5, False)[0][0]

        plt.subplot(2, 1, 1)
        plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c = "skyblue")
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c = "salmon")
        plt.axis("equal")
        plt.title('Graph dots')

        clear_output(wait=True)
        plt.subplot(2, 1, 2)
        plt.plot(range(len(loss)),loss)
        plt.title('Graph error')
        plt.show()
        time.sleep(0.5)
