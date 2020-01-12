
import h5py
"""
MNIST_data = h5py.File("../MNISTdata.hdf5", 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
x_test  = np.float32(MNIST_data['x_test'][:])
y_test  = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
MNIST_data.close()
"""
import numpy as np
import opt
from mlxtend.data import loadlocal_mnist

#load MNIST data
#MNIST_data = gzip.File("/home/dyrim/CNN/mnist.hdf5",'r')
"""
train image has 60000*784 matrix 
train label has have 60000*1   matrix
"""
##reading ubyte type dataset
xs_train, ys_train = loadlocal_mnist(images_path = '/home/dyrim/CNN/train-images-idx3-ubyte', labels_path='train-labels-idx1-ubyte')
xs_test, ys_test = loadlocal_mnist(images_path='/home/dyrim/CNN/t10k-images-idx3-ubyte', labels_path='t10k-labels-idx1-ubyte')

#printing first row of train set 
#print('DimensionsL %s x %s' % (xs_train.shape[0], xs_train.shape[1]))
#print('\n1st row', xs_train[0])


x_train = np.float32(xs_train[:])
y_train = np.int32(np.array(ys_train[:]))
x_test = np.float32(xs_test[:])
y_test = np.int32(np.array(ys_test[:]))

# stack together for next step
X = np.vstack((x_train, x_test)) #concatenates vertically
y = np.vstack((y_train[60000:1].T, y_test[10000:1].T)) # y_train 60000*1, y_test 10000*1 
y = y.T

#one-hot encoding
digits = 10
examples = y.shape[0] #y.shape[0] = 70000, y.shape[1] = 1 
y = y.reshape(1, examples) #line 34 : y = np. hstack((y_train, y_test)) << could be done easily
Y_new = np.eye(digits)[y.astype('int32')]#eye makes 10*10 identity matrix
Y_new = Y_new.T.reshape(digits,examples) #(10 , 70000)T <<<여기 잘 모르겠음. 

#number of training set
m = 60000
m_test = X.shape[0] - m #??? used nowhere??? 여기말고는 쓰이는 곳이 없는데 왜 쓰여있는거지?
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]

#shuffle training set
shuffle_index = np.random.permutation(m)
#X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

#initialization
params = {  "W1" : np.random.randn(opt.n_h, opt.n_x)*np.sqrt(1. / opt.n_x),
            "b1": np.zeros((opt.n_h, 1))*np.sqrt(1. / opt.n_x),
            "W2" : np.random.randn(digits, opt.n_h)* np.sqrt(1./opt.n_h),
            "b2": np.zeros((digits, 1))*np.sqrt(1. / opt.n_h)}

def sigmoid(z):
    """
    sigmoid activation function.

    inputs : z
    outputs: sigmoid(z)
    """
    s = 1. /(1. + np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    """
    compute loss function
    """
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m)*L_sum

    return L

    def feed_forward(X, params) :
        """
        feed forward network: 2 - layer neural net

        inputs:
            params: dictionay a dictionary contains all the weights and biases
        
        return:
            cache: dictionay a dictionary contains all the fully connected units and activations
        """
        cache={}

        # Z1 = W1.dot(x) + b1
        cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]

        # A1 = sigmoid(Z1)
        cache["A1"] = sigmoid(cache["Z1"])

        # Z2 = W2.dot(A1) +b2
        cache["Z2"] = np.matmul(params["W2"], cache["A1"] + params["b2"])

        # A2 = softmax(Z2) 
        cache["A2"] = np.exp(cache["Z2"])/np.sum(np.exp(cache["Z2"]), axis = 0)

        return cache

def back_propagate(Z, Y, params, cache, m_batch) : 
    """
    back propagation

    inputs: 
        params: dictionay a dictionary conatin all the weights and biases
        cache: dictionay a dictionary contains all the full connected units and activations

    return:
        grads: dictionay a dictionary contains the gradients of corresponding weights and biases
    """

    #error at last layer
    dZ2 = cache["A2"] - Y

    #gradients at last layer (Py2 need 1. to transform to float)
    dW2 = (1. / m_batch ) * np.matmul(dZ1, X.T)
    db2 = (1. / m_batch ) * np.sum(dZ2, axis=1, keepdims=True)

    #back propagate through first layer
    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1- sigmoid(cache["Z1"]))

    #gradients at first layer (Py2 need 1. to transform to float)
    dW1 = (1./ m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./ m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1" : dW1, "db1" : db1, "dW2" : dW2, "db2" : db2 }

    return grads

import argparse

parser = argparse.ArgumentParser()

#hyperparameters setting

parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
parser.add_argument('--n_h',type=int, default=64,help='number of hidden units')
parser.add_argument('--beta',type=float, default=0.9, help='parameter for momentum')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

#parse the arguments
opt = parser.parse_args()

#training 

for i in range(opt.epochs) :

    #shuffle training set
    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):
        #get mini-batch
        begin = j*opt.batch_size
        end = min(begin + opt.batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        #forward and backward
        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache, m_batch)

        #with momentum(optional)
        dW1 = (opt.beta*dW1 + (1. - opt.beta)*grads["dW1"])
        db1 = (opt.beta*db1 + (1. - opt.beta)*grads["db1"])
        dW2 = (opt.beta*dW2 + (1. - opt.beta)*grads["dW2"])
        db2 = (opt.beta*db2 + (1. - opt.beta)*grads["db2"])
         
    #forward pass on training set
    cache = feed_forward(X_train, params)
    train_loss = compute_loss(Y_train, cache["A2"])

    #forward pass on test set
    cache = feed_forward(X_test, params)
    test_loss = compute_loss(Y_test, cache["A2"])
    print("Epoch {} : training loss = {}, test loss = {}".format(i+1,train_loss,test_loss))








