import numpy as np

from urllib import request
import gzip
import pickle

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()


print("step0")
X_train = load()[0]     #mnist data         (60000,784)
Y_train = load()[1].T   #mnist label        (60000, )
X_test  = load()[2]     #mnist test data    (10000,784)
Y_test  = load()[3].T   #mnist test label   (10000, )

print("step1")
print(X_train)
print("step2")
import argparse
#parse the arguments
parser = argparse.ArgumentParser()

print("step3")
#hyperparameters setting

parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--n_x', type=int, default=784, help='number of inputs')
parser.add_argument('--n_h',type=int, default=64,help='number of hidden units')
parser.add_argument('--beta',type=float, default=0.9, help='parameter for momentum')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
print("step4")

#initialization
print("step5")
opt = parser.parse_args()
digits = 10
params = {  "W1" : np.random.randn(opt.n_h, opt.n_x) * np.sqrt(1. / opt.n_x),
            "b1": np.zeros((opt.n_h, 1)) * np.sqrt(1. / opt.n_x),
            "W2" : np.random.randn(digits, opt.n_h) * np.sqrt(1./opt.n_h),
            "b2": np.zeros((digits, 1)) * np.sqrt(1. / opt.n_h)}
print("step6")

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
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]

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


