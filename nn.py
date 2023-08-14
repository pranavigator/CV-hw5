import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    ##########################
    ##### your code here #####
    ##########################

    b = np.zeros(out_size)

    W = np.random.uniform(-np.sqrt(6)/np.sqrt(in_size + out_size), np.sqrt(6)/np.sqrt(in_size + out_size), 
                          size = (in_size, out_size))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    ##########################
    ##### your code here #####
    ##########################
    
    res = 1/(1 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################

    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = np.zeros(x.shape)

    ##########################
    ##### your code here #####
    ##########################

    for i in range(x.shape[0]):
        c = -np.max(x[i,:])
        res[i,:] = np.exp(x[i,:] + c)/np.sum(np.exp(x[i,:] + c))

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):

    ##########################
    ##### your code here #####
    ##########################
    # loss = 0.0
    loss = - np.sum(np.multiply(y, np.log(probs)))

    y_pred = np.argmax(probs, axis = 1)
    
    label = np.argmax(y, axis = 1)

    match_count = np.sum(y_pred == label)

    acc = match_count/len(label)

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    grad_X, grad_W, grad_b = np.zeros(X.shape), np.zeros(W.shape), np.zeros(b.shape)

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    ##########################
    ##### your code here #####
    ##########################

     # print("Res:", res.shape)
    # print("X:",X.shape)
    # print("W:", W.shape)
    # print("b:", b.shape)
    # print("delta:", delta.shape)

    # for i in range(X.shape[0]):
    #     res_iter = np.reshape(res[i,:],(res[i,:].shape[0],1))
    #     X_iter = np.reshape(X[i,:], (X[i,:].shape[0], 1))
    #     W_iter = np.reshape(W[i,:], (W[i,:].shape[0], 1))

    #     print("res iter:", res_iter.shape)
    #     print("X_iter:", X_iter.shape)
    #     print("W_iter:", W_iter.shape)
    #     # print(res_iter.T.shape)
    #     # grad_W_iter = np.dot(res_iter, X_iter.T)
    #     # print(grad_W_iter.shape)
    #     grad_W += np.dot(res_iter, X_iter.T).T
    #     grad_b = grad_b + res_iter
    #     grad_X[i, :] += np.dot(W_iter, res_iter)

    res = delta * activation_deriv(post_act)

    grad_W = np.dot(res.T, X).T
    grad_X = np.dot(res, W.T)
    grad_b = np.sum(res, axis = 0)
    # print(grad_b.shape)


    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################

    num_batches = x.shape[0]//batch_size

    batch_indices = np.random.choice(x.shape[0], size = (num_batches, batch_size), replace = False)

    for i in range(num_batches):
        row = batch_indices[i,:]
        batch_x = x[row]
        batch_y = y[row]
        batches.append((batch_x, batch_y))

    # print(batches)
    return batches
