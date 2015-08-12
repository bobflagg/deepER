from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred, target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))

def compute_f1(y_true, y_pred, tagnames):
    _, _, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    return 100*sum(f1[1:] * support[1:])/sum(support[1:])

##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3, dims=[None, 100, 5], reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate
        self.nclass = dims[2] # number of output classes
        self.windowsize = windowsize # size of context window
        self.n = wv.shape[1] # dimension of word vectors

        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(
            W=(dims[1], dims[0]),
            b1=(dims[1],),
            U=(dims[2], dims[1]),
            b2=(dims[2],),
        )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####
        self.sparams.L = wv.copy() # store own representations
        self.params.W = random_weight_matrix(*self.params.W.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)
        # self.params.b1 = zeros((dims[1],1)) # done automatically!
        # self.params.b2 = zeros((self.nclass,1)) # done automatically!
        #### END YOUR CODE ####

    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####
        ##
        # Forward propagation
        # build input context
        x = self.build_input_context(window)
        # first hidden layer
        z1 = self.params.W.dot(x) + self.params.b1
        a1 = tanh(z1)
        # second hidden layer
        z2 = self.params.U.dot(a1) + self.params.b2
        a2 = softmax(z2)
        ##
        # Backpropagation
        # second hidden layer
        delta2 = a2 - make_onehot(label, self.nclass)
        self.grads.b2 += delta2
        self.grads.U += outer(delta2, a1) + self.lreg * self.params.U
        # first hidden layer
        delta1 = (1.0 - a1**2) * self.params.U.T.dot(delta2)
        self.grads.b1 += delta1
        self.grads.W += outer(delta1, x) + self.lreg * self.params.W
        
        for j, idx in enumerate(window): 
            start = j * self.n
            stop = (j + 1) * self.n
            self.sgrads.L[idx] = self.params.W[:,start:stop].T.dot(delta1)
        #### END YOUR CODE ####

    def build_input_context(self, window):
        x = zeros((self.windowsize * self.n,))
        for j, idx in enumerate(window): 
            start = j * self.n
            stop = (j + 1) * self.n
            x[start:stop] = self.sparams.L[idx]
        return x
        
    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####
        # doing this first as a loop
        n_windows = len(windows)
        P = zeros((n_windows,self.nclass))
        for i in range(n_windows):
            x = self.build_input_context(windows[i])
            # first hidden layer
            z1 = self.params.W.dot(x) + self.params.b1
            a1 = tanh(z1)
            # second hidden layer
            z2 = self.params.U.dot(a1) + self.params.b2
            P[i,:] = softmax(z2)
        '''
        x = np.zeros((n_windows,self.windowsize * self.n))
        for i in range(n):
            x[i,:] = self.build_input_context(window[i])
        # first hidden layer
        z1 = self.params.W.dot(x) + self.params.b1
        a1 = np.tanh(z1)
        # second hidden layer
        z2 = self.params.U.dot(a1) + self.params.b2
        a2 = softmax(z2)
        '''
        #### END YOUR CODE ####

        return P # rows are output for each input

    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####
        P = self.predict_proba(windows)
        c = argmax(P, axis=1)
        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####
        P = self.predict_proba(windows)
        N = P.shape[0]
        J = -1.0 * sum(log(P[range(N),labels]))
        J += (self.lreg / 2.0) * (sum(self.params.W**2.0) + sum(self.params.U**2.0))
        #### END YOUR CODE ####
        return J