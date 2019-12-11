import numpy as np

from exercise_code.layers import *
from exercise_code.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0, loss_function='softmax'):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        if loss_function != 'softmax':
            raise Exception('Wrong loss function')
        else:
            self.loss_function = 'softmax'

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # weights matrices need to have the correct shape
        # W1.shape = inputs as rows and  hidden_dim as
        # columns also scaled
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        # W2.shape = hidden_dim as rows and output(num_classes) as
        # columns also scaled
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        # biases are just a vector
        # b1 needs size of hidden_dim
        self.params['b1'] = np.zeros(hidden_dim)
        # b2 needed size of output(num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        # Store self variables
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        reg = self.reg
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # out_layer_1 will be forewarded also cache_1 will be used
        # for backprop; affine_relu_forward takes input X, W1, b1
        out_layer_1, cache_1 = affine_relu_forward(X, W1, b1)
        # scores calculation is done after this foreward pass
        # chache_2 will be used for backprop;
        # affine_forward takes the output from the previous layer
        # out_layer_1, W2, b2
        scores, cache_2 = affine_forward(out_layer_1, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # softmax_loss returns the loss given the scores and labels
        # and also the gradients with respect to x
        loss, dx = softmax_loss(scores, y)
        # L2 regularization
        loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))

        # affine_backward takes cache_2 and the gradients
        # to perform backprop and stores the gradients for W2, b2
        dx2, dw2, db2 = affine_backward(dx, cache_2)
        # regularization
        grads['W2'] = dw2 + reg * W2
        grads['b2'] = db2

        # affine_relu_forward takes cache_1 and the gradients
        # to perform backprop and stores the gradients for W1, b1
        dx1, dw1, db1 = affine_relu_backward(dx2, cache_1)
        # regularization
        grads['W1'] = dw1 + reg * W1
        grads['b1'] = db1
        # loss and gradients have been computed and will now be returned
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - loss function

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None,
                 loss_function='softmax'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.loss_function = loss_function
        if loss_function == 'softmax':
            self.chosen_loss_function = softmax_loss
        elif loss_function == 'l2':
            self.chosen_loss_function = l2_loss
        else:
            raise Exception('Wrong loss function')

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        # unpack hidden_dims
        # store the dimensions in the right order in the dims array
        dims = [input_dim, *hidden_dims, num_classes]
        # initialize Wx and bx as before but this time in a loop
        # ranging from 1 (W1, b1) up to len(dims) (Wn, bn)
        for i in range(1, len(dims)):
            # if batchnorm is wanted add a gamma and beta term for every layer
            # except for the last one which does per definition not use it
            if use_batchnorm and i != self.num_layers:
                self.params['gamma' + str(i)] = np.ones(dims[i])
                self.params['beta' + str(i)] = np.ones(dims[i])
            # make sure to initialize Wi with the correct dimensions
            self.params['W' + str(i)] = np.random.randn(dims[i - 1],
                                                        dims[i]) * weight_scale
            # initialize bi with zeros of size dim[i]
            self.params['b' + str(i)] = np.zeros(dims[i])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        """
        B=A creates a reference
        B[:]=A makes a copy
        numpy.copy(B,A) makes a copy
        https://stackoverflow.com/questions/19676538/numpy-array-assignment-with-copy
        """
        out = X.copy()
        num_layers = self.num_layers
        cache = {}

        # again use the same loop staring with 1
        for i in range(1, num_layers):
            # if batchnorm is used:
            if self.use_batchnorm:
                # use affine_batchnorm_relu_forward
                out, cache[i] = affine_batchnorm_relu_forward(
                    out,
                    self.params['W' + str(i)],
                    self.params['b' + str(i)],
                    self.params['gamma' + str(i)],
                    self.params['beta' + str(i)],
                    self.bn_params[i - 1])
            else:
                # if not: use affine_relu_forward
                out, cache[i] = affine_relu_forward(out,
                                            self.params['W' + str(i)],
                                            self.params['b' + str(i)])

            if self.use_dropout:
                # if dropout is used: use dropout_forward and merge the caches
                out, dropout_cache = dropout_forward(out, self.dropout_param)
                cache[i] = (cache[i], dropout_cache)
        # the last layer is again per definition affine_forward with index
        # equal to num_layers
        scores, cache[num_layers] = affine_forward(out,
                                            self.params['W' + str(num_layers)],
                                            self.params['b' + str(num_layers)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        # We compute loss and gradient of last layer
        # For another notebook, we will need to switch between different loss functions
        # By default we choose the softmax loss
        loss, dscores = self.chosen_loss_function(scores, y)
        #######################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store#
        # the loss in the loss variable and gradients in the grads dictionary.#
        #                                                                     #
        #1_FullyConnectedNets.ipynb                                           #
        # Compute                                                             #
        # data loss using softmax, and make sure that grads[k] holds the      #
        # gradients for self.params[k]. Don't forget to add L2 regularization!#
        #                                                                     #
        # When using batch normalization, you don't need to regularize the    #
        # scale and shift parameters.                                         #
        #                                                                     #
        # NOTE: To ensure that your implementation matches ours and you pass  #
        # the automated tests, make sure that your L2 regularization includes #
        # a factor of 0.5 to simplify the expression for the gradient.        #
        #                                                                     #
        #######################################################################
        num_layers = self.num_layers
        # calculate the loss over all the weights of all layers
        loss += 0.5 * self.reg * (np.sum(self.params['W'+str(num_layers)] *
                            self.params['W'+str(num_layers)]))
        # backprop the "first" layer of the NN -> use affine_backward
        # because it was the last layer to get gradients
        dx, dw, db = affine_backward(dscores, cache[num_layers])
        # and store them in the grads dictionary
        grads['W' + str(num_layers)] = dw + self.reg * self.params['W' + str(num_layers)]
        grads['b' + str(num_layers)] = db

        # now go the reverse direction -> the for loop counts down from last
        # to first layer
        for i in range(num_layers - 1, 0, -1):
            # if batchnorm is used we need to compute the backward pass
            # with affine_batchnorm_relu_forward
            # format is pretty ugly but there are so many variables :(
            if self.use_batchnorm:
                dx, dw,
                grads['b' + str(i)],
                grads['gamma' + str(i)],
                grads['beta' + str(i)] = affine_batchnorm_relu_backward(dx, cache[i])
            # if dropout is used we need to compute the backward pass
            # with dropout_backward, also use the dropout_cache!
            if self.use_dropout:
                cache[i], dropout_cache = cache[i]
                dx = dropout_backward(dx, dropout_cache)

            # no dropout and no batchnorm -> affine_relu_backward
            else:
                dx, dw, grads['b' + str(i)] = affine_relu_backward(dx, cache[i])

            # regularize the weights and store the gradients
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]

            # compute the loss and regularize it
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)] *
                                            self.params['W' + str(i)])
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################


        return loss, grads
