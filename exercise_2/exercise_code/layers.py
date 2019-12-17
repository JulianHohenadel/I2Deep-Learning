import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    """
    https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
    The criterion to satisfy for providing the new shape is that
    'The new shape should be compatible with the original shape'

    numpy allow us to give one of new shape parameter as -1
    (eg: (2,-1) or (-1,3) but not (-1, -1)). It simply means that it is
    an unknown dimension and we want numpy to figure it out.
    """
    # Reshaping x with x.shape[0] and -1
    # x has shape (2, 4, 5, 6), each x[i] has shape (4, 5, 6)
    # needed new shape: (2, 4*5*6)
    out = x.reshape(x.shape[0], -1)
    # standard x*w + b for foreward pass
    # print("out shape from affine" + str(out.shape))
    out = out.dot(w) + b

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    # Foreward = x*w + b

    # dx = upstream derivative * w
    # dx.shape is equal to x.shape thats why reshaping is needed
    dx = dout.dot(w.T).reshape(x.shape)
    # dw = upstream derivative * x
    # dw.shape = (N,D)^T.dot(N,M)
    dw = x.reshape((x.shape[0], -1)).T.dot(dout)
    # db = sum of all row entries of upstream derivative
    # db.shape collapses from (N, M) to (M, )
    db = dout.sum(axis=0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.maximum(0, x)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    """
    http://cs231n.github.io/neural-networks-case-study/#linear
    This turns out to be easy because ReLU during the backward pass is
    effectively a switch. Since r=max(0,x), we have that drdx=1(x>0).
    Combined with the chain rule, we see that the ReLU unit lets the
    gradient pass through unchanged if its input was greater than 0,
    but kills it if its input was less than zero during the forward pass.
    """
    dout[x <= 0] = 0
    dx = dout
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    -----------------------------------------------------------------
    The parameter μ is the mean or expectation of the distribution
    (and also its median and mode); and σ is its standard deviation.
    The variance of the distribution is σ^2.
    -----------------------------------------------------------------

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Look at the training-time forward pass implementation for batch     #
        # normalization.                                                            #
        # We use minibatch statistics to compute the mean and variance, use these   #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #############################################################################

        # calculate the mean of the data x (minibatch)
        sample_mean = np.mean(x, axis=0)

        # subtract the mean from the data x
        x_minus_mean = x - sample_mean

        # squared - for variance calculation
        sq = x_minus_mean ** 2

        # variance = 1/N * sum (x - mean)^2
        var = 1. / N * np.sum(sq, axis=0)

        # sqrt(var) (eps as numerical_stabilizer)
        sqrtvar = np.sqrt(var + eps)

        # variance inverse
        ivar = 1. / sqrtvar

        # normalized = x - mean * 1/variance
        # --> (x-m)/s
        x_norm = x_minus_mean * ivar

        # scale parameter gamma (*)
        gammax = gamma * x_norm

        # shift parameter beta (+)
        out = gammax + beta

        # adjust running variance and mean with momentum
        # take old running mean / var primarily and adjust slightly
        # with current mean / var
        running_var = momentum * running_var + (1 - momentum) * var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean

        # cache everything more or less
        cache = (out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps)

        #############################################################################
        #                             END OF CODE                                   #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Look at the test-time forward pass for batch normalization.         #
        #############################################################################

        # normalize x with the running_mean and running_var
        x = (x - running_mean) / np.sqrt(running_var)

        # also scale and shift with gamma and beta
        out = x * gamma + beta
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    """
    Awesome tutorial on batchnorm and batchnorm backprop
    https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    def batchnorm_forward(x, gamma, beta, eps):

        N, D = x.shape

        #step1: calculate mean
        mu = 1./N * np.sum(x, axis = 0)

        #step2: subtract mean vector of every trainings example
        xmu = x - mu

        #step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        #step4: calculate variance
        var = 1./N * np.sum(sq, axis = 0)

        #step5: add eps for numerical stability, then sqrt
        sqrtvar = np.sqrt(var + eps)

        #step6: invert sqrtwar
        ivar = 1./sqrtvar

        #step7: execute normalization
        xhat = xmu * ivar

        #step8: Nor the two transformation steps
        gammax = gamma * xhat

        #step9
        out = gammax + beta

        #store intermediate
        cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

    return out, cache

    def batchnorm_backward(dout, cache):

        #unfold the variables stored in cache
        xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

        #get the dimensions of the input/output
        N,D = dout.shape

        #step9
        dbeta = np.sum(dout, axis=0)
        dgammax = dout #not necessary, but more understandable

        #step8
        dgamma = np.sum(dgammax*xhat, axis=0)
        dxhat = dgammax * gamma

        #step7
        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar

        #step6
        dsqrtvar = -1. /(sqrtvar**2) * divar

        #step5
        dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

        #step4
        dsq = 1. /N * np.ones((N,D)) * dvar

        #step3
        dxmu2 = 2 * xmu * dsq

        #step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

        #step1
        dx2 = 1. /N * np.ones((N,D)) * dmu

        #step0
        dx = dx1 + dx2

    return dx, dgamma, dbeta

    """

    # unfold the variables stored in cache
    # cache = (out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps)
    out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps = cache

    # get the dimensions of the input/output
    N, D = dout.shape

    # step9
    dbeta = np.sum(dout, axis=0)
    dgammax = dout  # not necessary, but more understandable

    # step8
    dgamma = np.sum(dgammax*x_norm, axis=0)
    dxhat = dgammax * gamma

    # step7
    divar = np.sum(dxhat*x_minus_mean, axis=0)
    dxmu1 = dxhat * ivar

    # step6
    dsqrtvar = -1. / (sqrtvar**2) * divar

    # step5
    dvar = 0.5 * 1. / np.sqrt(var+eps) * dsqrtvar

    # step4
    dsq = 1. / N * np.ones((N, D)) * dvar

    # step3
    dxmu2 = 2 * x_minus_mean * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

    # step1
    dx2 = 1. / N * np.ones((N, D)) * dmu

    # step0
    dx = dx1 + dx2

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    """
    http://cthorey.github.io./backpropagation/
    https://github.com/cthorey/CS231
    mu = 1./N*np.sum(h, axis = 0)

    var = 1./N*np.sum((h-mu)**2, axis = 0)

    dbeta = np.sum(dy, axis=0)

    dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)

    dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) *
    (N * dy - np.sum(dy, axis=0) - (h - mu) * (var + eps)**(-1.0) *
    np.sum(dy * (h - mu), axis=0))
    """

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None
    """
    http://cs231n.github.io/neural-networks-2/

    """

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################

        # Create a mask with has the same dimensions as x
        # assign mask random values (with seed if given)
        mask = np.random.random(x.shape)

        # make the mask binary: 0 if below p (dropped), 1 if above p (kept)
        mask = np.where(mask < p, 0, 1)
        # mask[mask < p] = 0
        # mask[mask >= p] = 1

        # multiply the input data with the mask, this is the output
        out = x * mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        ###########################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.       #
        ###########################################################################

        # this is just the identity
        out = x
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        """
        https://wiseodd.github.io/techblog/2016/06/25/dropout/
        During the backprop, what we need to do is just to consider the Dropout.
        The killed neurons don’t contribute anything to the network,
        so we won’t flow the gradient through them.
        """
        # dout are the upstream derivarives
        # set them to 0 where neurons have been dropped
        dx = dout * mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    numerical_stabilizer = 1e-10
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y] + numerical_stabilizer)) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def l2_loss(x, y):
    """
    Computes the loss and gradient for regression using the l2 norm.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Real data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss = np.sqrt(np.sum((x-y)**2))
    dx = (x-y)/loss
    return loss, dx
