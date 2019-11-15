"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

# taken from https://deepnotes.io/softmax-crossentropy


def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    # print("W shape:" + str(W.shape))
    # print("X shape:" + str(X.shape))
    # print("y shape:" + str(y.shape))
    # print("dW shape:" + str(dW.shape))
    # print("reg:" + str(reg))
    # cross-entropy loss: sum i to n: sum k to k: y_ik * log(ŷ_ik)
    images = X.shape[0]
    classes = W.shape[1]

    for i in range(images):
        # Calculate the activation (ŷ) for every image. shape=(10, )
        activation = X[i].dot(W)

        # Regularization with softmax to normalize the activation (num stable)
        y_hat = stable_softmax(activation)

        # add up all losses; y stores the index of the correct class
        # because cross-entropy-loss has the -1 constant we use -=
        # y = 4 can be written in probabilitys as [0,0,0,1,0,0,0,0,0,0]
        # thats because the following assignment is still correct
        # with that the loss is (numerically stable) calculated
        loss -= np.log(y_hat[y[i]])

        # i dont know why this is needed but it fixes the rel error = 1
        # could be because the dW values else would be too big
        y_hat[y[i]] -= 1

        # ∂L/∂W=∂L/∂s⋅∂s/∂W
        # loss derivation wrt. weights is:
        # loss derivation wrt. score (= ground truth)
        # score derivation wrt. weights (= prediction)
        # works as follows:
        # dW is (3073, 10) X is (500, 3073) y_hat is (10, )
        # fill each column of dW (size 3073) with current image * prediction
        # of this column (class). Do that for every image
        # Final: every column/class in dW: sum of (all pictures * their
        # prediction for that class)
        for j in range(classes):
            dW[:, j] += X[i] * y_hat[j]

    # Compute the average
    loss /= images
    dW /= images

    # Adding the regularization reg
    # taken from http://cs231n.github.io/neural-networks-case-study/
    loss += reg * np.sum(W*W)
    dW += reg * W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    images = X.shape[0]
    activation = X.dot(W)

    # exps.shape = activation.shape = (500, 10)
    # because images are stored in rows -> axis=1 to find max in columns
    # dimensions need to be kept
    # taken from https://deepnotes.io/softmax-crossentropy
    exps = np.exp(activation - np.max(activation, axis=1, keepdims=True))
    exps /= np.sum(exps, axis=1, keepdims=True)

    # print(W.shape)
    # print(X.shape)
    # print(activation.shape)
    # print(exps.shape)

    # exps[range(images), y]: for all images pick the correct column with y
    # and take the average of the sum
    loss = np.sum(-np.log(exps[range(images), y])) / images

    # regularization
    # taken from http://cs231n.github.io/neural-networks-case-study/
    loss += reg * np.sum(W * W)

    # again to avoid exploding gradients i guess
    exps[range(images), y] -= 1

    # again ground truth * prediction
    # Shapes: X(500, 3073) * exps(500,10) -> need to transpose X
    dW = X.T.dot(exps) / images

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   #
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    # for learning rate
    #   for regularization
    #       softmax classifier
    #       train
    #       predict
    #       accuracy
    #       compare
    pass

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################

    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
