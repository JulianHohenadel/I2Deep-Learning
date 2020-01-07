
import numpy as np
import torch


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):

        # Resets train and val histories for the accuracy and the loss.

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def run_epoch(self, model, optim, dataloader, train, epoch, iterations, log_nth):

        model.train()
        device = next(model.parameters()).device
        val_loss = 0.0
        val_acc = 0.0

        with torch.set_grad_enabled(train):
            for iteration, sample in enumerate(dataloader):
                if iteration >= len(dataloader) and train:
                    break

                if train:
                    optim.zero_grad()

                batch = sample[0].detach().to(device)
                labels = sample[1].detach().to(device)
                output = model(batch)

                if train:
                    train_loss = self.loss_func(output, labels)
                    train_loss.backward()
                    optim.step()
                    iter_per_epoch = len(dataloader)
                    current_iteration = iteration + epoch * iter_per_epoch + 1
                    if current_iteration % log_nth == 0:
                        print(
                            f'[Iteration {current_iteration}/{iterations}] TRAIN loss: {train_loss}')
                        self.train_loss_history.append(train_loss.detach())
                else:
                    maxs, predict = torch.max(output, 1)
                    val_loss += self.loss_func(output, labels)
                    val_acc += float((labels == predict).sum()) / len(predict)

            if train:
                maxs, predict = torch.max(output, 1)
                train_acc = float((labels == predict).sum()) / len(predict)
                return train_loss, train_acc
            else:
                val_loss /= len(dataloader)
                val_acc /= len(dataloader)
                return val_loss, val_acc

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        #######################################################################
        # TODO:                                                               #
        # Write your own personal training method for our solver. In each     #
        # epoch iter_per_epoch shuffled training batches are processed. The   #
        # loss for each batch is stored in self.train_loss_history. Every     #
        # log_nth iteration the loss is logged. After one epoch the training  #
        # accuracy of the last mini batch is logged and stored in             #
        # self.train_acc_history. We validate at the end of each epoch, log   #
        # the result and store the accuracy of the entire validation set in   #
        # self.val_acc_history.                                               #
        #                                                                     #
        # Your logging could like something like:                             #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                           #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                           #
        #   ...                                                               #
        #######################################################################

        # iteration counter
        iterations = num_epochs * iter_per_epoch

        # Store:
        #     loss:
        #         train: each batch
        #         val:   after each epoch
        #     acc:
        #         train: after each epoch train acc of last mini batch store
        #         val:   after each epoch

        # Logging:
        #     loss:
        #         train: every log_nth iteration
        #         val:   after each epoch
        #     acc:
        #         train: after each epoch train acc of last mini batch store
        #         val:   after each epoch

        for epoch in range(num_epochs):
            print('***********************************************************')

            # first training:
            train_loss, train_acc = self.run_epoch(model, optim, train_loader,
                                                   True, epoch, iterations, log_nth)

            # train_loss is logged in run_epoch
            # train_acc is stored after each epoch
            self.train_acc_history.append(train_acc)

            # then validation:
            val_loss, val_acc = self.run_epoch(model, optim, val_loader,
                                               False, epoch, iterations, log_nth)

            # val_acc is stored after each epoch
            self.val_acc_history.append(val_acc)

            # padding if consistency with train loss is wanted
            # for i in range(int(iter_per_epoch / log_nth) - 1):
            #     self.val_loss_history.append(None)

            self.val_loss_history.append(val_loss)

            print(f'[Epoch {epoch + 1}/{num_epochs}] TRAIN\tacc/loss: {train_acc} / {train_loss}')
            print(f'[Epoch {epoch + 1}/{num_epochs}] VAL\tacc/loss: {val_acc} / {val_loss}')

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
