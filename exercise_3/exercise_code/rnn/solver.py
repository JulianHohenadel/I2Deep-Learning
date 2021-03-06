import numpy as np
import torch


class Solver(object):
    default_adam_args = {"lr": 1e-2,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), 
                 stop_limit=100, stop_early=False):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self.best_val_acc = 0
        self.best_model = None
        self.early_stop_limit = stop_limit
        self.early_stop_counter = 0
        self.stop_early = stop_early

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

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
        if self.stop_early:
            print("Early stopping: activated.")
            print("Limit: " + str(self.early_stop_limit))
        else:
            print("Early stopping deactivated.")
        print('START TRAIN.')

        for epoch in range(num_epochs):
            # TRAINING
            model.train()
            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = inputs.to(device), targets.to(device)

                optim.zero_grad()

                outputs = model(inputs.squeeze(1).permute(1, 0, 2).float())
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.detach().cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' %
                          (i + epoch * iter_per_epoch,
                           iter_per_epoch * num_epochs,
                           train_loss))

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with label >= 0 e.g. for segmentation
            targets_mask = targets >= 0
            train_acc = np.mean((preds == targets)[
                                targets_mask].detach().cpu().numpy())
            self.train_acc_history.append(train_acc)
            #if log_nth:
            #    print('[Epoch %d/%d] TRAIN acc/loss: %.2f/%.3f' % (epoch + 1,
            #                                                       num_epochs,
            #                                                       train_acc*100,
            #                                                       train_loss))
            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model.forward(inputs.squeeze(1).permute(1, 0, 2).float())
                loss = self.loss_func(outputs, targets)
                val_losses.append(loss.detach().cpu().numpy())

                _, preds = torch.max(outputs, 1)

                # Only allow images/pixels with target >= 0 e.g. for segmentation
                targets_mask = targets >= 0
                scores = np.mean((preds == targets)[
                                 targets_mask].detach().cpu().numpy())
                val_scores.append(scores)

            model.train()
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.2f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc*100,
                                                                   val_loss))
                if self.stop_early:
                    print("Current patience: " +
                            str(self.early_stop_counter + 1))
            if val_acc > self.best_val_acc:
                print("New best validition accuracy: " + str(val_acc*100)[:5])
                self.best_val_acc = val_acc
                self.best_model = model
                if self.stop_early:
                    self.early_stop_counter = -1
            if self.stop_early:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_limit:
                    print("Early stopping.")
                    print("No improvement for " + 
                            str(self.early_stop_limit) + " epochs")
                    break
        print("Best VAL acc: " + str(self.best_val_acc*100)[:5])
        print('FINISH.')
        return self.best_model
