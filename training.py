import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TODO: Add early stopping when loss function converges, or val metrics stop improving
# TODO: Add circular learning rate decay
def train_single_model(model,
                       epochs,
                       X_train_batches,
                       y_train_batches,
                       X_val_batches,
                       y_val_batches,
                       criterion,
                       optimizer,
                       optimizer_args=None,
                       use_circular_lr=False,
                       circular_lr_tol_loss=1e-3,
                       circular_lr_tol_epoch=50,
                       device='cpu',
                       l1_factor=None,
                       l2_factor=None,
                       output_to_y_cb=None,
                       input_preprocess_cb=None,
                       plot_losses=True,
                       plot_metrics=True,
                       plot_rate = 2,
                       train_metrics_cb=None,
                       val_metrics_cb=None,
                       save_best_fit=False,
                       save_epoch_tolerance = 5,
                       saved_model_path=None):
    """
    :model: subclass of nn.Module
    :X_train_batches: sequence of tuples with the inputs to the `forward` function of the model during training
    :y_train_batches: batches of the known y values to be used during training
    :X_val_batches: sequence of tuples with the inputs to the `forward` function of the model during validation
    :y_val_batches: batches of the known y values to be used during validation
    :criterion: a criterion to be evaluated for each batch of data
    :optimizer: an optimizer class to use for training e.g. `torch.optim.Adam`
    :device: string, the specifier of the device. e.g. 'cpu' or 'cuda:3'
    :output_to_y_cb: callback to get the relevant tensor from model output
    :input_preprocess_cb: callback to preprocess the input data defaults to None.
    :plot_losses: bool make plot of losses
    :plot_metrics: bool make plot of metrics
    :plot_rate: int, number of epochs to wait before replotting results
    :train_metrics_cb: callback to calculate metrics from model output during training
    :val_metrics_cb: callback to calculate metrics from model output during validation
    :save_best_fit: bool. Make saved models from the best fitted model
    :save_epoch_tolerance: If the model has improved, but epochs since last save
    is less than save_epoch_tolerance, then do not save the model.
    :saved_model_path: full path to save the best model after training
    """

    if len(X_train_batches) != len(y_train_batches):
        raise ValueError("X_train_batches and y_train_batches must have the same length")
    
    # Do we need validation?
    validate = False
    if X_val_batches != None or y_val_batches != None:
        if X_val_batches==None or y_val_batches == None:
            raise ValueError("You must specify both X_val_batches and y_val_batches!")
        else:
            validate = True
            if len(X_val_batches) != len(y_val_batches):
              raise ValueError("X_val_batches and y_val_batches must have the same length")

    if optimizer==None:
        raise TypeError("You must specify a valid optimizer!")
    
    if criterion==None:
        raise TypeError("You must specify a valid criterion!")
    
    # Construct the optimizer
    if optimizer_args!=None:
        optimizer = optimizer(model.parameters(), **optimizer_args)
        optimizer.zero_grad()
    else:
        optimizer = optimizer(model.parameters())
        optimizer.zero_grad()

    # This will hold the training history
    history = dict()
    history['train_loss'] = []
    history['train_loss_std'] = []
    
    if validate:
        history['val_loss'] = []
        history['val_loss_std'] = []
    
    if train_metrics_cb:
        history['train_metrics'] = []
        history['train_metrics_std'] = []


    if validate and val_metrics_cb:
        history['val_metrics'] = []
        history['val_metrics_std'] = []

    # For plotting
    if plot_losses or plot_metrics:
        from IPython.display import clear_output
        from datetime import timedelta
        from matplotlib import pyplot as plt

    # Measure time
    from timeit import default_timer as timer
    train_start=timer()
    
    # Record the best loss and accuracy to save the model if it has improved
    # If given val_metric_cb, use only val_metric else use val_loss
    best_val_loss = None
    best_val_metric = None
    epochs_since_save = 0
    
    # Parameters of circular learning rate
    clr_alpha = 0.4 # must be a positive float, usually between 0 and 1
    clr_start = 1e-1
    clr_end = 1e-5
    clr_q = np.exp(clr_alpha*(np.log(clr_end)-np.log(clr_start))/epochs)

    # Iterate over epochs
    for epoch in range(epochs):

        # Set training mode
        model = model.train()

        # Set to the specified device
        model = model.to(device) 

        # Store training losses for this epoch
        epoch_train_losses = []
        
        # Store training metrics for this epoch
        if train_metrics_cb:
            epoch_train_metrics = []

        # Circular learning rate
        # TODO: This algorithm may need some improvements !
        # TODO: Is it good to use adaptive tolerance?
        if use_circular_lr==True and epoch >= circular_lr_tol_epoch:

            # If there is no significant improvement, reset optimizer
            N = len(history['train_loss'])
            firstavg = np.average(history['train_loss'][N-circular_lr_tol_epoch:N-3*(circular_lr_tol_epoch//4)])
            secondavg = np.average(history['train_loss'][N-(circular_lr_tol_epoch//4):])
            firststd = 0.5*np.std(history['train_loss'][N-circular_lr_tol_epoch:N-3*(circular_lr_tol_epoch//4)])
            secondstd = 0.5*np.std(history['train_loss'][N-(circular_lr_tol_epoch//4):])
            if np.abs((firstavg-firststd)-(secondavg+secondstd)) < circular_lr_tol_loss:
                # Now reset the learning rate to lr_start
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_start
            else:
                # Multiply the current lr with q
                for param_group in optimizer.param_groups:
                    param_group['lr'] = clr_q*param_group['lr']

        # Iterate over training batches
        for idx, ipt in enumerate(X_train_batches):
            if input_preprocess_cb:
                opt = model(input_preprocess_cb(*ipt).to(device))
            else:
                opt = model(*tuple(map(lambda t: t.to(device), ipt)))

            if output_to_y_cb:
                opt = output_to_y_cb(opt)
            
            loss = criterion(opt, y_train_batches[idx].to(device))

            # Add L1 regularization if needed
            if l1_factor is not None:
                l1_loss = None
                for W in model.parameters():
                    if l1_loss is None:
                        l1_loss = l1_factor*W.norm(1)
                    else:
                        l1_loss = l1_loss + l1_factor*W.norm(1)
                loss = loss + l1_loss

            # Add L2 regularization if needed
            if l2_factor is not None:
                l2_loss = None
                for W in model.parameters():
                    if l2_loss is None:
                        l2_loss = l2_factor*W.norm(2)
                    else:
                        l2_loss = l2_loss + l2_factor*W.norm(2)
                loss = loss + l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.detach().cpu().item())
            if train_metrics_cb:
                epoch_train_metrics.append(train_metrics_cb(opt.detach().cpu().numpy(), y_train_batches[idx]))
        
        # If validation is needed, iterate over validation batches
        if validate:
            # Set model in eval mode
            model = model.eval()
            
            # Set to the specified device
            model = model.to(device)

            # Store validation losses for this epoch
            epoch_val_losses = []
        
            # Store validation metrics for this epoch
            if val_metrics_cb:
                epoch_val_metrics = []
            
            for (idx, ipt) in enumerate(X_val_batches):
                if input_preprocess_cb:
                    opt = model(input_preprocess_cb(*ipt).to(device))
                else:
                    opt = model(*tuple(map(lambda t: t.to(device), ipt)))

                if output_to_y_cb:
                    opt = output_to_y_cb(opt)

                loss = criterion(opt, y_val_batches[idx].to(device))
                epoch_val_losses.append(loss.detach().cpu().item())
                if val_metrics_cb:
                    epoch_val_metrics.append(val_metrics_cb(opt.detach().cpu().numpy(), y_val_batches[idx]))
                    
        # Calculate losses and metrics for the current epoch
        history['train_loss'].append(np.mean(epoch_train_losses))
        history['train_loss_std'].append(np.std(epoch_train_losses))

        if train_metrics_cb:
            history['train_metrics'].append(np.mean(epoch_train_metrics))
            history['train_metrics_std'].append(np.std(epoch_train_metrics))

        if validate:
            history['val_loss'].append(np.mean(epoch_val_losses))
            history['val_loss_std'].append(np.std(epoch_val_losses))
            
            # Save model if necessary
            if save_best_fit:
                if best_val_loss==None:
                    best_val_loss = history['val_loss'][-1]
                else:
                    if history['val_loss'][-1] < best_val_loss:
                        best_val_loss = history['val_loss'][-1]
                        if not val_metrics_cb and epochs_since_save >= save_epoch_tolerance:
                            epochs_since_save = 0
                            # Now save the model
                            torch.save({
                                'epoch': epoch+1,
                                'name': model.__class__.__name__,
                                'model': model,
                                'model_state_dict': model.state_dict(),
                                'optimizer_class_name': optimizer.__class__.__name__,
                                'optimizer_state_dict': optimizer.state_dict()
                            }, saved_model_path)
            
            if val_metrics_cb:
                history['val_metrics'].append(np.mean(epoch_val_metrics))
                history['val_metrics_std'].append(np.std(epoch_val_metrics))
                
                if save_best_fit:
                    if best_val_metric==None:
                        best_val_metric = history['val_metrics'][-1]
                    else:
                        if history['val_metrics'][-1] > best_val_metric:
                            best_val_metric = history['val_metrics'][-1]
                            if epochs_since_save >= save_epoch_tolerance:
                                epochs_since_save = 0
                                # Now save the model
                                torch.save({
                                    'epoch': epoch+1,
                                    'name': model.__class__.__name__,
                                    'model': model,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_class_name': optimizer.__class__.__name__,
                                    'optimizer_state_dict': optimizer.state_dict()
                                }, saved_model_path)
        
        epochs_since_save = epochs_since_save + 1
        
        # TODO: Add time elapsed as a second x axis
        if plot_losses==True:
            if epoch%plot_rate == 0:
                clear_output(wait=True)
                if train_metrics_cb or val_metrics_cb:
                  fig, axs = plt.subplots(2, figsize=(8,10), dpi=112)
                  axs[0].grid(True)
                  axs[1].grid(True)
                else:
                  fig, ax = plt.subplots(1, figsize=(8,5), dpi=112)
                  axs = [ax]
                  axs[0].grid(True)
                fig.suptitle("Time elapsed: {}s epoch = {} \ncurrent lr = {}".format(
                    timedelta(seconds=round(timer()-train_start)),
                    epoch,
                    optimizer.state_dict()['param_groups'][0]['lr']
                    ))
                axs[0].set_xlabel("epoch")
                axs[0].set_ylabel("Loss")
                if train_metrics_cb or (validate and val_metrics_cb):
                    axs[1].set_xlabel("epoch")
                    axs[1].set_ylabel("Metric")
                    
                axs[0].plot(
                    np.arange(len(history['train_loss'])),
                    history['train_loss'], label='train_loss'
                )
                axs[0].fill_between(np.arange(len(history['train_loss'])),
                                    np.array(history['train_loss']) + np.array(history['train_loss_std']),
                                    np.array(history['train_loss']) - np.array(history['train_loss_std']),
                                    alpha=0.4)
                if train_metrics_cb:
                    axs[1].plot(
                        np.arange(len(history['train_metrics'])), history['train_metrics'],
                        label='train_'+train_metrics_cb.__name__
                    )
                    axs[1].fill_between(
                        np.arange(len(history['train_metrics'])),
                        np.array(history['train_metrics']) + np.array(history['train_metrics_std']),
                        np.array(history['train_metrics']) - np.array(history['train_metrics_std']),
                        alpha=0.4
                    )
                if validate:
                    axs[0].plot(
                        np.arange(len(history['val_loss'])),
                        history['val_loss'], label='val_loss'
                    )
                    axs[0].fill_between(
                        np.arange(len(history['val_loss'])),
                        np.array(history['val_loss']) + np.array(history['val_loss_std']),
                        np.array(history['val_loss']) - np.array(history['val_loss_std']),
                        alpha=0.4
                    )
                    if val_metrics_cb:
                        axs[1].plot(
                            np.arange(len(history['val_metrics'])),
                            history['val_metrics'],
                            label='val_'+val_metrics_cb.__name__
                        )
                        axs[1].fill_between(
                            np.arange(len(history['val_metrics'])),
                            np.array(history['val_metrics']) + np.array(history['val_metrics_std']),
                            np.array(history['val_metrics']) - np.array(history['val_metrics_std']),
                            alpha=0.4
                        )
                
                axs[0].legend(loc='bottom left')
                if train_metrics_cb or val_metrics_cb:
                    axs[1].legend(loc='bottom left')
                plt.show();
    return history, model