import os
import models
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd

def get_quantiles(x, ps=[15, 50, 85]):
    """
    Calculate the specified quantiles of the input data.

    Parameters
    ----------
    x : array-like
        Input data for which quantiles are to be calculated.
    ps : list of int, optional
        List of percentiles to compute, default is [15, 50, 85].

    Returns
    -------
    list of float
        The calculated quantiles.
    """
    return [np.percentile(x, p) for p in ps]

def test_loop(model, dataloader, device, criterion):
    """
    Evaluate the model on the test dataset.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    dataloader : DataLoader
        DataLoader for the test dataset.
    device : torch.device
        The device to run the model on (CPU or GPU).
    criterion : loss function
        The loss function to evaluate the model's performance.

    Returns
    -------
    tuple
        A tuple containing the average test loss and the accuracy.
    """
    model.eval()
    with torch.no_grad():
        test_loss, correct = 0, 0
        for X, y in dataloader:
            pred = model(X.to(device))
            loss = criterion(pred, y.to(device)).item()
            test_loss += loss
            prediction = pred.argmax(axis=1)
            correct += (prediction == y.to(device)).sum().item()
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    model.train()
    return test_loss, correct

def train(model_base, batch_size=60, learning_rate=1e-4, n_epochs=20, check_gpu=True, qu=True):
    """
    Train a neural network model on the MNIST dataset.

    Parameters
    ----------
    model_base : nn.Module
        The model to train.
    batch_size : int, optional
        The size of the training batches, default is 60.
    learning_rate : float, optional
        The learning rate for the optimizer, default is 1e-4.
    n_epochs : int, optional
        The number of epochs to train the model, default is 20.
    check_gpu : bool, optional
        Whether to check for GPU availability, default is True.
    qu : bool, optional
        Whether to calculate quantiles of activations, default is True.

    Returns
    -------
    tuple
        A tuple containing the training and test losses, test accuracies, and optionally quantiles.
    """
    device = torch.device('cpu')
    if check_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    MEAN_MNIST = (0.1307,)
    STD_MNIST = (0.3081,)

    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_MNIST, STD_MNIST)
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform_mnist)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform_mnist)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    train_losses_base = []
    test_losses_base = []
    test_scores_base = []
    quantiles_base = []

    model_base = model_base.to(device)

    optimizer_base = torch.optim.Adam(model_base.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        running_loss_base = 0.0
        for i, (x, y) in enumerate(trainloader):
            inputs, labels = x.to(device), y.to(device)

            optimizer_base.zero_grad()

            outputs_base = model_base(inputs)
            loss_base = criterion(outputs_base, labels)
            loss_base.backward()

            running_loss_base += loss_base.item()
            optimizer_base.step()

            if qu:
                quantiles_iter_base = get_quantiles(model_base.check.value.cpu().detach().numpy().flatten())
                quantiles_base.append(quantiles_iter_base)

        train_loss_base = running_loss_base / len(trainloader)
        train_losses_base.append(train_loss_base)

        test_loss_iter_base, score_iter_base = test_loop(model_base, testloader, device, criterion)
        test_losses_base.append(test_loss_iter_base)
        test_scores_base.append(score_iter_base)

    if qu:
        model_base_results = (train_losses_base, test_losses_base, test_scores_base, quantiles_base)
    else:
        model_base_results = (train_losses_base, test_losses_base, test_scores_base)

    return model_base_results

def df_transform(model_results, ps=[15, 50, 85]):
    """
    Transform model training results into DataFrames.

    Parameters
    ----------
    model_results : tuple
        A tuple containing training and test losses, test accuracies, and optionally quantiles.
    ps : list of int, optional
        List of percentiles to include in the quantiles DataFrame, default is [15, 50, 85].

    Returns
    -------
    list of DataFrame
        A list containing the results DataFrame and optionally the quantiles DataFrame.
    """
    dic = {"Epoch": np.array(range(1, len(model_results[0]) + 1)),
           "Training Loss": model_results[0],
           "Test Loss": model_results[1],
           "Test Accuracy": model_results[2]}

    results_df = pd.DataFrame(dic)

    if len(model_results) == 4:
        quantiles = np.array(model_results[3])
        dic = {"Iteration": np.array(range(1, len(quantiles) + 1))}
        for i, p in enumerate(ps):
            title_key = f'{p}th percentile'
            dic[title_key] = quantiles[:, i]
        quantiles_df = pd.DataFrame(dic)
        ret = [results_df, quantiles_df]
    else:
        ret = [results_df]
    return ret

def save(group_name, model_name, dfs):
    """
    Save the training results to CSV files.

    Parameters
    ----------
    group_name : str
        The name of the group or experiment.
    model_name : str
        The name of the model.
    dfs : list of DataFrame
        A list containing the results DataFrame and optionally the quantiles DataFrame.
    """
    os.makedirs(f'dataframes/{group_name}', exist_ok=True)
    dfs[0].to_csv(f'dataframes/{group_name}/{group_name}_{model_name}_loss_results.csv', index=False)
    if len(dfs) == 2:
        dfs[1].to_csv(f'dataframes/{group_name}/{group_name}_{model_name}_quantiles_df.csv', index=False)
