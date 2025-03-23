import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import train
import models

def main():
    """
    Main function to run the Streamlit application for Batch and Layer Normalization
    in handwritten digit recognition.

    The application has three modes:
    - About: Provides information about the project.
    - Model Training: Allows users to train models with different normalization techniques.
    - Trained Examples: Displays results from pre-trained models.
    """
    st.title("Batch and Layer Normalization for Handwritten Digit Recognition")
    app_mode = st.sidebar.selectbox(
        "Select mode",
        ["About", "Model Training", "Trained Examples"]
    )

    if app_mode == 'About':
        app_mode_about()
    elif app_mode == "Model Training":
        model_training()
    elif app_mode == "Trained Examples":
        trained_examples("20epochslr1")

def app_mode_about():
    """
    Displays information about the project and allows users to visualize MNIST images.

    This function provides an overview of the project's goals and allows users to
    generate and visualize examples of MNIST images.
    """
    st.markdown(
        """
        ### About the project

        In this project we implement Batch and Layer Normalization on a Multi-Layered Perceptron aiming
        to classify correctly handwritten digits. The images used for the training and testing are
        grayscale 28x28 images from the MNIST Dataset.
        The goal of this project is to show how both Batch and Layer Normalization can speed up training
        and improve models' performances.

        You can visualize examples of MNIST images by clicking on the button.
        """
    )
    if st.button("Generate MNIST images"):
        visualize_MNIST()

def visualize_MNIST():
    """
    Visualizes a grid of 9 random MNIST images.

    This function downloads the MNIST dataset, selects 9 random images, and displays them
    in a 3x3 grid using matplotlib.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    indices = torch.randperm(len(mnist_dataset))[:9]
    images = [mnist_dataset[i][0].squeeze() for i in indices]

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')

    plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.05, right=0.95, top=0.95, bottom=0.05)
    st.pyplot(fig)

def plot_results(dic):
    """
    Plots the training and validation results for different normalization techniques.

    Parameters
    ----------
    dic : dict
        A dictionary containing the results for different normalization techniques.
        The keys are "No Normalization", "Batch Normalization", and "Layer Normalization",
        and the values are lists of DataFrames containing the training and validation metrics.

    This function creates tabs for training loss, validation loss, validation accuracy,
    and activation distribution, and plots the corresponding metrics for each normalization technique.
    """
    if (dic["No Normalization"] == 0 and dic["Batch Normalization"] == 0 and
        dic["Layer Normalization"] == 0):
        st.write("Train a model and come back then!")
    else:
        if (dic["No Normalization"] != 0 or dic["Batch Normalization"] != 0):
            tab_trainloss, tab_testloss, tab_testacc, tab_quantiles = st.tabs(["Training Loss",
                                                                            "Validation Loss",
                                                                            "Validation Accuracy",
                                                                            "Activation Distribution"])
            with tab_quantiles:
                st.write("In this plot we can see the evolution of input distributions to a typical"
                        "sigmoid over the course of training, shown as 15th, 50th and 85th percentile.")
                for key in ["No Normalization", "Batch Normalization"]:
                    if dic[key] != 0:
                        df = dic[key][1]
                        fig, ax = plt.subplots()
                        ax.set_xlabel("Iterations")
                        ax.set_ylabel("")
                        ax.set_title(f'Model with {key.lower()}')
                        ax.plot(df["15th percentile"], color = "red")
                        ax.plot(df["50th percentile"], color = "red")
                        ax.plot(df["85th percentile"], color = "red")
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            tab_trainloss, tab_testloss, tab_testacc = st.tabs(["Training Loss",
                                                                "Test Loss",
                                                                "Test Accuracy"])

        with tab_trainloss:
            st.write("In this plot we can see the evolution of the training loss over the epochs.")
            fig, ax = plt.subplots()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss")
            ax.set_title("Training Loss over time")
            for key in dic.keys():
                if dic[key] != 0:
                    df = dic[key][0]
                    ax.plot([i for i in range(1,len(df["Training Loss"])+1)],df["Training Loss"], label = key)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        with tab_testloss:
            st.write("In this plot we can see the evolution of the validation loss over the epochs.")
            fig, ax = plt.subplots()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Loss")
            ax.set_title("Validation Loss over time")
            for key in dic.keys():
                if dic[key] != 0:
                    df = dic[key][0]
                    ax.plot([i for i in range(1,len(df["Training Loss"])+1)],df["Test Loss"], label = key)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        with tab_testacc:
            st.write("In this plot we can see the evolution of the validation accuracy loss over the epochs.")
            fig, ax = plt.subplots()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Accuracy")
            ax.set_title("Validation Accuracy over time")
            for key in dic.keys():
                if dic[key] != 0:
                    df = dic[key][0]
                    ax.plot([i for i in range(1,len(df["Training Loss"])+1)],df["Test Accuracy"], label = key)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

def model_training():
    """
    Allows users to select normalization techniques and hyperparameters, and trains the models.

    This function provides a user interface for selecting normalization techniques and
    hyperparameters, and trains the models using the selected configurations. The training
    progress is displayed using a progress bar.
    """
    user_choice, visualizations = st.tabs(["Model Selection", "Visualization"])

    with user_choice:
        options = st.multiselect(
            "Which methods do you wanna train and visualize?",
            ["No Normalization", "Batch Normalization", "Layer Normalization"],
            ["No Normalization"],
        )

        epochs_o = np.arange(5, 101, 5)
        batch_size_o = [2**i for i in range(0, 11)]
        lr_o = [10**(-i)*j for i in range(1, 6)for j in [5,2,1] ]
        st.write("Choose the desired hyperparameters.")
        epochs_s = st.select_slider("Epoch Number", options = epochs_o, value = 20)
        batch_size_s = st.select_slider("Batch Size", options = batch_size_o, value = 64)
        lr_s = st.select_slider("Learning Rate", options = lr_o, value = 1e-3)
        gpu = st.toggle("Check for GPU and use it", True)

        dic = {"No Normalization":0, "Batch Normalization":0, "Layer Normalization":0}

        if st.button("Train"):
            bar = st.progress(0, text = "Operation in progres... Model training might take a while.")
            pct = 0
            for option in options:
                batch, layer, q = False, False, True
                if option == "Batch Normalization":
                    batch = True
                elif option == "Layer Normalization":
                    layer, q = True, False
                model = models.Model_MNIST(batch, layer)
                dic[option] = train.df_transform(train.train(model, batch_size_s, lr_s, epochs_s,gpu, q))
                pct += np.float64(1/len(options))
                bar.progress(pct, text = "Operation in progres... Model training might take a while.")
            bar.empty()
    with visualizations:
        plot_results(dic)

def trained_examples(prefix):
    """
    Displays the results from pre-trained models.

    Parameters
    ----------
    prefix : str
        The prefix used to identify the pre-trained models' results.

    This function reads the results from pre-trained models stored in CSV files and
    displays them using the plot_results function.
    """
    model_no_norm_r = [pd.read_csv(f'dataframes/{prefix}/{prefix}_base_loss_results.csv'),
                        pd.read_csv(f'dataframes/{prefix}/{prefix}_base_quantiles_df.csv')]
    model_batch_r = [pd.read_csv(f'dataframes/{prefix}/{prefix}_batch_loss_results.csv'),
                        pd.read_csv(f'dataframes/{prefix}/{prefix}_batch_quantiles_df.csv')]
    model_layer_r = [pd.read_csv(f'dataframes/{prefix}/{prefix}_layer_loss_results.csv')]

    dic = {"No Normalization":model_no_norm_r, "Batch Normalization":model_batch_r,
            "Layer Normalization":model_layer_r}
    plot_results(dic)

if __name__ == "__main__":
    main()
