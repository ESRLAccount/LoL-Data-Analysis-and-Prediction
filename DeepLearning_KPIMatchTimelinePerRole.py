import numpy as np
import pandas as pd
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras import regularizers
from keras import layers
import os

plt.rcParams["figure.figsize"] = (8, 10)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_name(num):
    if num == 1:
        return "Earlygame"
    elif num == 2:
        return "Midgame"
    elif num == 3:
        return "Lategame"

def plot_importance(importances):
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


##########################
class CancelOut(keras.layers.Layer):
    '''
    CancelOut layer, keras implementation.
    '''

    def __init__(self, activation='sigmoid', cancelout_loss=True, lambda_1=0.002, lambda_2=0.001):
        super(CancelOut, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cancelout_loss = cancelout_loss

        if activation == 'sigmoid': self.activation = tf.sigmoid
        if activation == 'softmax': self.activation = tf.nn.softmax

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(1),
            trainable=True,
        )

    def call(self, inputs):
        if self.cancelout_loss:
            self.add_loss(self.lambda_1 * tf.norm(self.w, ord=1) + self.lambda_2 * tf.norm(self.w, ord=2))
        return tf.math.multiply(inputs, self.activation(self.w))

    def get_config(self):
        return {"activation": self.activation}


##################
class VarImpVIANN(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.n = 0
        self.M2 = 0.0

    def on_train_begin(self, logs={}, verbose=1):
        if self.verbose:
            print("VIANN version 1.0 (Wellford + Mean) update per epoch")
        self.diff = self.model.layers[0].get_weights()[0]

    def on_epoch_end(self, batch, logs={}):
        currentWeights = self.model.layers[0].get_weights()[0]

        self.n += 1
        delta = np.subtract(currentWeights, self.diff)
        self.diff += delta / self.n
        delta2 = np.subtract(currentWeights, self.diff)
        self.M2 += delta * delta2

        self.lastweights = self.model.layers[0].get_weights()[0]

    def on_train_end(self, batch, logs={}):
        if self.n < 2:
            self.s2 = float('nan')
        else:
            self.s2 = self.M2 / (self.n - 1)

        scores = np.sum(np.multiply(self.s2, np.abs(self.lastweights)), axis=1)

        self.varScores = (scores - min(scores)) / (max(scores) - min(scores))
        if self.verbose:
            print("Most important variables: ",
                  np.array(self.varScores).argsort()[-10:][::-1])


def add_noise(X):
    # Specify the standard deviation of the Gaussian noise
    noise_std_dev = 0.1  # Adjust this value based on your preference

    # Generate random noise from a Gaussian distribution centered at zero
    noise = np.random.normal(0, noise_std_dev, X.shape)

    # Add the generated noise to the original DataFrame
    X = X + noise
    return X


def correlation_analysis(df):
    # Set correlation threshold (adjust as needed)
    correlation_threshold = 0.8

    # Perform Pearson correlation analysis
    correlation_matrix = df.corr()

    # Create a mask for highly correlated features
    mask = np.triu(np.ones(correlation_matrix.shape), k=1)
    highly_correlated = np.where(np.abs(correlation_matrix) > correlation_threshold, mask, 0)

    # Find indices of features to be removed
    indices_to_remove = np.unique(np.where(highly_correlated != 0)[1])

    # Remove highly correlated features
    df_filtered = df.drop(df.columns[indices_to_remove], axis=1)
    return (df_filtered)


def get_lowVarienceCols(X):
    # Specify the variance threshold
    variance_threshold = 0.01  # You can adjust this threshold based on your preference

    variances = X.var()
    high_variance_columns = variances[variances > variance_threshold].index

    # high_varcols=X.drop(columns=low_variance_columns,axis=1)
    return (high_variance_columns)


def plot_trainingAUC(history, y_test, y_prob):
    # Plot training history
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Show plots
    plt.tight_layout()
    plt.show()

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_ModelPerfromanceMetrics(X_train, X_test, y_train, y_test, model):
    # Predict on training and testing sets
    y_train_pred = np.round(model.predict(X_train))
    y_test_pred = np.round(model.predict(X_test))

    # Calculate evaluation metrics
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)

    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)

    # Create a bar plot
    labels = ['Precision', 'Recall', 'F1 Score']
    train_metrics = [precision_train, recall_train, f1_train]
    test_metrics = [precision_test, recall_test, f1_test]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, train_metrics, width, label='Training')
    rects2 = ax.bar(x + width / 2, test_metrics, width, label='Testing')

    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Display the scores on top of the bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()


def create_modelArt(model, input_size):
    # Assuming `model` is your PyTorch model
    # Input tensor example
    x = torch.randn(1, input_size)

    # Generate a computational graph
    y = model(x)
    graph = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

    # Save the graph as an image or display it
    graph.render("model_graph", format="png", cleanup=True)


def Normalize_perGameduration(df,fieldname):
    # Define the column names to exclude from normalization
    exclude_cols = ['win']  # Add any column names you want to exclude

    df = df.apply(pd.to_numeric, errors='coerce')

    # df['gameDuration'] = pd.to_numeric(df['gameDuration'], errors='coerce')

    # Get the list of columns to normalize (all columns except exclude_cols)
    normalize_cols = df.columns.difference(exclude_cols)
    normalized_df = df.copy()

    # Normalize all columns based on the game duration
    normalized_df[normalize_cols] = normalized_df[normalize_cols].div(normalized_df[fieldname], axis=0)

    return (normalized_df)

def make_deep_learning_captum(df, feature_names):
    from captum.attr import IntegratedGradients
    from captum.attr import LayerConductance
    from captum.attr import NeuronConductance


    from scipy import stats


    labels = df['win'].to_numpy()
    df = df.drop(['win'], axis=1)

    data = df.to_numpy()

    train_indices = np.random.choice(len(labels), int(0.7*len(labels)), replace=False)
    test_indices = list(set(range(len(labels))) - set(train_indices))
    train_features = data[train_indices]
    train_labels = labels[train_indices]
    test_features = data[test_indices]
    test_labels = labels[test_indices]

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, x, y):
            super().__init__()
            self.x = x
            self.y = y

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

        def __len__(self):
            return len(self.x)

    # define the train and test dataloader
    train_loader = torch.utils.data.DataLoader(Dataset(train_features, train_labels))
    test_loader = torch.utils.data.DataLoader(Dataset(test_features, test_labels))

    torch.manual_seed(1)

    # code a neural network with the nn module imported into the class
    class NN_Model(nn.Module):
        def __init__(self, input_size, hidden_size1,hidden_size2,output_size, dropout_rate, l2_penalty):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size1)
            self.sigmoid1 = nn.Sigmoid()
            self.linear2 = nn.Linear(hidden_size1, hidden_size2)
            self.sigmoid2 = nn.Sigmoid()
            self.linear3 = nn.Linear(hidden_size2, output_size)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            lin1_out = self.linear1(x)
            sigmoid1_out = self.sigmoid1(lin1_out)
            lin2_out = self.linear2(sigmoid1_out)
            sigmoid2_out = self.sigmoid2(lin2_out)
            lin3_out = self.linear3(sigmoid2_out)
            softmax_out = self.softmax(lin3_out)
            return softmax_out

            # Model parameters
    input_size = data.shape[1]
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 2
    dropout_rate = 0.5
    l2_penalty = 1e-5



    model = NN_Model(input_size, hidden_size1, hidden_size2, output_size, dropout_rate, l2_penalty)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    total_loss, total_acc = list(), list()
    feat_imp = np.zeros(train_features.shape[1])
    num_epochs = 200

    for epoch in range(num_epochs):
        losses = 0
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.float(), y.type(torch.LongTensor)
            x.requires_grad = True
            optimizer.zero_grad()
            # check if the progrma can be run with model(x) and model.forward()
            preds = model.forward(x)
            loss = criterion(preds, y)
            x.requires_grad = False
            loss.backward()
            optimizer.step()
            losses += loss.item()
        total_loss.append(losses / len(train_loader))
        if epoch % 10 == 0:
            print("Epoch:", str(epoch + 1), "\tLoss:", total_loss[-1])

    model.eval()
    correct=0
    for idx, (x,y) in enumerate(test_loader):
        with torch.no_grad():
            x,y = x.float(), y.type(torch.LongTensor)
            pred = model(x)
            preds_class = torch.argmax(pred)
            if (preds_class.numpy()== y.numpy()[0]):
                correct+=1
    print("Accuracy = ", correct/len(test_indices))

def make_deep_learning(df, feature_names,outputdata_path):

    # Convert the DataFrame column to a Python array
    #feature_names = df_cols.iloc[:, 0].tolist()

    model_Type = '4'
    # Initialize the MinMaxScaler
    # scaler = MinMaxScaler()

    #df = df.select_dtypes(include='number')
    df=df[feature_names]

    X = df.drop(['win','Phase'], axis=1)

    # Calculate the threshold for exclusion (60% of data being zero)
    # threshold = 0.6 * len(df)

    # Exclude columns where more than 60% of the data is zero
    # X = X.drop(columns=X.columns[(X == 0).sum() > threshold])

    y = df['win']

    #X=add_noise(X)

    featurelist_forML = get_lowVarienceCols(X)

    X = X[featurelist_forML]
    ##corlation analysis
    X = correlation_analysis(X)

    #X.fillna(0.0000001, inplace=True)

    # Normalize the data
    #scaler = MinMaxScaler()
    # scaler = StandardScaler()
    #X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Combine the normalized features with the target variable
    df_normalized = pd.DataFrame(X, columns=X.columns)
    df_normalized['win'] = y



    if model_Type == '4':  ###pytorch
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Assuming y is a pandas Series
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        ##making a copy for using in shap
        X_train_tensor_forshap = X_train_tensor
        X_test_tensor_forshap = X_test_tensor

        X_train_tensor = X_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)

        # Define a simple neural network with regularization
        class RegularizedModel(nn.Module):
            def __init__(self, input_size, hidden_size1,hidden_size2,output_size, dropout_rate, l2_penalty):
                super(RegularizedModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size1)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size1, hidden_size2)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()
                self.fc3 = nn.Linear(hidden_size2, output_size)
                self.l2_penalty = l2_penalty
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.dropout(x)
                x = self.relu(x)
                x = self.fc3(x)

                return x
            # Define a function that takes a NumPy array as input and returns PyTorch model predictions

        def predict(input_data):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            model_output = model(input_tensor)
            return model_output.detach().numpy()

        # Model parameters
        input_size = X_train.shape[1]
        hidden_size1 = 128
        hidden_size2 = 64
        output_size = 1
        dropout_rate = 0.5
        l2_penalty = 1e-5

        # Create the model, loss function, and optimizer
        model = RegularizedModel(input_size, hidden_size1,hidden_size2,output_size, dropout_rate, l2_penalty)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=l2_penalty)

        # Training the model
        # Training parameters
        num_epochs = 200
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Training accuracy
            with torch.no_grad():
                y_train_pred = (torch.sigmoid(outputs) >= 0.5).float()
                train_accuracy = torch.sum(y_train_pred == y_train_tensor).item() / len(y_train)
                train_accuracies.append(train_accuracy)

            # Validation
            model.eval()

            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                val_losses.append(val_loss.item())

                # Validation accuracy
                y_val_pred = (torch.sigmoid(val_outputs) >= 0.5).float()
                val_accuracy = torch.sum(y_val_pred == y_test_tensor).item() / len(y_test)
                val_accuracies.append(val_accuracy)

            # Print epoch statistics
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Training Loss: {loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Plot training loss and accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

        """

        # select a set of background examples to take an expectation over
        #background = X_train_tensor[np.random.choice(X_train_tensor.shape[0], 1000, replace=False)]

        # explain predictions of the model on four images
        e = shap.DeepExplainer(model, background)
        # ...or pass tensors directly
        # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
        shap_values = e.shap_values(X_test_tensor[1:1000])

        #shap.summary_plot(shap_values, X_test_tensor[1:1000], feature_names=X.columns, max_display=17)


        mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
        top_feature_indices = np.argsort(mean_abs_shap_values)[-10:][::-1]
        top_features = X.columns[top_feature_indices]
        top_scores = mean_abs_shap_values[top_feature_indices]

        # Create a DataFrame with the top features and their scores
        df_top_features = pd.DataFrame({'Feature': top_features, 'SHAP Score': top_scores})

        #'with open(outputdata_path, 'a') as f:
            #f.write(f'Top 10 KPIs in {subject} ')  # Write title
        df_top_features.to_csv(outputdata_path, index=False)

        # shap.plots.beeswarm(shap_values, max_display=20)

        # shap.plots.bar(shap_values)

        plt.show()

        # plot the feature attributions
        # shap.image_plot(shap_values, -X_test_tensor[1:5])

        ############################################################################
        # SHAP PLOTS
        # Create an explainer
  
        explainer = shap.Explainer(predict, X_train_tensor_forshap.numpy())

        # Calculate SHAP values for the test set
        shap_values = explainer.shap_values(X_test_tensor_forshap.numpy())

        # Create a beeswarm plot
        shap.summary_plot(shap_values, X_test_tensor_forshap, feature_names=X.columns, auto_size_plot=True)

        shap.summary_plot(shap_values, X_test_tensor_forshap, feature_names=X.columns, auto_size_plot=True,
                          plot_type="bar")
        # shap.plots.bar(shap_values,max_display=15)
        plt.show()
        """
        ###########################################################################

        feature_importances = model.fc1.weight.detach().numpy()[0]

        # Get the indices of the top 10 most important features
        top10_indices = np.argsort(np.abs(feature_importances))[:]

        # Print the names of the top 10 most important features
        top10_feature_names = X.columns[top10_indices]
        print("Top 10 Most Important Features:")
        print(top10_feature_names)


        # Create a dictionary from the lists
        data = {'Feature': top10_feature_names, 'SHAP Score': abs(feature_importances[top10_indices])}

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data)
        df = df.sort_values(by='Feature')
        df.to_csv(outputdata_path, index=False)

        # Plot bar chart for feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(top10_feature_names, feature_importances[top10_indices], color='#087E8B')

        # plt.bar(range(1, 11), feature_importances[top10_indices], align="center")
        # plt.xticks(range(1, 11), top10_feature_names, rotation=45, ha="right")
        plt.xlabel("Feature")
        plt.ylabel("Importance Score")
        plt.title("Top 15 Most Important Features")
        plt.show()

    if model_Type == '5':
        # Assume X_train, y_train are your training data
        X_train, X_test, y_train, y_test = train_test_split(df.drop('win', axis=1), df['win'], test_size=0.2,
                                                            random_state=42)

        # Standardize the data (important for some deep learning models)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define the deep learning model with two hidden layers
        class DeepLearningModel(nn.Module):
            def __init__(self, input_size, hidden_size1, hidden_size2):
                super(DeepLearningModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size1)
                self.fc2 = nn.Linear(hidden_size1, hidden_size2)
                self.fc3 = nn.Linear(hidden_size2, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        # Initialize the model
        input_size = X_train.shape[1]
        hidden_size1 = 64  # Number of units in the first hidden layer
        hidden_size2 = 64  # Number of units in the second hidden layer
        model = DeepLearningModel(input_size, hidden_size1, hidden_size2)

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        # Train the model
        num_epochs = 10
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Convert the model back to evaluation mode
        model.eval()

        # Example of making predictions
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        predictions = model(X_test_tensor).detach().numpy().flatten()
        # select a set of background examples to take an expectation over

        background = X_train_tensor[np.random.choice(X_train_tensor.shape[0], 1000, replace=False)]

        # explain predictions of the model on four images
        e = shap.DeepExplainer(model, background)
        # ...or pass tensors directly
        # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
        shap_values = e.shap_values(X_test_tensor[1:1000])

        shap.summary_plot(shap_values, X_test_tensor[1:1000], feature_names=X.columns, max_display=17)

        # shap.plots.beeswarm(shap_values, max_display=20)

        # shap.plots.bar(shap_values)

        plt.show()

#######################
def DeepAnalysis_Matchchallenges_PerRole(Inputdata_path1):
        df = pd.read_csv(Inputdata_path1)
        print(len(df))
        df.to_csv(outpdata_path2)
        df['win'] = df['win'].astype(int)
        df['role'] = df['role'].replace('NONE', 'JUNGLE')
        df['lane'] = df['lane'].replace('NONE', 'JUNGLE')

        for role in Role_list:
            df_role = df[df['role'] == role]
            print(role, len(df_role))
            filtered_df = Normalize_perGameduration(df_role, 'challenges_gameLength')
            make_deep_learning(df_role, df_cols)

#######################

def DeepAnalysis_MatchTimelinePerRole(Inputdata_path2,outpdata_path):
    df = pd.read_csv(Inputdata_path2)
    print(len(df))

    df_cols = df.select_dtypes(include='number').columns

    df_cols = [x for x in df_cols if x not in excludedcols]

    df['win'] = df['win'].astype(int)
    df['role'] = df['role'].replace('NONE', 'JUNGLE')
    #df['lane'] = df['lane'].replace('NONE', 'JUNGLE')
   ###finding kpi for all role
    output_file = outpdata_path + '_' + 'all' + '.csv'
    make_deep_learning(df, df_cols, output_file)


    for role in Role_list:
        df_role = df[df['role'] == role]
        print(role, len(df_role))
        filtered_df = Normalize_perGameduration(df_role,'gameDuration')
        output_file=outpdata_path+'_'+role+'.csv'
        #make_deep_learning_captum  (filtered_df, df_cols)
        make_deep_learning(df, df_cols,output_file)

#######################
def DeepAnalysis_MatchTimelinePerRolePhase(Inputdata_path2):
    df = pd.read_csv(Inputdata_path2)
    print(len(df))

    df_cols = df.select_dtypes(include='number').columns

    df_cols = [x for x in df_cols if x not in excludedcols]

    df['win'] = df['win'].astype(int)
    df['role'] = df['role'].replace('NONE', 'JUNGLE')
    df['lane'] = df['lane'].replace('NONE', 'JUNGLE')

    for role in Role_list:
        df_role = df[df['role'] == role]
        print(role, len(df_role))
        filtered_df = Normalize_perGameduration(df_role,'gameDuration')
        make_deep_learning(filtered_df, df_cols)

#######################
def DeepAnalysis_MatchTimelinePerPhase(Inputdata_path2,outpdata_path):
    df = pd.read_csv(Inputdata_path2)
    print(len(df))
    df = df.dropna(subset=['win'])
    print(len(df),'after removing null')
    df['win'] = df['win'].astype(int)

    df_cols = df.select_dtypes(include='number').columns
    df_cols = [x for x in df_cols if x not in excludedcols]
    #filtered_df = Normalize_perGameduration(df_phase, 'gameDuration')

    output_file = outpdata_path + '_' + 'all' + '.csv'
    make_deep_learning(df, df_cols, output_file)


    for phase in phase_list:
        df_phase = df[df['Phase'] == phase]
        print(phase, len(df_phase))
        #filtered_df = Normalize_perGameduration(df_phase,'gameDuration')

        output_file = outpdata_path + '_' + get_name(phase) + '.csv'
        make_deep_learning(df_phase, df_cols,output_file)

####Read dataset
Inputdata_path1 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchPhase.csv"

Inputdata_path2 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchRole.csv"

outpdata_path2 = "data/Output/MasterFile/MatchChallenge_PerRole_sample100.csv"

outputdata_path1 = "data/OutputRank/FeatureSelection/KPI_MatchTimelinePerRole"

outputdata_path2 = "data/OutputRank/FeatureSelection/KPI_MatchTimelinePerPhase"

excludedcols=['gameDuration','position_x','position_y','participantId','time','Unnamed: 0','team_id']
###using cpu instead of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
Role_list=['SOLO','JUNGLE','CARRY','SUPPORT','DUO']
phase_list=[1,2,3]

inputdata_pathforSelectedCols = "data/Input/MatchResume_FinalCols.csv"
df_cols = pd.read_csv(inputdata_pathforSelectedCols)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

##ROLE Specific analysis in matchchallenge master file
#DeepAnalysis_Matchchallenges_PerRole(Inputdata_path1)

##Phase Specific analysis in matchchallenge master file
DeepAnalysis_MatchTimelinePerPhase(Inputdata_path1,outputdata_path2)

##ROLE Specific analysis in MatchTimeline master file
#DeepAnalysis_MatchTimelinePerRole(Inputdata_path2,outputdata_path1)

##ROLE/Phase Specific analysis in MatchTimeline master file
#DeepAnalysis_MatchTimelinePerRolePhase(Inputdata_path2)