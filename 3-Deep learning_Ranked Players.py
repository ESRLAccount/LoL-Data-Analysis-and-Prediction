
#### Indentifying kpi for players based on their rank
import numpy as np
import pandas as pd
import shap

import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_squared_error, r2_score

from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot

import tensorflow as tf


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras import regularizers
from keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import os
from sklearn.preprocessing import LabelEncoder

plt.rcParams["figure.figsize"] = (8, 10)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def plot_importance(importances):
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

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
    variance_threshold = 0.05  # You can adjust this threshold based on your preference

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


def Normalize_perGameduration(df):
    # Define the column names to exclude from normalization
    exclude_cols = ['win']  # Add any column names you want to exclude

    df = df.apply(pd.to_numeric, errors='coerce')

    # df['gameDuration'] = pd.to_numeric(df['gameDuration'], errors='coerce')

    # Get the list of columns to normalize (all columns except exclude_cols)
    normalize_cols = df.columns.difference(exclude_cols)
    normalized_df = df.copy()

    # Normalize all columns based on the game duration
    normalized_df[normalize_cols] = normalized_df[normalize_cols].div(normalized_df['gameDuration'], axis=0)

    return (normalized_df)

def Normalize_perGameduration (df):
    # Define the column names to exclude from normalization
    exclude_cols = ['tier']  # Add any column names you want to exclude

    df = df.apply(pd.to_numeric, errors='coerce')

    #df['gameDuration'] = pd.to_numeric(df['gameDuration'], errors='coerce')

    # Get the list of columns to normalize (all columns except exclude_cols)
    normalize_cols = df.columns.difference(exclude_cols)
    normalized_df = df.copy()

    # Normalize all columns based on the game duration
    normalized_df[normalize_cols] = normalized_df[normalize_cols].div(normalized_df['gameDuration'], axis=0)

    return(normalized_df)

def make_deep_learning(df, df_cols,outputdata_path):
    # Convert the DataFrame column to a Python array
    feature_names = df_cols.iloc[:, 0].tolist()

    model_Type = '4'
    df=df[feature_names]
    df['tierRank']=df['tier']# + '_' + df['rank']



###########################select partial data related to biggest groups
    grouped = df.groupby('tierRank').size().reset_index(name='Count')
    print(grouped)
    selected_rank = grouped[grouped['Count'] > 20000]

    Rank_names = selected_rank.iloc[:, 0].tolist()
    #print('Rank_names', Rank_names)
    df = df[df['tierRank'].isin(Rank_names)]

    num_categories = len(df['tierRank'].unique())
    print(num_categories, 'num_categories')
########################################

    # Initialize LabelEncoder
    le = LabelEncoder()
    #le=OneHotEncoder()

    # Encode the categorical target variable
    df['tierRank'] = le.fit_transform(df['tierRank'])

    # Split the data into features and target
    y = df['tierRank']
    df = df.select_dtypes(include='number')
    # df=df[feature_names]

    #df = Normalize_perGameduration(df)


    X = df.drop(['tierRank'],axis=1)#,'leaguePoints','adjustedPoints'], axis=1)


    #y= df['adjustedPoints']
    #y=df['leaguePoints']
    #X=add_noise(X)

    featurelist_forML = get_lowVarienceCols(X)

    X = X[featurelist_forML]
    ##corlation analysis
    X = correlation_analysis(X)

    #X.fillna(0.000001, inplace=True)




    # Combine the normalized features with the target variable


    if model_Type=='1':
        # Convert target variable to one-hot encoding
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        y_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_categories)
        y_one_hottest = tf.keras.utils.to_categorical(y_test, num_classes=num_categories)

        # Build a simple Deep Learning model
        model = Sequential()
        model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))  # Input layer with 100 features
        model.add(Dense(32, activation='relu'))  # Hidden layer
        model.add(Dense(32, activation='relu'))  # Hidden layer
        model.add(Dense(num_categories, activation='softmax'))  # Output layer with 5 nodes for 5 classes

        # Compile the model
        adamOpti = Adam(lr=0.001)
        model.compile(optimizer=adamOpti, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_one_hot, epochs=300, batch_size=64, validation_split=0.3)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_one_hottest)
        print(f"Model accuracy: {accuracy}")

        background = X[np.random.choice(X_train.shape[0], 1000, replace=False)]

        # explain predictions of the model on four images
        e = shap.DeepExplainer(model, background)
        # ...or pass tensors directly
        # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
        shap_values = e.shap_values(X_test[1:1000])

        shap.summary_plot(shap_values, X_test[1:1000], feature_names=X.columns, max_display=17)

        # shap.plots.beeswarm(shap_values, max_display=20)

        # shap.plots.bar(shap_values)

        plt.show()

    if model_Type=='2':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)#.view(-1, 1)  # Assuming y is a pandas Series
        X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)#.view(-1, 1)

        ##making a copy for using in shap
        X_train_tensor_forshap = X_train_tensor
        X_test_tensor_forshap = X_test_tensor

        X_train_tensor = X_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        # Define your neural network model
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(X_train.shape[1], 64)  # input layer (80) -> hidden layer (128)
                self.fc2 = nn.Linear(64, 5)  # hidden layer (128) -> output layer (5)

            def forward(self, x):
                x = torch.relu(self.fc1(x))  # activation function for hidden layer
                x = self.fc2(x)
                return x

        # Initialize the model, loss function, and optimizer
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Train the model
        losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total = y_train_tensor.size(0)
            correct = (predicted == y_train_tensor).sum().item()
            train_accuracy = correct / total
            train_accuracies.append(train_accuracy)

            # Calculate test accuracy
            test_outputs = model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs, 1)
            test_total = y_test_tensor.size(0)
            test_correct = (test_predicted == y_test_tensor).sum().item()
            test_accuracy = test_correct / test_total
            test_accuracies.append(test_accuracy)

        # Plot the loss and accuracy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(100), losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Loss Over Time')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(100), train_accuracies, label='Training Accuracy')
        plt.plot(range(100), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Time')
        plt.legend()
        plt.show()

        # Calculate feature importance
        feature_importances = []
        for param in model.parameters():
            feature_importances.extend(torch.abs(param).cpu().numpy().flatten())

        # Sort the feature importances
        sorted_importances = sorted(zip(range(X_train_tensor.shape[1]), feature_importances), key=lambda x: x[1], reverse=True)

        # Print the top 5 most important features
        print("Top 5 Most Important Features:")
        for i in range(5):
            print(f"Feature {sorted_importances[i][0]}: Importance {sorted_importances[i][1]}")

    if model_Type=='3':

        # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long).view(-1, 1)

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

        class Multiclass(nn.Module):
            def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_rate,
                             l2_penalty):

                super().__init__()

                self.hidden = nn.Linear(input_size, hidden_size1)
                self.act = nn.ReLU()
                self.output = nn.Linear(hidden_size1, output_size)

            def forward(self, x):
                x = self.act(self.hidden(x))
                x = self.output(x)
                return x

        input_size = X_train.shape[1]
        hidden_size1 = 128
        hidden_size2 = 128
        hidden_size3 = 64
        output_size = num_categories
        dropout_rate = 0.1
        l2_penalty = 1e-8

        # loss metric and optimizer
        model = Multiclass(input_size, hidden_size1, hidden_size2,hidden_size3, output_size, dropout_rate, l2_penalty)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # prepare model and training parameters
        n_epochs = 200
        batch_size = 5
        batches_per_epoch = len(X_train) // batch_size

        best_acc = - np.inf  # init to negative infinity
        best_weights = None
        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []

        # training loop
        for epoch in range(n_epochs):
            epoch_loss = []
            epoch_acc = []
            # set model in training mode and run through each batch
            model.train()
            with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")
                for i in bar:
                    # take a batch
                    start = i * batch_size
                    X_batch = X_train[start:start + batch_size]
                    y_batch = y_train[start:start + batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    #loss = loss_fn(y_pred, y_batch)

                    outputs = model(X_train)
                    loss = loss_fn(outputs, y_train)

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # compute and store metrics
                    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                    epoch_loss.append(float(loss))
                    epoch_acc.append(float(acc))
                    bar.set_postfix(
                        loss=float(loss),
                        acc=float(acc)
                    )
            # set model in evaluation mode and run through the test set
            model.eval()
            y_pred = model(X_test)
            ce = loss_fn(y_pred, y_test)
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
            ce = float(ce)
            acc = float(acc)
            train_loss_hist.append(np.mean(epoch_loss))
            train_acc_hist.append(np.mean(epoch_acc))
            test_loss_hist.append(ce)
            test_acc_hist.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc * 100:.1f}%")

        # Restore best model
        model.load_state_dict(best_weights)

        # Plot the loss and accuracy
        plt.plot(train_loss_hist, label="train")
        plt.plot(test_loss_hist, label="test")
        plt.xlabel("epochs")
        plt.ylabel("cross entropy")
        plt.legend()
        plt.show()

        plt.plot(train_acc_hist, label="train")
        plt.plot(test_acc_hist, label="test")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()#

    if model_Type == '4':  ###pytorch
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        # Apply SMOTE to oversample the minority class in the training set
       # smote = SMOTE(random_state=42)
       # X_train, y_train = smote.fit_resample(X_train, y_train)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)#.view(-1, 1)  # Assuming y is a pandas Series
        X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)#.view(-1, 1)

        ##making a copy for using in shap
        X_train_tensor_forshap = X_train_tensor
        X_test_tensor_forshap = X_test_tensor

        X_train_tensor = X_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)



        # Define a simple neural network model
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size, hidden_size1, hidden_size2,hidden_size3, output_size, dropout_rate, l2_penalty):
                super(NeuralNetwork, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size1)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size1,hidden_size2)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()
                self.fc3 = nn.Linear(hidden_size2,output_size)
                #self.dropout = nn.Dropout(dropout_rate)
                #self.relu = nn.ReLU()
               # self.fc4 = nn.Linear(hidden_size3, output_size)
                self.l2_penalty = l2_penalty
                #self.softmax = nn.Softmax(dim=1)
                self.elu = nn.ELU()

            def forward(self, x):
                x = self.fc1(x)
                #x = self.dropout(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                #x = self.dropout(x)
                #x = self.relu(x)
                #x = self.fc3(x)
                #x = self.relu(x)
               # x = self.fc4(x)
                return x


            # Define a function that takes a NumPy array as input and returns PyTorch model predictions

        def predict(input_data):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            model_output = model(input_tensor)
            return model_output.detach().numpy()

        def one_hot_ce_loss(outputs, targets):
            criterion = nn.CrossEntropyLoss()
            _, labels = torch.max(targets, dim=1)
            return criterion(outputs, labels)

        # Model parameters
       # print (X.columns)
        input_size = X_train.shape[1]
        hidden_size1 = 128
        hidden_size2 = 64
        hidden_size3 = 64
        hidden_size4 = 64
        output_size = num_categories
        dropout_rate = 0
        l2_penalty = 1e-8

        # Initialize the model, loss function, and optimizer
        model = NeuralNetwork(input_size, hidden_size1, hidden_size2,hidden_size3, output_size, dropout_rate, l2_penalty)
        model.to(device)
        # Compute class weights
        #class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        #class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Modify the criterion to include class weights
        #criterion = nn.CrossEntropyLoss(weight=class_weights)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)



        # Training parameters
        num_epochs = 200
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
                from sklearn.metrics import classification_report

                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())


                _, predicted = torch.max(outputs, 1)
                total = y_train_tensor.size(0)
                correct = (predicted == y_train_tensor).sum().item()
                train_accuracy = correct / total
                train_accuracies.append(train_accuracy)

                # Validation
                model.eval()

                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_loss = criterion(val_outputs, y_test_tensor)
                    val_losses.append(val_loss.item())

                    # Calculate test accuracy
                    test_outputs = model(X_test_tensor)
                    _, test_predicted = torch.max(test_outputs, 1)
                    test_total = y_test_tensor.size(0)
                    test_correct = (test_predicted == y_test_tensor).sum().item()
                    val_accuracy = test_correct / test_total
                    val_accuracies.append(val_accuracy)


                # Print epoch statistics
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Training Loss: {loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
                      f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Create DataLoader with weighted sampler
        """train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=32, sampler=sampler)
        test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=32)

        # Train model
        for epoch in range(10):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate model
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy: {100 * correct / total}%')
        """
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


        # create_modelArt(model,input_size)
        # select a set of background examples to take an expectation over
       # background = X_train_tensor[np.random.choice(X_train_tensor.shape[0], 100, replace=False)]

        #explainer = shap.Explainer(predict, X_train_tensor_forshap.numpy())

        # Calculate SHAP values for the test set
      #  shap_values = explainer.shap_values(X_test_tensor_forshap.numpy())

        # explain predictions of the model on four images
        #e = shap.DeepExplainer(model, background)
        # ...or pass tensors directly

        # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
       # shap_values = e.shap_values(X_test_tensor_forshap.numpy())

       # shap.summary_plot(shap_values, X_test_tensor[1:100], feature_names=X.columns, max_display=17)

        # shap.plots.beeswarm(shap_values, max_display=20)

        # shap.plots.bar(shap_values)

       # plt.show()

        # plot the feature attributions
        # shap.image_plot(shap_values, -X_test_tensor[1:5])

        ############################################################################
        # SHAP PLOTS
        # Create an explainer

        """explainer = shap.Explainer(predict, X_train_tensor_forshap.numpy())

        # Calculate SHAP values for the test set
        shap_values = explainer.shap_values(X_test_tensor_forshap.numpy())

        # Create a beeswarm plot
        shap.summary_plot(shap_values, X_test_tensor_forshap, feature_names=X.columns, auto_size_plot=True)

        shap.summary_plot(shap_values, X_test_tensor_forshap, feature_names=X.columns, auto_size_plot=True,
                          plot_type="bar")
        # shap.plots.bar(shap_values,max_display=15)
        plt.show()
    """
    if model_Type == '5':
        # Assume X_train, y_train are your training data
        X_train, X_test, y_train, y_test = train_test_split(df.drop('Target', axis=1), df['Target'], test_size=0.2,
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
        hidden_size2 = 32  # Number of units in the second hidden layer
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

    if model_Type=='6':

        # Define the neural network model
        class RegressionModel(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(RegressionModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, 1)  # Output layer with 1 neuron for regression

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out

        # Prepare the data (X: input features, y: target variable)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data (important for some deep learning models)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # .view(-1, 1)  # Assuming y is a pandas Series
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)  # .view(-1, 1)


        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Set up the model, loss function, and optimizer
        model = RegressionModel(input_size=X_train.shape[1], hidden_size=64)
        criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training parameters
        num_epochs = 600
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            from sklearn.metrics import classification_report

            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            total = y_train_tensor.size(0)
            correct = (predicted == y_train_tensor).sum().item()
            train_accuracy = correct / total
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()

            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                val_losses.append(val_loss.item())

                # Calculate test accuracy
                test_outputs = model(X_test_tensor)
                _, test_predicted = torch.max(test_outputs, 1)
                test_total = y_test_tensor.size(0)
                test_correct = (test_predicted == y_test_tensor).sum().item()
                val_accuracy = test_correct / test_total
                val_accuracies.append(val_accuracy)

            # Print epoch statistics
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Training Loss: {loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    if model_Type=='7':
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn.neighbors import KNeighborsClassifier
        # Split the original dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply oversampling using RandomOverSampler
        oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
        X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)

        # Apply undersampling using RandomUnderSampler
        undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)

        # Fit KNN classifier on the original train set
        knn_original = KNeighborsClassifier(n_neighbors=5)
        knn_original.fit(X_train, y_train)

        # Fit KNN classifier on the oversampled train set
        knn_oversampled = KNeighborsClassifier(n_neighbors=5)
        knn_oversampled.fit(X_train_oversampled, y_train_oversampled)

        # Fit KNN classifier on the undersampled train set
        knn_undersampled = KNeighborsClassifier(n_neighbors=5)
        knn_undersampled.fit(X_train_undersampled, y_train_undersampled)

        # Make predictions on train sets
        y_train_pred_original = knn_original.predict(X_train)
        y_train_pred_oversampled = knn_oversampled.predict(X_train_oversampled)
        y_train_pred_undersampled = knn_undersampled.predict(X_train_undersampled)

        # Make predictions on test sets
        y_test_pred_original = knn_original.predict(X_test)
        y_test_pred_oversampled = knn_oversampled.predict(X_test)
        y_test_pred_undersampled = knn_undersampled.predict(X_test)

        # Calculate and print accuracy for train sets
        print("Accuracy on Original Train Set:", accuracy_score(y_train, y_train_pred_original))
        print("Accuracy on Oversampled Train Set:", accuracy_score(y_train_oversampled, y_train_pred_oversampled))
        print("Accuracy on Undersampled Train Set:", accuracy_score(y_train_undersampled, y_train_pred_undersampled))

        # Calculate and print accuracy for test sets
        print("\nAccuracy on Original Test Set:", accuracy_score(y_test, y_test_pred_original))
        print("Accuracy on Oversampled Test Set:", accuracy_score(y_test, y_test_pred_oversampled))
        print("Accuracy on Undersampled Test Set:", accuracy_score(y_test, y_test_pred_undersampled))

def get_name(num):
    if num == 1:
        return "Earlygame"
    elif num == 2:
        return "Midgame"
    elif num == 3:
        return "Lategame"


####Read dataset
Inputdata_path = "data/OutputRank/MatchResume/FinalRankedMatchResume_Masterfile.csv"


Inputdata_pathT = "data/OutputRank/MatchTimeline/RankedMatchTimeline_Masterfile.csv"

outputdata_path1 = "data/OutputRank/FeatureSelection/KPI_MatchTimelinePerRank"
#df = pd.read_csv(Inputdata_path)


inputdata_pathforSelectedCols = "data/Input/MatchResume_FinalColsRanked.csv"
df_cols = pd.read_csv(inputdata_pathforSelectedCols)


inputdata_pathforTimelineSelectedCols = "data/Input/MatchTimeline_FinalCols.csv"
df_colsTimeline = pd.read_csv(inputdata_pathforTimelineSelectedCols)

###using cpu instead of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

phase_list=[1,2,3]
#make_deep_learning(df, df_cols)

###for matchtimeline_ranked player
df = pd.read_csv(Inputdata_pathT)
new_rows = pd.DataFrame({'featurename': ['tier', 'rank']})
df_colsTimeline=pd.concat([df_colsTimeline, new_rows], ignore_index=True)
for phase in phase_list:
    df_phase = df[df['Phase'] == phase]
    print(phase, len(df_phase))
    # filtered_df = Normalize_perGameduration(df_phase,'gameDuration')

    output_file = outputdata_path1 + '_' + get_name(phase) + '.csv'
    make_deep_learning(df_phase, df_colsTimeline, output_file)


