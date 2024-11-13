

import numpy as np
import pandas as pd
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU,Dropout

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

plt.rcParams["figure.figsize"] = (8,10)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def Deeplearning_forMatchResume ():
    df = pd.read_csv(Inputdata_pathMatchResume)

    inputdata_pathforSelectedCols = "../data/Input/MatchResume_FinalCols.csv"
    df_cols = pd.read_csv(inputdata_pathforSelectedCols)

    #filtered_df=Normalize_perGameduration(df,'challenges_gameLength')
    make_deep_learning(df, df_cols)
##########################################################################
def Deeplearning_forMatchTimeline ():
    df = pd.read_csv(Inputdata_pathMatchTimeline)

    filtered_df=Normalize_perGameduration(df,'gameDuration')

    phase=3
    if phase==0:
        df_phase1=df
    else:
        df_phase1=filtered_df[filtered_df['Phase']==phase]

    inputdata_pathforSelectedCols = "../data/Input/MatchTimeline_FinalCols.csv"
    df_cols = pd.read_csv(inputdata_pathforSelectedCols)


    df_phase1 = df_phase1.drop(['matchId', 'gameDuration', 'Unnamed: 0','Unnamed: 0.1','Phase','position_x','position_y'], axis=1)

    make_deep_learning_forMatchTimeline(df_phase1, df_cols,phase)
#########################################################################
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
    variance_threshold = 0.05  # You can adjust this threshold based on your preference



    variances = X.var()
    high_variance_columns = variances[variances > variance_threshold].index

   # high_varcols=X.drop(columns=low_variance_columns,axis=1)
    return (high_variance_columns)


def plot_trainingAUC(history,y_test,y_prob):
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


def plot_ModelPerfromanceMetrics(X_train, X_test, y_train, y_test,model):
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

def create_modelArt(model,input_size):


    # Assuming `model` is your PyTorch model
    # Input tensor example
    x = torch.randn(1, input_size)

    # Generate a computational graph
    y = model(x)
    graph = make_dot(y, params=dict(model.named_parameters()),show_attrs=True, show_saved=True)

    # Save the graph as an image or display it
    graph.render("model_graph", format="png", cleanup=True)

def Normalize_perGameduration (df,gameduration):
    # Define the column names to exclude from normalization
    exclude_cols = ['win','Phase']  # Add any column names you want to exclude

    df = df.apply(pd.to_numeric, errors='coerce')

    #df['gameDuration'] = pd.to_numeric(df['gameDuration'], errors='coerce')

    # Get the list of columns to normalize (all columns except exclude_cols)
    normalize_cols = df.columns.difference(exclude_cols)
    normalized_df = df.copy()

    # Normalize all columns based on the game duration
    normalized_df[normalize_cols] = normalized_df[normalize_cols].div(normalized_df[gameduration], axis=0)

    return(normalized_df)
#####
def make_deep_learning_forMatchTimeline(df, df_cols,phase):
    # Convert the DataFrame column to a Python array
    feature_names = df_cols.iloc[:, 0].tolist()

    model_Type = '4'
    # Initialize the MinMaxScaler
    # scaler = MinMaxScaler()

    df = df.select_dtypes(include='number')
    # df=df[feature_names]

    X = df.drop(['win'], axis=1)
    print(X.columns)
    # Calculate the threshold for exclusion (60% of data being zero)
    #threshold = 0.6 * len(df)

    # Exclude columns where more than 60% of the data is zero
    #X = X.drop(columns=X.columns[(X == 0).sum() > threshold])

    y = df['win']

    # X=add_noise(X)

    featurelist_forML = get_lowVarienceCols(X)

    X = X[featurelist_forML]
    ##corlation analysis
    # X = correlation_analysis(X)

    #X.fillna(0.0000001, inplace=True)

    # Normalize the data
    # scaler=MinMaxScaler()
    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Combine the normalized features with the target variable



    if model_Type == '3':
        df_normalized = pd.DataFrame(X, columns=X.columns)

        # Step 1: Split the data into training and temporary sets while maintaining class balance
        df_train, df_temp = train_test_split(df_normalized, test_size=0.3, stratify=y, random_state=42)

        # Step 2: Further split the temporary set into testing and validation sets
        df_test, df_val = train_test_split(df_temp, test_size=0.5, stratify=df_temp['win'], random_state=42)

        # Extract features (X) and target variable (y) for each set
        X_train, y_train = df_train.drop('win', axis=1), df_train['win']
        X_test, y_test = df_test.drop('win', axis=1), df_test['win']
        X_val, y_val = df_val.drop('win', axis=1), df_val['win']

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define a simple neural network for regression
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1]))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.5))
        """
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.5))
        """
        model.add(Dense(1, activation='sigmoid'))  # Output layer for regression

        # Compile the model
        custom_optimizer = Adam(learning_rate=0.001)  # , loss='mean_squared_error')

        model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[early_stopping], verbose=1, shuffle=True)

        plot_ModelPerfromanceMetrics(X_train, X_test, y_train, y_test, model)
        y_prob = model.predict(X_test)

        y_pred = np.round(y_prob)
        plot_trainingAUC(history, y_test, y_pred)
        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {test_accuracy}')

        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        # Get feature importance scores
        last_layer_weights = model.layers[0].get_weights()[0]

        importance_scores = np.mean(np.abs(last_layer_weights), axis=0)

        # Create a DataFrame to associate feature names with importance scores
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})

        # Sort the DataFrame by importance scores in descending order
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Display the top features
        print(importance_df.head())

        """
        # Sort features based on importance
        sorted_features = np.argsort(feature_importance)[::-1]

        # Print the feature ranking
        print("Feature Ranking:")
        for i, feature_index in enumerate(sorted_features):
            print(f"Rank {i + 1}: Feature {feature_index} - Importance: {feature_importance[feature_index]}")
        """

    if model_Type == '4':  ###pytorch
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
            def __init__(self, input_size, hidden_size1, hidden_size2,hidden_size3,output_size, dropout_rate, l2_penalty):
                super(RegularizedModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size1)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()
                #self.fc2 = nn.Linear(hidden_size1, hidden_size2)
                #self.dropout = nn.Dropout(dropout_rate)
                #self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size1, output_size)
                self.l2_penalty = l2_penalty

            def forward(self, x):
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.relu(x)
                x = self.fc2(x)
                #x = self.dropout(x)
                #x = self.relu(x)
                #x = self.fc3(x)
                return x
            # Define a function that takes a NumPy array as input and returns PyTorch model predictions

        def predict(input_data):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            model_output = model(input_tensor)
            return model_output.detach().numpy()

        # Model parameters
        input_size = X_train.shape[1]
        hidden_size1 = 128
        hidden_size2 = 128
        hidden_size3 = 128
        output_size = 1
        dropout_rate = 0.5
        l2_penalty = 1e-5

        # Create the model, loss function, and optimizer
        model = RegularizedModel(input_size, hidden_size1, hidden_size2,hidden_size3,output_size, dropout_rate, l2_penalty)
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=l2_penalty, betas=(0.9, 0.999))



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
            #loss = hinge_loss(outputs, y_train_tensor)
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
                #val_loss = hinge_loss(val_outputs, y_test_tensor)

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
        plt.title(f'Training and Validation Loss Over Time for :{phase}')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Validation Accuracy Over Time:{phase}')
        plt.legend()

        plt.tight_layout()
        plt.show()
        ###################################
        df = pd.DataFrame({
            'Epoch': range(1, num_epochs + 1),
            'Train Loss': train_losses,
            'Train Accuracy': train_accuracies,
            'Val Loss': val_losses,
            'Val Accuracy': val_accuracies
        })

        # Save DataFrame to CSV
        df.to_csv(lossoutput, index=False)
        #####################################


        ###########################################################################
        print (X.columns)
        feature_importances = model.fc1.weight.detach().numpy()[0]

        abs_feature_importances = np.abs(feature_importances)

        sorted_indices = np.argsort(abs_feature_importances)[::-1]

        # Get the feature names and sorted importance scores
        sorted_feature_names = X.columns[sorted_indices]
        sorted_importances = feature_importances[sorted_indices]

        # Create a DataFrame with feature names and importance scores
        feature_importance_df = pd.DataFrame({
            'Feature': sorted_feature_names,
            'Importance Score': sorted_importances
        })

        # Save to CSV
        outputfile=OUTPUTdata_path+str(phase)+'.csv'
        feature_importance_df.to_csv(outputfile, index=False)

        # Plot bar chart for feature importance
        plt.figure(figsize=(14, 8))
        plt.bar(sorted_feature_names, sorted_importances, color='#087E8B')
        plt.xlabel("Feature")
        plt.ylabel("Importance Score")
        plt.title("Feature Importance (All Features Sorted by Absolute Value)")
        plt.xticks(rotation=45, ha="right",
                   rotation_mode="anchor")  # Rotate labels at 45 degrees, anchored to the right
        plt.tight_layout()  # Adjust layout to ensure labels fit without overlapping

        plt.show()

        # create_modelArt(model,input_size)
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




#######

def make_deep_learning(df, df_cols):

    # Convert the DataFrame column to a Python array
    feature_names = df_cols.iloc[:, 0].tolist()

    model_Type='4'
    # Initialize the MinMaxScaler
    #scaler = MinMaxScaler()
    df=df[feature_names]
    df = df.select_dtypes(include='number')



    X=df.drop(['win'],axis=1)

    # Calculate the threshold for exclusion (60% of data being zero)
    #threshold = 0.6 * len(df)

    # Exclude columns where more than 60% of the data is zero
    #X = X.drop(columns=X.columns[(X == 0).sum() > threshold])

    y=df['win']

    #X=add_noise(X)

    featurelist_forML = get_lowVarienceCols(X)

    X = X[featurelist_forML]
    ##corlation analysis
    X = correlation_analysis(X)
    X = X.loc[:, ~X.columns.str.contains('_diff')]

    X.fillna(0.0000001, inplace=True)




    # Normalize the data
    #scaler=MinMaxScaler()
    #scaler = StandardScaler()
    #X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Combine the normalized features with the target variable
    df_normalized = pd.DataFrame(X, columns=X.columns)
    df_normalized['win'] = y

    if model_Type=='1' :

        #####CancelOut######################
        #('Sigmoid + Loss')##################################
        inputs = keras.Input(shape=(X.shape[1],))


        x = CancelOut(activation='sigmoid')(inputs)
        x = layers.Dense(32, activation="relu")(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.summary()
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        df_train, df_temp = train_test_split(df_normalized, test_size=0.3, stratify=df['win'], random_state=42)

        # Step 2: Further split the temporary set into testing and validation sets
        df_test, df_val = train_test_split(df_temp, test_size=0.5, stratify=df_temp['win'], random_state=42)

        # Extract features (X) and target variable (y) for each set
        X_train, y_train = df_train.drop('win', axis=1), df_train['win']
        X_test, y_test = df_test.drop('win', axis=1), df_test['win']
        X_val, y_val = df_val.drop('win', axis=1), df_val['win']


        # Train the model

        history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[early_stopping], verbose=1, shuffle=True)

        y_prob = model.predict(X_test)

        y_pred = np.round(y_prob)
        plot_trainingAUC(history,y_test,y_pred)

        #model.fit(X, y, epochs=20, batch_size=8)
        cancelout_feature_importance_sigmoid = model.get_weights()[0]

        #print('Sigmoid + Loss')
        #plot_importance(cancelout_feature_importance_sigmoid)

    if model_Type=='2' :

        VIANN = VarImpVIANN(verbose=1)

        model = Sequential()
        model.add(Dense(64, input_dim=X.shape[1], activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(l1=0,l2=0.01)))
        model.add(Dense(128, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(l1=0,l2=0.01)))
        model.add(Dense(64, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(l1=0,l2=0.01)))
        model.add(Dense(1, activation='softmax', kernel_initializer='normal'))

        optimizer=Adam(learning_rate=0.00001, beta_1=0.9    )
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        history=model.fit(X, y, validation_split=0.05, epochs=30, batch_size=32, shuffle=True,
              verbose=1, callbacks = [VIANN])

        print(VIANN.varScores)
        for i in range(len(VIANN.varScores)):
            print (X.columns[i])

    if model_Type=='3' :
        # Step 1: Split the data into training and temporary sets while maintaining class balance
        df_train, df_temp = train_test_split(df_normalized, test_size=0.3, stratify=df['win'], random_state=42)

        # Step 2: Further split the temporary set into testing and validation sets
        df_test, df_val = train_test_split(df_temp, test_size=0.5, stratify=df_temp['win'], random_state=42)

        # Extract features (X) and target variable (y) for each set
        X_train, y_train = df_train.drop('win', axis=1), df_train['win']
        X_test, y_test = df_test.drop('win', axis=1), df_test['win']
        X_val, y_val = df_val.drop('win', axis=1), df_val['win']


        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



        # Define a simple neural network for regression
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1]))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.5))
        """
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.5))
        """
        model.add(Dense(1, activation='sigmoid'))  # Output layer for regression

        # Compile the model
        custom_optimizer = Adam(learning_rate=0.00001)#, loss='mean_squared_error')

        model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val),  callbacks=[early_stopping],verbose=1,shuffle=True)

        plot_ModelPerfromanceMetrics(  X_train,X_test,y_train,y_test,model)
        y_prob = model.predict(X_test)

        y_pred = np.round(y_prob)
        plot_trainingAUC(history,y_test,y_pred)
        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {test_accuracy}')

        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


        # Get feature importance scores
        last_layer_weights  = model.layers[0].get_weights()[0]

        importance_scores  = np.mean(np.abs(last_layer_weights), axis=0)

        # Create a DataFrame to associate feature names with importance scores
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})

        # Sort the DataFrame by importance scores in descending order
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Display the top features
        print(importance_df.head())

        """
        # Sort features based on importance
        sorted_features = np.argsort(feature_importance)[::-1]
    
        # Print the feature ranking
        print("Feature Ranking:")
        for i, feature_index in enumerate(sorted_features):
            print(f"Rank {i + 1}: Feature {feature_index} - Importance: {feature_importance[feature_index]}")
        """

    if model_Type=='4': ###pytorch
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
            X_train_tensor_forshap=X_train_tensor
            X_test_tensor_forshap=X_test_tensor


            X_train_tensor = X_train_tensor.to(device)
            X_test_tensor = X_test_tensor.to(device)
            y_train_tensor = y_train_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)

            # Define a simple neural network with regularization
            class RegularizedModel(nn.Module):
                def __init__(self, input_size, hidden_size, output_size, dropout_rate, l2_penalty):
                    super(RegularizedModel, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.dropout = nn.Dropout(dropout_rate)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_size, output_size)
                    self.l2_penalty = l2_penalty

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.dropout(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    return x
                # Define a function that takes a NumPy array as input and returns PyTorch model predictions
            def predict(input_data):
                    input_tensor = torch.tensor(input_data, dtype=torch.float32)
                    model_output = model(input_tensor)
                    return model_output.detach().numpy()

            # Model parameters
            input_size = X_train.shape[1]
            hidden_size = 64
            output_size = 1
            dropout_rate = 0.5
            l2_penalty = 1e-5

            # Create the model, loss function, and optimizer
            model = RegularizedModel(input_size, hidden_size, output_size, dropout_rate, l2_penalty)
            model.to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_penalty)

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

            ###################################
            df = pd.DataFrame({
                'Epoch': range(1, num_epochs + 1),
                'Train Loss': train_losses,
                'Train Accuracy': train_accuracies,
                'Val Loss': val_losses,
                'Val Accuracy': val_accuracies
            })

            # Save DataFrame to CSV
            df.to_csv(lossoutput, index=False)

            #create_modelArt(model,input_size)
            # select a set of background examples to take an expectation over
            background = X_train_tensor[np.random.choice(X_train_tensor.shape[0], 1000, replace=False)]

            # explain predictions of the model on four images
            e = shap.DeepExplainer(model, background)
            # ...or pass tensors directly
            # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
            shap_values = e.shap_values(X_test_tensor[1:1000])

            shap.summary_plot(shap_values, X_test_tensor[1:1000], feature_names=X.columns,max_display=17)


            #shap.plots.beeswarm(shap_values, max_display=20)

           # shap.plots.bar(shap_values)

            plt.show()

            # plot the feature attributions
           # shap.image_plot(shap_values, -X_test_tensor[1:5])




            ############################################################################
            #SHAP PLOTS
            # Create an explainer



            explainer = shap.Explainer(predict, X_train_tensor_forshap.numpy())

            # Calculate SHAP values for the test set
            shap_values = explainer.shap_values(X_test_tensor_forshap.numpy())

            # Create a beeswarm plot
            shap.summary_plot(shap_values, X_test_tensor_forshap, feature_names=X.columns, auto_size_plot=True)


            shap.summary_plot(shap_values, X_test_tensor_forshap, feature_names=X.columns,auto_size_plot=True,plot_type="bar")
            #shap.plots.bar(shap_values,max_display=15)
            plt.show()


            ###########################################################################




            feature_importances = model.fc1.weight.detach().numpy()[0]

            # Get the indices of the top 10 most important features
            top10_indices = np.argsort(np.abs(feature_importances))[-10:]

            # Print the names of the top 10 most important features
            top10_feature_names = X.columns[top10_indices]
            shap_values_df=pd.DataFrame(feature_importances[-10:],top10_feature_names)
            shap_values_df.to_csv(OUTPUTdata_pathMatchResume)

            print("Top 10 Most Important Features:")


            # Plot bar chart for feature importance
            plt.figure(figsize=(10, 6))
            plt.bar(top10_feature_names,feature_importances[top10_indices], color='#087E8B')

            #plt.bar(range(1, 11), feature_importances[top10_indices], align="center")
           # plt.xticks(range(1, 11), top10_feature_names, rotation=45, ha="right")
            plt.xlabel("Feature")
            plt.ylabel("Importance Score")
            plt.title("Top 10 Most Important Features")
            plt.show()

    if model_Type=='5' :
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

####Read dataset  MatchResume
Inputdata_pathMatchResume = "../data/OutputRank/MatchResume/FinalTeamMatchResume_Masterfile.csv"

OUTPUTdata_pathMatchResume = "../data/OutputRank/FeatureSelection/KPI_MatchOutcome.csv"

OUTPUTdata_path = "../data/OutputRank/FeatureSelection/KPI_MatchOutcome_"
####Read dataset MatchTimelineperPhase
Inputdata_pathMatchTimeline = "../data/OutputRank/MatchTimeline/MatchTimeline_PerMatchPhase.csv"

lossoutput="../data/OutputRank/ReadyToPlot/training_metrics.csv"
###using cpu instead of GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

#Deeplearning_forMatchResume()

Deeplearning_forMatchTimeline()





