# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense
# import matplotlib.pyplot as plt
# import keras
# from keras.optimizers import Adam
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
#
# import seaborn as sns
#
#
# import numpy as np
#
# # Specify the path to your CSV file
# file_path = 'data/masterdata.csv'
#
# # Read the CSV file into a Pandas DataFrame
# df = pd.read_csv(file_path)
# # Generate some random data for demonstration
# data = np.random.random((1000, 20))
# labels = np.random.randint(2, size=(1000, 1))
#
# X=df[['abs glat_Avg','BRAKE _Avg','understeer_Avg','Brake_length_Avg','steering duration_Avg','number of crossing_Avg','Throttle release application_Avg','number of direction changed_Avg']]
# y=df['LapTime']
# # Build a simple model
# normalize_features=keras.utils.normalize(X.values)
# X_train,X_test,y_train,y_test=train_test_split(normalize_features,y,test_size=0.4,random_state=42)
#
# model = Sequential()
# model.add(Dense(8,  activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(3, activation='relu'))
#
# model.add(Dense(1, activation='relu'))
#
# # Compile the model
# optimizer = Adam(learning_rate=0.001)
# #model.compile(optimizer='adam', loss='mse', metrics=['mse'])
# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
#
#
# #simple early stopping
# from keras.callbacks import EarlyStopping
# es=EarlyStopping(monitor='val_loss', mode='min', verbose=1)
#
#
# # Train the model
# history=model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=100, batch_size=32)#,callbacks=[es])
#
# test_predictions=model.predict(X_test)
# pred_train=model.predict(X_train)
# print (np.sqrt(mean_squared_error(y_train,pred_train)))
#
# print (np.sqrt(mean_squared_error(y_test,test_predictions)))
#
# print (pred_train[:100])
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model')
# plt.ylabel('loss')
# plt.xlabel('Epoch')
# plt.legend(['train','val'], loc='upper right')
# plt.show()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap
# Generate a synthetic dataset for illustration purposes
# Replace this with your actual dataset
np.random.seed(42)
num_samples = 50000
time_steps = 10
num_features = 5

data = np.random.rand(num_samples, time_steps, num_features)
targets = np.random.rand(num_samples, 3)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

# Build a simple LSTM-based neural network
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, num_features)))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='linear'))  # 3 output nodes for the three target variables

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Create a SHAP explainer
explainer = shap.Explainer(model)

# Get SHAP values for the test set
shap_values = explainer.shap_values(X_test_scaled)

# Summarize feature importance across all time steps
feature_importance = np.abs(shap_values).mean(axis=(0, 1))

# Plot feature importance
shap.summary_plot(shap_values, X_test_scaled, feature_names=[f'Feature {i+1}' for i in range(num_features)], show=False)
plt.title('Feature Importance Across Time Steps')
plt.show()

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)

# Calculate mean squared error for each target variable at each time step
mse_by_time_step = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(3)]

# Plot the importance of features at different time steps
plt.bar(range(1, time_steps + 1), mse_by_time_step)
plt.xlabel('Time Step')
plt.ylabel('Mean Squared Error')
plt.title('Importance of Features at Different Time Steps')
plt.show()