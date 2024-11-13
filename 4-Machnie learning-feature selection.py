# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:25:32 2023

@author: Fazilat
"""
import numpy as np
import pandas as pd
import os
import shutil

import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from unicodedata import normalize
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

import shap
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold

def get_lowVarienceCols(X):
    # Specify the variance threshold
    variance_threshold = 0.01  # You can adjust this threshold based on your preference



    variances = X.var()
    high_variance_columns = variances[variances > variance_threshold].index

   # high_varcols=X.drop(columns=low_variance_columns,axis=1)
    return (high_variance_columns)

###########################
def lowVariance_Filtering(y, data):
    # storing the variance and name of variables

    norm = preprocessing.normalize(data)
    data_scaled = pd.DataFrame(norm)
    variance = data_scaled.var()

    columns = data.columns

    # saving the names of variables having variance more than a threshold value

    variable = list()
    print(len(variance), 'low varience')
    print(variance)
    for i in range(0, len(variance)):

        if variance[i] >= 0:  # setting the threshold as 1%
            variable.append(columns[i])
            # print (variance[i],columns[i])

    print(len(variable), 'len')
    return (variable)


################################################
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


def kfold_crossvalidation(X, y, model, modelname):
    # kf=KFold(n_splits=15,random_state=None, shuffle=False)

    list_training_error = []
    list_testing_error = []

    kf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    # kf=split(X, y, groups=None)

    for train_index, test_index in kf.split(X, y):
        # X_train,X_test=X['index1'=train_index],X['index1'=test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_train_data_pred = model.predict(X_train)
        y_test_data_pred = model.predict(X_test)
        fold_training_error = mean_absolute_error(y_train, y_train_data_pred)
        fold_testing_error = mean_absolute_error(y_test, y_test_data_pred)
        list_training_error.append(fold_training_error)
        list_testing_error.append(fold_testing_error)
        # print (fold_training_error,fold_testing_error)

    plt.subplot(1, 2, 1)
    plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), 'o-')
    plt.xlabel('number of fold')
    plt.ylabel('training error')
    plt.title('Training error across folds %s' % modelname)
    plt.tight_layout()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel(), 'o-')
    plt.xlabel('number of fold')
    plt.ylabel('testing error')
    plt.title('Testing error across folds %s' % modelname)
    plt.tight_layout()
    plt.show()
    return (kf)


# Creating function for scaling
def Standard_Scaler(df, col_names):
    features = df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df[col_names] = features

    return df


# run feature selection for all cluster
def run_models(X, y,featurelist_forML):
    f = plt.figure()
    f.set_figwidth(30)
    f.set_figheight(20)

    importance_score = []
   # best_importance = pd.DataFrame(columns=['cluster', 'Attribute', 'Importance'])
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, shuffle=True)

    # Feature Scaling
    X_train = Standard_Scaler(X_train, featurelist_forML)
    X_test = Standard_Scaler(X_test, featurelist_forML)

    results = []
    names = []
    df_mae = pd.DataFrame()
    seed = 50
    scoring = 'accuracy'
    for name, model in models:
        print(name, model)
        # kfold = model_selection.KFold(n_splits=10, random_state=seed)
        #kfold=kfold_crossvalidation(X,y,model,name)
        kfold = 4
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)  # , scoring=scoring)
        model.fit(X_train, y_train)

        # find number of features
        # select_optimalnumberofFeatures(X_train,y_train,kfold,model)
        # accuracy
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        # confusion matrix
        predictions = model.predict(X_test).round()
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        perm_importance = permutation_importance(model, X_test, y_test)
        plt.rcParams.update({'font.size': 10})

        # shap summary
        # shap.initjs()
        if name == 'SVR':
            importance_score = abs(perm_importance.importances_mean)

            explainer = shap.KernelExplainer(model.predict, X_train, feature_names=featurelist_forML)
            # shap_values = explainer.shap_values(X_train)
        # preparedataforShapPlot(model,X_train)
        if name == 'LR':
            importance_score = abs(model.coef_[0])

            explainer = shap.Explainer(model, X_train, feature_names=featurelist_forML)
            shap_values = explainer(X_train)
            # shap.plots.waterfall(shap_values[0])
            # preparedataforShapPlot(model,X_train)
            # shap.plots.bar(shap_values,show=False)
            # shap.summary_plot(explainer, feature_names=X_test.columns, plot_type='bar')
        # plt.title('for Cluster All- Logistic Regression')
        # plt.show()

        if name == 'RF':
            importance_score = abs(model.feature_importances_)

            # Our Code
            explainer = shap.TreeExplainer(model)

            # Visualize one value
            single_shap_value = explainer(X_test.sample(n=1))
            shap.summary_plot(single_shap_value, feature_names=X_test.columns, plot_type='bar')
            # plt.title('for Cluster All- Random Forest')
            # plt.show()
            explainer = shap.Explainer(model, X_train, feature_names=featurelist_forML)
            shap_values = explainer(X_train)
            # shap.plots.waterfall(shap_values[0])
            # preparedataforShapPlot(model,X_train)
            shap.plots.bar(shap_values)
            plt.title("Random Forest", y=1.75)

            plt.show()

        if name == 'XG':
            importance_score = abs(model.feature_importances_)
            explainer = shap.Explainer(model, X_train, feature_names=featurelist_forML)

            single_shap_value = explainer(X_test.sample(n=1))
            shap.summary_plot(single_shap_value, feature_names=X_test.columns, plot_type='bar')
            # shap_values = explainer(X_train)
            shap.plots.bar(single_shap_value)
            # shap.plots.waterfall(single_shap_value[0])
            #preparedataforShapPlot(model, X_train)

        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f) (%f)" % (name, cv_results.mean(), cv_results.std(), mae)
        print(msg)

        importances = pd.DataFrame(data={
            'cluster': 'all',
            'Attribute': featurelist_forML,
            'Importance': importance_score
        })

        if name == 'LR':
            best_importance = importances

        new_row = pd.Series(data={'cluster': 'all',
                                  'Attribute': name,
                                  'Importance': abs(mae)}, name='x')
        #df_mae = df_mae.append(new_row, ignore_index=False)
        df_mae=pd.concat([df_mae,pd.DataFrame([new_row])], ignore_index=False)
        if IS_SORTED == True:
            importances = importances.sort_values(by='Importance', ascending=ascending)

        plt.bar(importances['Attribute'], importances['Importance'], color='#087E8B')
        plt.show()

        new_row = pd.Series(data={'cluster': 'all',
                                  'Attribute': name,
                                  'Importance': mae}, name='x')
        df_mae = pd.concat([df_mae, pd.DataFrame([new_row])], ignore_index=False)

        ####
        # importances.to_excel(writer, sheet_name=name)
        output_filenamecsv = output_filename + '_' + name + '.csv'

        print(output_filenamecsv)

        importances.to_csv(output_filenamecsv)

    # df_mae.to_excel(writer, sheet_name=name)
    # df_mae.to_csv(output_filenamecsv)


def preparedataforShapPlot(model, df):
    from types import SimpleNamespace
    import copy
    model = model  # insert your model here
    inputs = df  # insert your inputs as pandas Dataframe

    explainer = shap.Explainer(model)  # change explainer as needed
    shap_values = explainer(inputs)
    feature_names = inputs.columns

    sh = copy.deepcopy(shap_values)
    sh.values = sh.values[:, :, 0]  # leave only 1st y
    shap.plots.beeswarm(sh)


"""
    to_pass = SimpleNamespace(**{
                          'values': np.array(shap_values[0].values[:,0]),
                          'data': np.array(shap_values[0].data),
                          'feature_names': feature_names,
                          'base_values': shap_values[0].base_values[0]
            })

    shap.plots.waterfall(to_pass)

    to_pass = SimpleNamespace(**{
                              'values': np.array(shap_values[0].values[:,1]),
                              'data': np.array(shap_values[0].data),
                              'feature_names': feature_names,
                              'base_values': shap_values[0].base_values[0]
            })

    shap.plots.waterfall(to_pass)

    # workaround...
    to_pass = SimpleNamespace(**{
                              'values': np.array(shap_values[0].values),
                              'data': np.array(shap_values[0].data),
                              'feature_names': inputs.columns,
                              'base_values': shap_values[0].base_values[0]
                })
    shap.plots.waterfall(to_pass)
    """


############################ preparing data for applying ML algorithms
def Feature_Selection(input_file, output_filename,df_cols):
    df = pd.read_csv(input_file)

    feature_names = df_cols.iloc[:, 0].tolist()
    df=df[feature_names]


    #threshold = 0.7 * len(df)

    le = LabelEncoder()

    # Encode the categorical target variable
    df['Rank'] = le.fit_transform(df['Rank'])

    y = df['Rank']

    df = df.select_dtypes(include='number')
    df=df.fillna(0)
    X = df.drop(['Rank'], axis=1)
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Normalize the data
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    featurelist_forML= get_lowVarienceCols(X)

    X=X[featurelist_forML]
    ##corlation analysis
    X=correlation_analysis(X)



    run_models(X, y,X.columns)


MAE_reg = []
MAE_xg = []
MAE_PCA = []
MAE_RF = []

#input_file = "data/Output/MatchChallenges/MatchChallengesNA.csv"
input_file = "data/OutputRank/MatchResume/RankedMatchResumeMasterfile_NoOutlier.csv"
#input_file = "data/Output/Masterfile/Matserfile_diffScore.csv"
outputdata_path = "data/Output/MatchChallenges/FeatureSelection/"

inputdata_pathforSelectedCols = "data/Input/MatchResume_FinalColsRanked.csv"
df_cols = pd.read_csv(inputdata_pathforSelectedCols)



#warnings.filterwarnings('ignore', category=DeprecationWarning)

models = []
models.append(('LR', LogisticRegression(max_iter=10000)))

#models.append(('SVR', SVR(kernel='rbf')))
models.append(('LR', LinearRegression())) #LinearRegression()
xgb = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                    gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=12,
                    min_child_weight=1, n_estimators=500, nthread=-1,
                    objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                    scale_pos_weight=1, seed=0, silent=True, subsample=1, _label_encoder=False)

models.append(('XG', xgb))
models.append(('RF',
               RandomForestRegressor(oob_score=True, n_jobs=-1, random_state=50, max_features=12, min_samples_leaf=50,
                                     n_estimators=30)))

ascending = True
# channel_list=Make_channelList(dc,is_all,False)

###if we want to include all of the stats for each channel this var should be True , othervise False
plt.rcParams["figure.figsize"] = (30, 20)
plt.rcParams.update({'font.size': 40})
###if we want tto use data for heatmap, it is False , otherwise True
IS_SORTED = True
####Creating output directory for the results
if not os.path.exists(outputdata_path):
    os.makedirs(outputdata_path)
else:

    # remove folder with all of its files
    shutil.rmtree(outputdata_path)
    os.makedirs(outputdata_path)

output_filename = outputdata_path + 'FeatureRanking'
# writer = pd.ExcelWriter(output_filename, engine = 'xlsxwriter')
Feature_Selection(input_file, output_filename,df_cols)

# writer.save()
# writer.close()
