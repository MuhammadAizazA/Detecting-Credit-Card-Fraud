# The code is importing various libraries and modules that are commonly used in data preprocessing and analysis tasks. 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import auc, roc_curve

# These functions are likely custom functions that have been defined in the `data_utils` module
# to perform specific data preprocessing tasks. By importing these functions, the code can use 
# them to preprocess the data in a convenient and modular way.
from data_utils import prepare_data,label_encode_data,onehot_encode_data,frequency_encode_data,scale_data,balance_data,drop_columns

def load_data(path):
    dataframe=pd.read_csv(path)
    return dataframe

def complete_clean_data(path):
    df=load_data(path)
    # Preprocessing train data using the 'prepare_data' function
    df=prepare_data(df)
    
    df=label_encode_data(df,['city_pop','category','job','merchant','city','street','state'])
    df=frequency_encode_data(df,['cc_num'])
    df=onehot_encode_data(df,['day_type'])
    df=onehot_encode_data(df,['gender'])
    df=drop_columns(df, ['trans_date_trans_time','trans_time','trans_date','dob','merch_lat','merch_long','lat','long','during_officehours'])
    columns=df.drop(columns='is_fraud').columns
    df=scale_data(df,columns,'standard')
    df=df.dropna(axis=0)
    return df
    
def training_model(X, y):
    # The degree of the polynomial
    degree = 2

    # Initialize the PolynomialFeatures
    poly_features = PolynomialFeatures(degree=degree)

    # Defining the model with hyperparameter
    model = LogisticRegression(
        max_iter=3000,
        C=0.001,
        penalty='l2',
        solver='lbfgs',
        tol=1e-4,
        fit_intercept=True,
        class_weight={0:1, 1:1.5},
        random_state=42,
        multi_class='ovr'
    )

    # Create a pipeline of Polynomial features and Logistic regression
    pipeline = make_pipeline(poly_features, model)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = pipeline.predict(X_test)

    # Print performance metrics
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test, y_pred))

    # Introduce Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Mean Cross-Validation Score: {cv_scores.mean()}')

    return pipeline

def pridictive_function(X,y,pipeline):
    # Prediction
    y_pred = pipeline.predict(X)

    # Print performance metrics
    print('Accuracy: ', accuracy_score(y, y_pred))
    print('Confusion Matrix: \n', confusion_matrix(y, y_pred))
    
def roc_auc_score_function(X, y, pipeline):
    # Prediction
    y_pred_proba = pipeline.predict_proba(X)[:, 1]

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Print ROC AUC score
    print('ROC AUC Score: ', roc_auc)

    return roc_auc
if __name__=='__main__': 
    train_dataset_path='src/data/fraudTrain.csv'
    validation_dataset_path='src/data/fraudTest.csv'

    train_dataset=complete_clean_data(train_dataset_path)
    validation_dataset=complete_clean_data(validation_dataset_path)
    
    # Balancing the training data using SMOTE or other methods
    train_dataset=balance_data(train_dataset,'random_undersampling')
    
    # # Saving the cleaned training data to a CSV file
    # fraud_df_pre.to_csv('clean_train_data.csv', index=False)
    # # Saving the cleaned test data to a CSV file
    # validation_dataset.to_csv('clean_test_data.csv', index=False)
    
    # Creating resampled training set
    y_resampled = train_dataset['is_fraud']
    X_resampled = train_dataset.drop(columns=['is_fraud'])
    
    # Creating resampled validation set
    y_validation_resampled = validation_dataset['is_fraud']
    X_validation_resampled = validation_dataset.drop(columns=['is_fraud'])
    
    del train_dataset
    del validation_dataset
    
    print('Report for Training Model')
    trained_pipeline=training_model(X_resampled,y_resampled)
    print('Report for new unseen Data')
    pridictive_function(X_validation_resampled,y_validation_resampled,trained_pipeline)
    roc_auc = roc_auc_score_function(X_validation_resampled, y_validation_resampled, trained_pipeline)











