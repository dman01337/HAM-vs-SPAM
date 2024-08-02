import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin


# Create a class to be used in our preprocessing pipeline that returns the engineered features
class EngineeredFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, best_subject_predict_proba):
        self.best_subject_predict_proba = best_subject_predict_proba
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        X_eng = pd.DataFrame(X, columns=['body'])
        
        # Add a column to the dataframe that contains the length of the email body
        body_length = X_eng['body'].apply(len)
        # Add a column to the dataframe that contains the number of words in the email body
        X_eng['word_count'] = X_eng['body'].apply(lambda x: len(x.split())) / body_length
        # Add a column to the dataframe that contains the number of digits in the email body
        X_eng['digit_count'] = X_eng['body'].apply(lambda x: len(re.findall(r'\d', x))) / body_length
        # Add a column to the dataframe that contains the number of uppercase letters in the email body
        X_eng['upper_count'] = X_eng['body'].apply(lambda x: len(re.findall(r'[A-Z]', x))) / body_length
        # Add a column to the dataframe that contains the number of words in all caps in the email body
        X_eng['all_caps_count'] = X_eng['body'].apply(lambda x: len(re.findall(r'\b[A-Z]{2,}\b', x))) / body_length
        # Add a column to the dataframe that contains the number of exclamation points in the email body
        X_eng['exclamation_count'] = X_eng['body'].apply(lambda x: len(re.findall(r'!', x))) / body_length
        # Add a column to the dataframe that contains the number of question marks in the email body
        X_eng['question_count'] = X_eng['body'].apply(lambda x: len(re.findall(r'\?', x))) / body_length
        # Add a column to the dataframe that contains the subject line predicted probabilities
        X_eng['subject_predict_proba'] = self.best_subject_predict_proba

        return X_eng.drop('body', axis=1)