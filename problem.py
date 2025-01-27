# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'MBTI: classify psychological profiles of people'
_target_column_name = 'type'
#_prediction_label_names = [0, 1, 2, 3, 4, 5] 
_prediction_label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
nom = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTJ','ISTP', 'ISFP', 'ENFJ', 'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ', 'ENTJ']


# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

#Soft_score_matrix generation
soft_score_matrix=[]
toadd= []
matcher = 0
for actual in nom:
    for autre in nom:
        if actual[0]==autre[0]:
            matcher = matcher + 0.25
        if actual[1]==autre[1]:
            matcher = matcher + 0.25
        if actual[2]==autre[2]:
            matcher = matcher + 0.25
        if actual[3]==autre[3]:
            matcher = matcher + 0.25
        toadd.append(matcher)
        matcher = 0
        
    #Ici on a fini une ligne, on va ajouter la ligne à la matrice
    soft_score_matrix.append(toadd)
    toadd = []
soft_score_matrix = np.array(soft_score_matrix)
        

# soft_score_matrix = np.array([
    # [1, 0.8, 0, 0, 0, 0],
    # [0.4, 1, 0.4, 0, 0, 0],
    # [0, 0.4, 1, 0.4, 0, 0],
    # [0, 0, 0.4, 1, 0.4, 0],
    # [0, 0, 0, 0.4, 1, 0.4],
    # [0, 0, 0, 0, 0.8, 1],
# ])


# true_false_score_matrix = np.array([
    # [1, 1, 1, 0, 0, 0],
    # [1, 1, 1, 0, 0, 0],
    # [1, 1, 1, 0, 0, 0],
    # [0, 0, 0, 1, 1, 1],
    # [0, 0, 0, 1, 1, 1],
    # [0, 0, 0, 1, 1, 1],
# ])

# score_types = [
    # # rw.score_types.SoftAccuracy(
        # # name='sacc', score_matrix=soft_score_matrix, precision=3),
    # rw.score_types.Accuracy(name='acc', precision=3),
    # # rw.score_types.SoftAccuracy(
        # # name='tfacc', score_matrix=true_false_score_matrix, precision=3),
# ]

score_types = [
    rw.score_types.SoftAccuracy(name='sacc', score_matrix=soft_score_matrix, precision=3),
    rw.score_types.Accuracy(name='acc'),
    #rw.score_types.NegativeLogLikelihood(name='nll'),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train-light.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test-light.csv'
    return _read_data(path, f_name)
