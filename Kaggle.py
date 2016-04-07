import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
import requests, zipfile, StringIO
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics

os.chdir('~/Santander-Kaggle')

def LoadData():
	train = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	kaggle_data={"train":train,"test":test}
	return kaggle_data

kaggle_data=LoadData()

clf = linear_model.LogisticRegression(penalty="l1",class_weight=None)

clf.fit(kaggle_data["train"].drop(["ID","TARGET"],axis=1).values,kaggle_data["train"]['TARGET'].values)

train_preds = clf.predict_proba(kaggle_data["train"].drop(["ID","TARGET"],axis=1).values)
calibration_curve(kaggle_data["train"]['TARGET'].values, train_preds[:,1], n_bins=10)
print metrics.roc_auc_score(kaggle_data["train"]['TARGET'].values, train_preds[:, 1])


test_probs = clf.predict_proba(kaggle_data["test"].drop(["ID"],axis=1).values)
test_probs=pd.DataFrame(test_probs)

test_final=pd.DataFrame({"ID":kaggle_data["test"]["ID"].values,"TARGET":test_probs.ix[:,1].values})
test_final.to_csv("test_final.csv",index=False)