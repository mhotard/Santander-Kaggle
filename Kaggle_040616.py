import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
import requests, zipfile, StringIO
import xgboost as xgb
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn import metrics
from sklearn.decomposition import PCA


os.chdir('/Users/michaelhotard/Santander-Kaggle')


def LoadData():
	train = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	kaggle_data={"train":train,"test":test}
	return kaggle_data

kaggle_data=LoadData()

############PCA analysis
#create the data from the training data, dropping ID and TARGET
kaggle_data_clean = kaggle_data["train"].drop(["ID","TARGET"],axis=1).values

#set the components to get in our pca model
pca = PCA(n_components=10)

#fit the data to our pca model
pca.fit(kaggle_data_clean)

#creating our eigenvectors for the training data
var1 = pca.transform(kaggle_data_clean)

#adding the eigenvectors to our training data
kaggle_w_pca = np.concatenate((kaggle_data_clean, var1), axis=1)


#doing some xgboost which builds trees on residuals
#creating a DMatrix with the new pca data, #not sure what label is
dtrain = xgb.DMatrix(data = kaggle_w_pca, label=kaggle_data["train"]["TARGET"])

#setting parameters for the xgboost, max-depth is trees, objective is type of model
param = {'max-depth':9, 'eta':.1, 'silent':0, 'objective':'reg:logistic'}
#this is trying to find the cutoff for when we lose value
history1 = xgb.cv(param, dtrain, num_boost_round=3000, nfold=4, show_progress=True, metrics={"auc"})

#running a training set on the training matrix
pred1 = xgb.train(param, dtrain, 88)

#predicting on the test data
#getting the pca for the test data. first getting test data
kaggle_data_clean_pred = kaggle_data["test"].drop(["ID"],axis=1).values
#getting the eigenvectors for the test data (uses the same pca as before?)
var2 = pca.transform(kaggle_data_clean_pred)

#adding the new variables in
kaggle_w_pca_pred = np.concatenate((kaggle_data_clean_pred, var2), axis=1)

#creating a DMatrix for the test data
dtest = xgb.DMatrix(data=kaggle_w_pca_pred)

#using pred1 to make predictions on the test data
preds = pred1.predict(dtest)

#creating a data frame for the target
#column one is ID from test ID column, column two is labeled Target from the preds variable)
upload_pca = pd.DataFrame({"ID":kaggle_data["test"]["ID"].values,"TARGET":preds})
#upload the data to a csv
upload_pca.to_csv("upload_pca.csv", index=False)

#trying our xgboost without the pca
kaggle_data_clean = kaggle_data["train"].drop(["ID","TARGET"],axis=1).values
dtrain2 = xgb.DMatrix(data = kaggle_data_clean, label=kaggle_data["train"]["TARGET"])
param = {'max-depth':9, 'eta':.1, 'silent':0, 'objective':'reg:logistic'}

pred2 = xgb.train(param, dtrain2, 88)

kaggle_data_clean_pred = kaggle_data["test"].drop(["ID"],axis=1).values
dtest2 = xgb.DMatrix(data=kaggle_data_clean_pred)

preds2 = pred2.predict(dtest2)

upload_no_pca = pd.DataFrame({"ID":kaggle_data["test"]["ID"].values, "TARGET":preds2})
upload_no_pca.to_csv("upload_no_pca.csv", index=False)





upload_pca = pd.DataFrame({"ID":kaggle_data["test"]["ID"].values,"TARGET":preds})
upload_pca.to_csv("upload_pca.csv", index=False)


clf = linear_model.LogisticRegression(penalty="l1",class_weight=None)

clf.fit(kaggle_data["train"].drop(["ID","TARGET"],axis=1).values,kaggle_data["train"]['TARGET'].values)

train_preds = clf.predict_proba(kaggle_data["train"].drop(["ID","TARGET"],axis=1).values)
calibration_curve(kaggle_data["train"]['TARGET'].values, train_preds[:,1], n_bins=10)
print metrics.roc_auc_score(kaggle_data["train"]['TARGET'].values, train_preds[:, 1])


test_probs = clf.predict_proba(kaggle_data["test"].drop(["ID"],axis=1).values)
test_probs=pd.DataFrame(test_probs)

test_final=pd.DataFrame({"ID":kaggle_data["test"]["ID"].values,"TARGET":test_probs.ix[:,1].values})
test_final.to_csv("test_final.csv",index=False)