# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:11:41 2019

@author: XMM
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.metrics import mean_absolute_error
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
import argparse

def parse_args():

	parser = argparse.ArgumentParser(description="Run APR.")

	parser.add_argument('--input_trainFile', nargs='?', default='features/train_features_k25_cas50.csv',
	                    help='Input train cascadaes features file path')

	parser.add_argument('--input_testFile', nargs='?', default='features/validation_features_k25_cas10.csv',
	                    help='Input test cascadaes features file path')

	parser.add_argument('--output', nargs='?', default='../dataset/features/result.csv',
	                    help='Features path')
	parser.add_argument('--k', type=int, default=25,
	                    help='Number of early adopters. Default is 25.')
	parser.add_argument('--myModel', type=int, default=1,
	                    help='1 denotes user MyModel featuers. 0 denotes APR features.')
	return parser.parse_args()


class Log_reg():
    def __init__(self):
        self.log_reg = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
        self.ss = StandardScaler()
        
    def read_features_csv(self, file, My_Model):
        #需要注意第一行数据未读取
        df = pd.read_csv(file)
        columns_size = df.columns.size
        if My_Model == 0:
            X = df.iloc[:,1: columns_size - args.k - 2]
        else:
            X = pd.concat([df.iloc[:,1: columns_size - 2* args.k -2] ,df.iloc[:, columns_size - args.k -2: -2]], axis=1)
        Y = df.iloc[:,-1]
#        print(X.head())
#        print(X,Y)
        return X, Y
    #    print(df.head())
    
    def fit_logistic_regression(self, train_file):
        x, y = self.read_features_csv(train_file, args.myModel)           
#        x = self.ss.fit_transform(x)
        self.log_reg.fit(x, y)

    
    def predict_test(self, test_file):
        x, y = self.read_features_csv(test_file, args.myModel)
#        x = self.ss.fit_transform(x)
        y_pre = self.log_reg.predict(x)
        accuracy = accuracy_score(y, y_pre)
        precision = precision_score(y,y_pre)
        recall = recall_score(y, y_pre)
        F1 = f1_score(y, y_pre)
        print("accuracy, precision, recall, F1:", accuracy, precision, recall, F1)
        
        
def main(args):
    log_reg = Log_reg()
    log_reg.fit_logistic_regression(args.input_trainFile)
    log_reg.predict_test(args.input_testFile)
    
if __name__ == "__main__":
	args = parse_args()
	main(args)
