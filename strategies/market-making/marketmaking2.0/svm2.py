#svm
import pandas as pd
import numpy as np
from sklearn import svm,preprocessing,datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
class SVM(object):
    def __init__(self,csv1,csv2,csv3):
        self.csv_1=csv1
        self.csv_2 = csv2
        self.csv_3 = csv3

    def svm3days(self,test_csv):
        out_csv = pd.concat([self.csv_1,self.csv_2,self.csv_3], axis=0,ignore_index=False)
        # diff列表示本日和上日收盘价的差
        df = out_csv[['start_time', 'end_time', 'averageprice', 'maxprice', 'minprice', 'dprice', 'ROC_1', 'ROC_2', 'ROC_3','ROC_4', 'ROC_5', 'SD', 'SD2', 'D', 'PSL', 'IV']]
        df.reset_index(drop=True, inplace=True)
        df['diff'] = None
        df['diff'] = df['averageprice'] - df['averageprice'].shift(1)
        df['diff'].fillna(0, inplace=True)
        # up列表示本日是否上涨,1表示涨，0表示跌
        len1 = len(df)
        df_v = df['diff'].values
        df_v = df_v.flatten()
        df_new = np.zeros(len1)
        df_new[len1 - 1] = 0
        # df_new[0:len] = df_v[0:len]
        df_new[0:len1 - 2] = df_v[1:len1 - 1]
        df['up'] = df_new
        df['up'][df['up'] >= 0.005] = 1
        df['up'][df['up'] <= -0.005] = -1
        for iii in range(0, len1):
            if df.loc[iii, 'up'] != 1 and df.loc[iii, 'up'] != -1:
                df.loc[iii, 'up'] = 0
        # 预测值暂且初始化为0
        # df['predictForUp'] = 0
        # 训练集的特征值和目标值
        target = df['up']
        # 选择指定列作为特征列
        df.dropna(inplace=True)
        feature = df[['start_time', 'end_time', 'averageprice', 'maxprice', 'minprice', 'dprice', 'ROC_1', 'ROC_2', 'ROC_3', 'ROC_4', 'ROC_5', 'SD', 'SD2', 'D', 'PSL', 'IV']]
        # 标准化处理特征值
        feature = preprocessing.scale(feature)
        # 训练集的特征值和目标值
        featureTrain = feature[1:len(df)-1]
        targetTrain = target[1:len(df)-1]
       # svmTool = svm.SVC(kernel='linear')

        from sklearn.model_selection import GridSearchCV
        svmTool= GridSearchCV(SVC(), param_grid={"C": [0.1,0.5,1,2,3,4,10], "gamma": [1, 0.1, 0.02,0.01,0.001]}, cv=5)
        svmTool.fit(featureTrain, targetTrain)
        print("The best parameters are %s with a score of %0.2f"
              % (svmTool.best_params_, svmTool.best_score_))
        predictedIndex = 0
        # 逐行预测测试集
        len2=len(test_csv)
        test_csv['diff'] = None
        test_csv['diff'] = test_csv['averageprice'] - test_csv['averageprice'].shift(1)
        test_csv['diff'].fillna(0, inplace=True)
        test_v = test_csv['diff'].values
        test_v = test_v.flatten()
        test_new = np.zeros(len2)
        test_new[len2 - 1] = 0
        # df_new[0:len] = df_v[0:len]
        test_new[0:len2 - 2] = test_v[1:len2 - 1]
        test_csv['up'] = test_new
        test_csv['up'][test_csv['up'] >= 0.005] = 1
        test_csv['up'][test_csv['up'] <= -0.005] = -1
        for i in range(0, len2):

            if test_csv.loc[i, 'up'] != 1 and test_csv.loc[i, 'up'] != -1:
                test_csv.loc[i, 'up'] = 0

        while predictedIndex < len2:
            testFeature = feature[predictedIndex:predictedIndex + 1]
            predictForUp = svmTool.predict(testFeature)
            test_csv.loc[predictedIndex, 'predictForUp'] = predictForUp
            predictedIndex = predictedIndex + 1

        sum_right = 0
        for ii in range(0, len2):
            if test_csv['up'][ii] == test_csv['predictForUp'][ii]:
                sum_right = sum_right + 1
        rate = sum_right /len2
        print(rate)
        return test_csv,rate