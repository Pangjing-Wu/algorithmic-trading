#svm不是随机森林
import pandas as pd
from sklearn import svm,preprocessing
import matplotlib.pyplot as plt
origDf=pd.read_csv('D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\TI.csv',encoding='gbk')
df=origDf#[['Close', 'Low','Open' ,'Vol','Date']]
#diff列表示本日和上日收盘价的差
df['diff'] = df["price"]-df["price"].shift(1)
df['diff'].fillna(0, inplace = True)
#up列表示本日是否上涨,1表示涨，0表示跌
df['up'] = df['diff']
df['up'][df['diff']>0] = 1
df['up'][df['diff']==0] = 0
df['up'][df['diff']<0] = -1
#预测值暂且初始化为0
df['predictForUp'] = 0
#训练集的特征值和目标值
target = df['up']
length=len(df)
trainNum=int(length*0.8)
predictNum=length-trainNum
#选择指定列作为特征列
feature=df[['ROC_1','ROC_2','ROC_3','ROC_4','ROC_5', 'SD', 'PSL','IV']]
 #标准化处理特征值
feature=preprocessing.scale(feature)
print(feature)
#训练集的特征值和目标值
featureTrain=feature[1:trainNum-1]
targetTrain=target[1:trainNum-1]
svmTool = svm.SVC(kernel='linear')
svmTool.fit(featureTrain,targetTrain)
predictedIndex=trainNum
print(predictedIndex)
#逐行预测测试集
df.loc[predictedIndex,'predictForUp']=1
print(df.loc[predictedIndex,'predictForUp'])
while predictedIndex<length:
    testFeature=feature[predictedIndex:predictedIndex+1]           
    predictForUp=svmTool.predict(testFeature)
    df.loc[predictedIndex,'predictForUp']=predictForUp
    predictedIndex = predictedIndex+1
#该对象只包含预测数据，即只包含测试集
dfWithPredicted = df[trainNum:1420]
#开始绘图，创建两个子图
figure = plt.figure()
#创建子图
(axprice, axUpOrDown) = figure.subplots(2, sharex=True)
dfWithPredicted['price'].plot(ax=axprice)
dfWithPredicted['predictForUp'].plot(ax=axUpOrDown,color="red", label='Predicted Data')
dfWithPredicted['up'].plot(ax=axUpOrDown,color="blue",label='Real Data')
plt.legend(loc='best') #绘制图例
#设置x轴坐标标签和旋转角度
major_index=dfWithPredicted.index[dfWithPredicted.index%2==0]
major_xtics=dfWithPredicted['start_time'][dfWithPredicted.index%2==0]
plt.xticks(major_index,major_xtics)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.title("通过SVM预测涨跌情况")
plt.rcParams['font.sans-serif']=['SimHei']
plt.show()
df.to_csv(r'D:\\python\my程序\\2020_8_6\\algorithmic-trading-master\\strategies\\market-making\\svm.csv')