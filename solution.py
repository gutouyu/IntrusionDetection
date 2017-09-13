# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

HIIDEN_LAYER_CEIL_NUM = 800
LEARN_RATE = 0.00001
# LEARN_RATE = 0.001
DROP = 0.5
TRAIN_STEP = 10000
VALIDATION_RATE = 0.25
BATCH_SIZE = 200
L2_RATE = 0.01

data = pd.read_csv('./kddcup.data_10_percent.csv')
dataTest = pd.read_csv('./corrected.csv')
dataForDDosTrain = data.loc[(data.classes == 'smurf.') |
                            (data.classes == 'normal.') |
                            (data.classes == 'back.') |
                            (data.classes == 'neptune.') |
                            (data.classes == 'land.') |
                            (data.classes == 'pod.') |
                            (data.classes == 'teardrop.')]

dataForDDosTest = dataTest.loc[(dataTest.classes == 'smurf.') |
                               (dataTest.classes == 'normal.') |
                               (dataTest.classes == 'back.') |
                               (dataTest.classes == 'neptune.') |
                               (dataTest.classes == 'land.') |
                               (dataTest.classes == 'pod.') |
                               (dataTest.classes == 'teardrop.')]


### get the ddos attack data

def preprcoessDataTcpBaseFeature(data):
    duration_mat = data['duration'].as_matrix().reshape(1, -1)
    scaler = StandardScaler().fit(duration_mat)
    data['dur_std'] = scaler.transform(duration_mat).reshape(-1)
    dummies_protocol_type = pd.get_dummies(data['protocol_type'], prefix='protocol_type')

    scaler2 = StandardScaler().fit(data['src_bytes'])
    data['src_bytes_std'] = scaler2.transform(data['src_bytes'])
    scaler3 = StandardScaler().fit(data['dst_bytes'])
    data['dst_bytes_std'] = scaler3.transform(data['dst_bytes'])
    data.drop(['duration', 'src_bytes', 'dst_bytes', 'service'], axis=1, inplace=True)

    ### preprocessing the  continuous data (std---z-zero) 连续数据 标准化

    data.loc[data.flag == 'REJ', 'flag'] = 2
    data.loc[data.flag == 'RSTO', 'flag'] = 3
    data.loc[data.flag == 'SF', 'flag'] = 0
    data.loc[data.flag == 'S0', 'flag'] = 1
    data.loc[data.flag == 'RSTR', 'flag'] = 4
    data.loc[(data.flag == 'S1') |
             (data.flag == 'S2') | (data.flag == 'S3')
             | (data.flag == 'OTH') | (data.flag == 'RSTOS0'), 'flag'] = 5

    dumies_flag = pd.get_dummies(data['flag'], prefix='flag')
    dumies_wrong_fragment = pd.get_dummies(data['wrong_fragment'], prefix='wrong_fragment')

    data.drop(['flag', 'wrong_fragment', 'protocol_type'], axis=1, inplace=True)
    return pd.concat([data, dumies_flag, dumies_wrong_fragment, dummies_protocol_type], axis=1)


### class-one data  classfier and preprocessing:one-hot encoding :tcp基本连接特征


def preprcoessDataTcpContentFeature(data):
    data.loc[data.hot > 2, 'hot'] = 3
    data.loc[(data.hot == 1) | (data.hot == 2), 'hot'] = 1
    dumies_hot = pd.get_dummies(data['hot'], prefix='hot')
    data.loc[data.num_compromised > 1, 'num_compromised'] = 2
    dummies_num_compromiesd = pd.get_dummies(data['num_compromised'], prefix='num_compromised')

    data.loc[(data.num_root <= 50) & (data.num_root > 0), 'num_root'] = 1
    data.loc[data.num_root > 50, 'num_root'] = 2
    dummies_num_root = pd.get_dummies(data['num_root'], prefix='num_root')
    data.drop(['hot', 'num_compromised', 'num_root'], axis=1, inplace=True)

    return pd.concat([data, dumies_hot, dummies_num_compromiesd, dummies_num_root], axis=1)


### class-two data 这一个类别的数据，分析数据，没有太多可以做的,（对ddos攻击而言）TCP连接的内容特征

def preprocessDataTimeForDataFlow(data):
    data.loc[(data.srv_count <= 25), 'srv_count'] = 0
    data.loc[(data.srv_count <= 100) & (data.srv_count > 25), 'srv_count'] = 1
    data.loc[(data.srv_count <= 150) & (data.srv_count > 100), 'srv_count'] = 2
    data.loc[(data.srv_count <= 350) & (data.srv_count > 150), 'srv_count'] = 3
    data.loc[data.srv_count > 350, 'srv_count'] = 4
    dumies_srv_count = pd.get_dummies(data['srv_count'], prefix='srv_count')
    data.drop(['srv_count'], axis=1, inplace=True)

    data.loc[data.serror_rate > 0.95, 'serror_rate'] = 1
    data.loc[
        (data.serror_rate <= 0.95) & (data.serror_rate > 0.5), 'serror_rate'] = 2
    data.loc[data.serror_rate <= 0.5, 'serror_rate'] = 3
    dumies_serror_rate = pd.get_dummies(data['serror_rate'], prefix='serror_rate')
    data.drop(['serror_rate'], axis=1, inplace=True)

    data.loc[data.srv_serror_rate >= 0.90, 'srv_serror_rate'] = 1
    data.loc[data.srv_serror_rate < 0.90, 'srv_serror_rate'] = 0
    dumies_srv_serror_rate = pd.get_dummies(data['srv_serror_rate'], prefix='srv_serror_rate')
    data.drop(['srv_serror_rate'], axis=1, inplace=True)

    data.loc[data.same_srv_rate == 1.00, 'same_srv_rate'] = 1
    data.loc[
        (data.same_srv_rate <= 0.15), 'same_srv_rate'] = 2
    data.loc[
        (data.same_srv_rate > 0.15) & (data.same_srv_rate < 1.00), 'same_srv_rate'] = 3

    dumies_same_srv_rate = pd.get_dummies(data['same_srv_rate'], prefix='same_srv_rate')
    data.drop(['same_srv_rate'], axis=1, inplace=True)
    return pd.concat([data, dumies_same_srv_rate, dumies_srv_count, dumies_serror_rate, dumies_srv_serror_rate], axis=1)


### class-three 面向数据流的时间的统计规则 有些数据离散化,有些数据标准化


def preprocessDataTimeForServer(data):
    scaler6 = StandardScaler().fit(data['dst_host_count'])
    data['dst_host_count_std'] = scaler6.transform(data['dst_host_count'])

    scaler7 = StandardScaler().fit(data['dst_host_srv_count'])
    data['dst_host_srv_count_std'] = scaler7.transform(data['dst_host_srv_count'])

    data.drop(['dst_host_count', 'dst_host_srv_count'], axis=1, inplace=True)


### class-four 面向主机的时间统计规则 简单数据分析过后，感觉没有什么太多的特征,将一些大值的数据连续化


def preprocessClasses(data):
    # data.loc[(data.classes != 'normal.'), 'classes'] = 0
    # data.loc[(data.classes == 'normal.'), 'classes'] = 1
    dumies_classes = pd.get_dummies(data['classes'], prefix='classes')
    data.drop(['classes'], axis=1, inplace=True)
    return dumies_classes


def dataPreprocess(data):
    data = preprcoessDataTcpBaseFeature(data)
    # print(data.columns)
    data = preprcoessDataTcpContentFeature(data)
    # print(data.columns)
    data = preprocessDataTimeForDataFlow(data)
    # print(data.columns)
    preprocessDataTimeForServer(data)
    # print(data.columns)
    return data


def weightMat(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype='float32'))


def biasMat(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype='float32'))


def accuracy(y, y_):
    return np.sum(np.argmax(y, axis=1) == np.argmax(y_, axis=1)) / (np.shape(y)[0] * 1.0)


def nextTrainXY():
    global train_x, train_y, total_train_num, train_index_start
    start = train_index_start
    end = start + BATCH_SIZE
    # print("start is %d" % start)
    # print("end is %d" % end)
    if end >= total_train_num:
        perm = np.arange(total_train_num)
        np.random.shuffle(perm)
        train_x = train_x[perm]
        train_y = train_y[perm]
        start = 0
        end = BATCH_SIZE
        train_index_start = 0
    train_index_start = train_index_start + BATCH_SIZE
    return train_x[start:end], train_y[start:end], start, end


def splitDataSet(data):
    data = data.sample(frac=1).reset_index(drop=True)
    totalLen = len(data.index)
    startVaildation = 0
    endVaildation = int(VALIDATION_RATE * totalLen) + 1
    dataValidation = data.iloc[startVaildation:endVaildation]
    dataTrain = data.iloc[endVaildation:totalLen]
    return dataValidation, dataTrain


def getTrainDataAndValiationData(data):
    dataSmurf = data.loc[(data.classes == 'smurf.')]
    dataSmurfVaildation, dataSmurfTrain = splitDataSet(dataSmurf)

    dataNeptune = data.loc[(data.classes == 'neptune.')]
    dataNeptuneVaildation, dataNeptuneTrain = splitDataSet(dataNeptune)

    dataLand = data.loc[(data.classes == 'land.')]
    dataLandVaildation, dataLandTrain = splitDataSet(dataLand)

    dataPod = data.loc[(data.classes == 'pod.')]
    dataPodVaildation, dataPodTrain = splitDataSet(dataPod)

    dataTeardrop = data.loc[(data.classes == 'teardrop.')]
    dataTeardropVaildation, dataTeardropTrain = splitDataSet(dataTeardrop)

    dataBack = data.loc[(data.classes == 'back.')]
    dataBackVaildation, dataBackTrain = splitDataSet(dataBack)

    dataNormal = data.loc[(data.classes == 'normal.')]
    dataNormalVaildation, dataNormalTrain = splitDataSet(dataNormal)

    dataVailation = pd.concat([dataNormalVaildation, dataSmurfVaildation,
                               dataNeptuneVaildation, dataLandVaildation,
                               dataPodVaildation, dataTeardropVaildation, dataBackVaildation], axis=0).reset_index(
        drop=True)
    dataTrain = pd.concat([dataNormalTrain, dataBackTrain,
                           dataTeardropTrain, dataLandTrain,
                           dataPodTrain, dataNeptuneTrain, dataSmurfTrain], axis=0).reset_index(drop=True)
    return dataVailation, dataTrain


dataForDDosTrain = dataPreprocess(dataForDDosTrain)
dataForDDosValition, dataForDDosTrain = getTrainDataAndValiationData(dataForDDosTrain)
dataForDDosTest = dataPreprocess(dataForDDosTest)
classVaildation = preprocessClasses(dataForDDosValition)
classesTrain = preprocessClasses(dataForDDosTrain)
classesTest = preprocessClasses(dataForDDosTest)

### data train and data test and classes

train_x = dataForDDosTrain.as_matrix().astype(np.float32)
train_y = classesTrain.as_matrix().astype(np.float32)

test_x = dataForDDosTest.as_matrix().astype(np.float32)
test_y = classesTest.as_matrix().astype(np.float32)

vaildation_x = dataForDDosValition.as_matrix().astype(np.float32)
vaildation_y = classVaildation.as_matrix().astype(np.float32)

### get data for trainx,y and test x,y

total_train_num = np.shape(train_x)[0]
feature_num = np.shape(train_x)[1]
label_count = np.shape(train_y)[1]

x = tf.placeholder('float32', shape=[None, 64])
y = tf.placeholder('float32', shape=[None, label_count])
## x [batchSize,64],y [batchSize,64]
w_input_hidden = weightMat([feature_num, HIIDEN_LAYER_CEIL_NUM])
## [64,512]
b_input_hiiden = biasMat([HIIDEN_LAYER_CEIL_NUM])
##  [512]
hidden_layer = tf.nn.relu(tf.matmul(x, w_input_hidden) + b_input_hiiden)

hidden_layer_dropout = tf.nn.dropout(hidden_layer, keep_prob=DROP)
## dropout 0.5

w_hiden_output = weightMat([HIIDEN_LAYER_CEIL_NUM, label_count])
## [512,10]
b_hidden_output = biasMat([label_count])
## [512]
y_output = tf.matmul(hidden_layer_dropout, w_hiden_output) + b_hidden_output
y_ = tf.nn.softmax(y_output)

loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-8, 1.0))))
l2Loss = tf.nn.l2_loss(w_hiden_output) + tf.nn.l2_loss(w_input_hidden)
loss = loss_function + L2_RATE * l2Loss
trainer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss)

## define bp-ANN network and [batchSize,64]->[batchSize,512]->[batchSize,7] 训练模型


x_in = tf.placeholder('float32', shape=[None, 64])

hidden_layer_testData = tf.nn.relu(tf.matmul(x_in, w_input_hidden) + b_input_hiiden)
prediction_Data = tf.nn.softmax(tf.matmul(hidden_layer_testData, w_hiden_output) + b_hidden_output)

train_index_start = total_train_num

for i in range(10):
    nextTrainXY()
    train_index_start = total_train_num

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    for i in range(TRAIN_STEP):
        batchX, batchY, start, end = nextTrainXY()
        t, trainRetY, loss, y_out = session.run([trainer, y_, loss_function, y_output],
                                                feed_dict={x: batchX, y: batchY})
        if i % 100 == 0:
            print("the start is {0} end is {1} \n".format(start, end))
            print("the train step is %d \n " % i)
            print("the loss is %f " % loss)
            print("the accuracy of train data is %f " % accuracy(trainRetY, batchY))
            print(
                "the accuracy of vaildation data is %f" % accuracy(prediction_Data.eval(feed_dict={x_in: vaildation_x}),
                                                                   vaildation_y))
            # print("the accuracy of test data is %f " % accuracy(
            #     prediction_Data.eval(feed_dict={x_in: test_x}), test_y))

    print("train finish")
    print("the  accuracy of final model on test data is %f " % accuracy(
        prediction_Data.eval(feed_dict={x_in: test_x}), test_y))
