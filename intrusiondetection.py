
# coding: utf-8

# # Deep Learning
# 
# ## preprocessing training dataset

import os
from MLP import MLP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def read_data():
    input_file_dir = "../datasets"
    train_file_name = "kddcup.data_10_percent.txt"
    test_file_name = "corrected.txt"
    header_file_name = "header.txt"
    train_files = os.path.join(input_file_dir, train_file_name)
    test_files = os.path.join(input_file_dir, test_file_name)
    header_files = os.path.join(input_file_dir, header_file_name)
    with open(header_files, 'r') as f:
        header = f.readline().strip().split(',')
    train_dataset = pd.read_csv(train_files)
    test_dataset = pd.read_csv(test_files)
    train_dataset.columns = header
    test_dataset.columns = header
    return train_dataset, test_dataset


def labels_map(label):
    label = str(label).split('.')[0]
    if label == 'normal':
        return 0
    if label in ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']: #PROBE
        return 1
    if label in ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm']: #DOS
        return 2
    if label in ['buffer_overflow', 'httptunnel', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']: #U2R
        return 3
    if label in ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop']: #R2L
        return 4

def filter_labels(dataset):
    dataset['labels'] = dataset['labels'].apply(labels_map)

    #只保留DOS 和 Normal
    dataset = dataset[(dataset['labels']==0) |
                      (dataset['labels']==2) |
                      (dataset['labels']==1) |
                      (dataset['labels']==3)   ]

    return dataset

def split_valid_from_train(train_dataset, valid_size):
    # Method 1
    train_dataset, valid_dataset, _, _ = train_test_split(train_dataset, train_dataset['labels'], test_size=valid_size, random_state=42)

    # Method 2
    #获取验证集
    # val_frac=0.25
    # valid_dataset_neg = train_dataset[(train_dataset['labels']==0)].sample(frac=val_frac)
    # valid_dataset_pos = train_dataset[(train_dataset['labels']==2)].sample(frac=val_frac)
    # valid_dataset = pd.concat([valid_dataset_neg, valid_dataset_pos], axis=0)

    # #train_dataset中分离出valid_dataset
    # train_dataset = train_dataset.select(lambda x: x not in valid_dataset.index, axis=0)

    # train_dataset_size = train_dataset.shape[0]
    # valid_dataset_size = valid_dataset.shape[0]
    # test_dataset_size = test_dataset.shape[0]
    # print 'Train dataset: ', train_dataset_size
    # print 'Valid dataset: ', valid_dataset_size
    # print 'Test  dataset: ', test_dataset_size
    return train_dataset, valid_dataset

def combine_train_valid_test(trainDF, validDF, testDF):
    all = pd.concat([trainDF, validDF, testDF], axis=0)
    return all, (trainDF.shape[0], validDF.shape[0], testDF.shape[0])

def process_data_features(all):

    # 独热编码 labels
    labels_dummies = pd.get_dummies(all['labels'], prefix='label')
    all = pd.concat([all,labels_dummies], axis=1)
    all = all.drop(['labels'], axis=1)

    # 独热编码 protocol_type
    protocal_type_dummies = pd.get_dummies(all.protocol_type, prefix='protocol_type')
    all = pd.concat([all, protocal_type_dummies], axis=1)
    all = all.drop(['protocol_type'], axis=1)

    # 独热编码 flag
    flag_dummies = pd.get_dummies(all.flag, prefix='flag')
    all = pd.concat([all, flag_dummies], axis=1)
    all = all.drop(['flag'], axis=1)

    # 独热编码 Service 共有66个 暂时先去掉
    # all.service.value_counts()
    # service_dummies = pd.get_dummies(all.service, prefix='service')
    # all = pd.concat([all, service_dummies], axis=1)
    all = all.drop(['service'], axis=1)

    # 去中心化 src_bytes, dst_bytes
    all['src_bytes_norm'] = all.src_bytes - all.src_bytes.mean()
    all['dst_bytes_norm'] = all.dst_bytes - all.dst_bytes.mean()
    all = all.drop(['src_bytes'], axis=1)
    all = all.drop(['dst_bytes'], axis=1)

    return all.astype('float')

def recover_data_after_process_features(comb, num_comb, labels_list=[0,1,2,3]):
    #分离出Train Valid Test
    train_dataset_size, valid_dataset_size, test_dataset_size = num_comb
    sub_train_dataset = comb.iloc[:train_dataset_size, :].sample(frac=1)
    sub_valid_dataset = comb.iloc[train_dataset_size: train_dataset_size+valid_dataset_size, :].sample(frac=1)
    sub_test_dataset = comb.iloc[train_dataset_size+valid_dataset_size:, :].sample(frac=1)
    # 分离出 label
    total_labels = ['label_%d' % i for i in labels_list]
    sub_train_labels = sub_train_dataset[total_labels]
    sub_valid_labels = sub_valid_dataset[total_labels]
    sub_test_labels = sub_test_dataset[total_labels]
    sub_train_dataset.drop(total_labels, axis=1, inplace=True)
    sub_valid_dataset.drop(total_labels, axis=1, inplace=True)
    sub_test_dataset.drop(total_labels, axis=1, inplace=True)
    data = {
        'X_train': sub_train_dataset.as_matrix(),
        'y_train': sub_train_labels.as_matrix(),
        'X_val': sub_valid_dataset.as_matrix(),
        'y_val': sub_valid_labels.as_matrix(),
        'X_test': sub_test_dataset.as_matrix(),
        'y_test': sub_test_labels.as_matrix()
    }
    for k, v in data.iteritems():
        print k, v.shape
    return data



def analysis_plot_loss_and_accuracy(model):

    print('Mean Accuracy( -top5 ) of train, valid, test: '), np.mean(model.train_acc_history[-5:]), np.mean(model.val_acc_history[-5:]), np.mean(model.test_acc_history[-5])
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(range(len(model.loss_history)), model.loss_history, '-o')
    ax.title('Loss')
    plt.plot()

    plt_len = len(model.train_acc_history)
    plt.subplot(2,1,2)
    plt.title('Accuracy')
    plt.plot(model.train_acc_history, '-o', label='train')
    plt.plot(model.test_acc_history, '-o', label='test')
    plt.plot(model.val_acc_history, '-o', label='valid')
    # plt.plot([90] * plt_len, 'k--')
    plt.xlabel('Iteration')
    plt.legend(loc='lower right')
    # plt.gcf().set_size_inches(12,8)
    plt.show()

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def analysis_confusion_matrix(data, model, header, verbose=True):
    y_pred = model.predict(data['X_test'])
    m = len(header)
    y_true = np.argmax(data['y_test'], axis=1)
    y_predIdx = np.argmax(y_pred, axis=1)
    mat = np.mat(np.zeros((m,m)))
    for real, pred in zip(y_true,y_predIdx):
        mat[real,pred] = mat[real,pred] + 1
    confMatrix = pd.DataFrame(data=mat, index= header, columns=header, dtype=int)
    if verbose:
        print ('Test Accuracy: %.f', accuracy(y_pred, data['y_test']))
        print confMatrix
    return confMatrix

if __name__ == '__main__':

    # Prepare Data
    trainDF, testDF = read_data()

    trainDF = filter_labels(trainDF)
    testDF = filter_labels(testDF)

    trainDF, validDF = split_valid_from_train(trainDF, 0.25)

    combine, num_combine = combine_train_valid_test(trainDF, validDF, testDF)

    combine = process_data_features(combine)

    data = recover_data_after_process_features(combine, num_combine, [0,1,2,3])

    # Model
    input_dim = data['X_train'].shape[1]
    output_dim = data['y_train'].shape[1]
    model = MLP(data, input_dim, [512],output_dim,
                learning_rate=1e-6, #1e-6
                dropout_prob=0.0,
                l2_strength=0.0,
                batch_size=200,
                num_epochs=1,
                print_every=200,
                verbose=True)

    model.train()

    # Analysis
    analysis_confusion_matrix(data, model, ['normal', 'probe', 'dos', 'u2r', 'r2l'])
    analysis_plot_loss_and_accuracy(model)
