import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier #集成学习中的随机森林
from sklearn.model_selection import train_test_split
from sklearn.metrics import * # 可以用该包导出模型的评价
from sklearn.model_selection import ShuffleSplit
from collections import defaultdict

# 读取csv文件数据
def read_csv(FilePath):
    data = pd.read_csv(FilePath, index_col=None, header=None)

    data_array = np.array(data)
    m, n = data_array.shape

    x_data = data_array[1:m, 2:n].astype(float)
    y_data = data_array[1:m, 1].astype(int)
    metabolite_ID_data = data_array[0, 2:n]

    return x_data, y_data, metabolite_ID_data

# 删除全为零的代谢物
def del_zero_metabolite(x_data):
    data_sum = x_data.sum(axis=0)
    del_index = np.where(data_sum == 0)[0]
    x_data =np.delete(x_data, del_index, axis=1)
    return x_data

# z_score标准化
def standardize_z_score(data_x):
    m, n = np.shape(data_x)  # m，n分别为样本数以及特征属性数
    for i in range(n):
        data = data_x[:, i].astype(float)
        data_mean = np.mean(data)
        data_std = np.std(data)
        for j in range(m):
            data_x[:, i][j] = (data[j] - data_mean) / data_std
    return data_x

# 模型评估
def model_evaluation(model, test_x, test_y):
    # Generate confusion matrix
    pred_y = model.predict(test_x)  # 验证集预测
    Matrix = confusion_matrix(test_y, pred_y)  # 混淆矩阵
    print(Matrix)

    # matrix = plot_confusion_matrix(rfc, test_x, test_y, cmap=plt.cm.Blues, normalize='true')
    matrix = plot_confusion_matrix(rfc, test_x, test_y, cmap=plt.cm.Blues)
    plt.title('Confusion matrix for our classifier')
    plt.show(matrix)
    plt.show()

    # 模型评估
    report = classification_report(test_y, pred_y)
    print(report)

# 特征重要性
def feature_selection(x_data, y_data, metabolite_ID_data):
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    scores = defaultdict(list)
    acc_score = []
    # crossvalidate the scores on a number of different random splits of the data
    idx = ShuffleSplit(n_splits=100, test_size=.3, random_state=1)
    for train_idx, test_idx in idx.split(x_data):
        X_train, X_test = x_data[train_idx], x_data[test_idx]
        Y_train, Y_test = y_data[train_idx], y_data[test_idx]
        rf.fit(X_train, Y_train)
        acc = rf.score(X_test, Y_test)
        acc_score.append(acc)
        for i in range(x_data.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = rf.score(X_t, Y_test)
            scores[metabolite_ID_data[i]].append((acc - shuff_acc) / acc)
    # print('Random Forest acc:{}'.format(np.mean(acc_score)))
    print("Features sorted by their score:")
    print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))

    sco = []
    for feat, score in scores.items():
        sco.append(np.mean(score))
    sco = np.array(sco)

    # 绘图画出重要性
    sorted_idx = sco.argsort()
    sorted_idx1 = sorted_idx[-40:-1]
    sorted_idx = np.append(sorted_idx1, sorted_idx[-1])
    plt.barh(metabolite_ID_data[sorted_idx], sco[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel('Feature')
    plt.show()

if __name__ == '__main__':
    filePath = "data.csv"  # 按实际修改
    data_x, data_y, datametabolite_ID = read_csv(filePath)
    data_x = del_zero_metabolite(data_x)
    data_x = standardize_z_score(data_x)

    X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=2023, stratify=data_y) #分层抽样
    # X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)

    rfc = RandomForestClassifier(n_estimators=100, random_state=2023)
    rfc = rfc.fit(X_train, Y_train)
    score_r = rfc.score(X_test, Y_test) # 准确率
    print('Random Forest accuracy:{}'.format(score_r)) # format是将分数转换放在{}中
    model_evaluation(rfc, X_test, Y_test)

    # 按准确率下降计算特征重要性并排序
    feature_selection(data_x, data_y, datametabolite_ID)
