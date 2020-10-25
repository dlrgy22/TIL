import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib import pyplot as plt

import numpy as np

import pandas as pd
from sklearn.ensemble import IsolationForest

def get_data(path):

    df = pd.read_csv(path)

    # name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    # 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    # 'aa', 'ab', 'ac', 'ad', 'ae']

    # for element in name:
    #     df[element] = preprocessing.scale(np.log(df[element] + 1))
    # print(df.describe())

    if path == './trainset.csv':
        y_data = df.values[:, 0]
        x_data = df.values[:, 1:]
    else:
        x_data = df.values[:, :]



    if path == './trainset.csv':
        return x_data, y_data, df.columns[1:]
    else:
        return x_data




def split_val_data(x_data, y_data):
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=1234)
    return x_train, x_val, y_train, y_val

def std_scale(x_data, test_data = []):
    standardScaler = StandardScaler()
    standardScaler.fit(x_data)
    scaled_data = standardScaler.transform(x_data)

    if len(test_data) != 0:
        test_data = standardScaler.transform(test_data)
        return scaled_data, test_data

    return scaled_data

def robuster_scale(x_data, test_data = []):
    robuster = RobustScaler()
    robuster.fit(x_data)
    scaled_data = robuster.transform(x_data)

    if len(test_data) != 0:
        test_data = robuster.transform(test_data)
        return scaled_data, test_data

    return scaled_data

def one_hot(y_data):
    y_data[np.where(y_data == 'PH')] = 0
    y_data[np.where(y_data == 'HI')] = 1
    y_data[np.where(y_data == 'EL')] = 2
    y_data[np.where(y_data == 'CO')] = 3
    y_data[np.where(y_data == 'GR')] = 4
    y_data[np.where(y_data == 'MI')] = 5
    print(len(np.where(y_data == 0)[0]))
    print(len(np.where(y_data == 1)[0]))
    print(len(np.where(y_data == 2)[0]))
    print(len(np.where(y_data == 3)[0]))
    print(len(np.where(y_data == 4)[0]))
    print(len(np.where(y_data == 5)[0]))
    y_one_hot = to_categorical(y_data, num_classes=6)
    y_one_hot.reshape((y_one_hot.shape[0], y_one_hot.shape[1], 1))
    return y_one_hot

def over_sampling(x_train, y_train):
    smote = SMOTE(random_state=1234)
    x_train, y_train = smote.fit_sample(x_train, y_train)
    return x_train, y_train

def save_submission(pred):
    data = []
    for i in range(len(pred)):
        if pred[i] == 0:
            data.append(['PH'])
        elif pred[i] == 1:
            data.append(['HI'])
        elif pred[i] == 2:
            data.append(['EL'])
        elif pred[i] == 3:
            data.append(['CO'])
        elif pred[i] == 4:
            data.append(['GR'])
        elif pred[i] == 5:
            data.append(['MI'])
    file_name = "32164193_정익효.csv"
    np.savetxt(file_name, data, fmt="%s", delimiter=",")


def save_submission2(pred):
    file_name = "32164193_정익효.csv"
    np.savetxt(file_name, pred, fmt="%s", delimiter=",")


def pca(x_data, components, x_test = []):
    P = PCA(n_components=components)
    train = P.fit_transform(x_data)
    if len(x_test) != 0:
        test = P.transform(x_test)

    train_df = pd.DataFrame(data=train, columns=[str(i) for i in range(components)])
    if len(x_test) == 0:
        return train_df.values
    else:
        test_df = pd.DataFrame(data = test, columns = [str(i) for i in range(components)])
        return train_df.values, test_df.values



def anomaly(x_data, y_data):

    PH = []
    HI = []
    EL = []
    CO = []
    GR = []
    MI = []

    for i in range(len(x_data)):
        if y_data[i] == 'PH':
            PH.append(x_data[i])
        elif y_data[i] == 'HI':
            HI.append(x_data[i])
        elif y_data[i] == 'EL':
            EL.append(x_data[i])
        elif y_data[i] == 'CO':
            CO.append(x_data[i])
        elif y_data[i] == 'GR':
            GR.append(x_data[i])
        elif y_data[i] == 'MI':
            MI.append(x_data[i])
    PH = np.array(PH)
    HI = np.array(HI)
    EL = np.array(EL)
    CO = np.array(CO)
    GR = np.array(GR)
    MI = np.array(MI)

    clf = IsolationForest(n_estimators=300,
                          max_samples=50,
                          contamination=float(0.05),
                          max_features=1.0,
                          bootstrap=False,
                          n_jobs=-1,
                          random_state=1234,
                          verbose=0,
                          behaviour='deprecated')

    clf.fit(PH)

    y_data = []

    pred = clf.predict(PH)
    PH = PH[np.where(pred == 1)]

    clf.fit(HI)
    pred = clf.predict(HI)
    HI = HI[np.where(pred == 1)]

    clf.fit(EL)
    pred = clf.predict(EL)
    EL = EL[np.where(pred == 1)]

    clf.fit(CO)
    pred = clf.predict(CO)
    CO = CO[np.where(pred == 1)]

    clf.fit(GR)
    pred = clf.predict(GR)
    GR = GR[np.where(pred == 1)]

    clf.fit(MI)
    pred = clf.predict(MI)
    MI = MI[np.where(pred == 1)]

    name = [[PH, 'PH'], [HI, 'HI'], [EL, 'EL'], [CO, 'CO'], [GR, 'GR'], [MI, 'MI']]

    for c, n in name:
        for i in range(len(c)):
            y_data.append(n)

    x_data = np.concatenate((PH, HI, EL, CO, GR, MI), axis=0)
    y_data = np.array(y_data)

    return x_data, y_data


path = './trainset.csv'
x_data, y_data, name= get_data(path)
anomaly(x_data, y_data)

# x_data = std_scale(x_data)
#
# x_data = pca(x_data)
# fig, ax = plt.subplots(1,1,figsize=(5,5))

# for i in range(len(x_data)):
#     if y_data[i] == 'PH':
#         ax.scatter(x_data[i][0], x_data[i][1], color='r', s=4)
#     elif y_data[i] == 'HI':
#         ax.scatter(x_data[i][0], x_data[i][1], color='g',s=4)
#     elif y_data[i] == 'EL':
#         ax.scatter(x_data[i][0], x_data[i][1], color='darkred',s=4)
#     elif y_data[i] == 'CO':
#         ax.scatter(x_data[i][0], x_data[i][1], color='dimgray',s=4)
#     elif y_data[i] == 'GR':
#         ax.scatter(x_data[i][0], x_data[i][1], color='aqua',s=4)
#     elif y_data[i] == 'MI':
#         ax.scatter(x_data[i][0], x_data[i][1], color='orange',s=4)

