#importing libraries

import joblib
import numpy as np
import pandas as pd
from joblib import dump, load
import sklearn
import sklearn.preprocessing
from sklearn import metrics
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import sys
from utilities import  ConvertToLinearOutput, SequentialModel
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
# from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
# from keras.utils import to_categorical
# from tensorflow.keras.utils import get_file, plot_model
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from fastapi import FastAPI, File, HTTPException, Body, UploadFile



app = FastAPI()

trainColumns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass']

testColumns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass']

GlobalModel = any
@app.post("/Upload_DataSets")
async def Upload_DataSets(trainingFile:UploadFile = File(...) , testingFile:UploadFile = File(...)):
    with open("train.csv","wb") as f:
        content = await trainingFile.read()
        f.write(content) 
    with open("test.csv","wb") as f:
        content = await testingFile.read()
        f.write(content)
    return {'message' : "Files Uploaded Successfully"}
    

@app.post("/Train_CNN_Model")
async def Train_CNN_Model():
    global trainColumns
    global testColumns
    global columns
    global attack_type

    df_train = pd.read_csv('train.csv')
    df_train.columns = trainColumns
    combined_data = pd.concat([df_train])
    normal_records = combined_data[combined_data['subclass'] == 'normal'].sample(n=100, random_state=42)     # data samplingg

    combined_data = combined_data[combined_data['subclass'] != 'normal']
    combined_data = pd.concat([df_train,normal_records])

    combined_data = one_hot(combined_data,columns)
    tmp = combined_data.pop('subclass') 
    new_df_train = normalize(combined_data,combined_data.columns)
    new_df_train['class'] = tmp
    y_train = new_df_train['class']
    combined_data_X = new_df_train.drop('class', axis=1)
    if 1:
        train_X, test_X, train_y, test_y = train_test_split(combined_data_X, y_train, test_size=0.2, random_state=42)
        x_columns_train = new_df_train.columns.drop('class')
        x_train_array = train_X[x_columns_train].values
        x_columns_test = new_df_train.columns.drop('class')
        x_test_array = test_X[x_columns_test].values
        
        Model = SequentialModel() 
        Model.fit(x_train_array, train_y)  # Train the CNN Model
        
        pred = Model.predict(x_test_array)  # Perform predictions using the trained model
        
        score = metrics.accuracy_score(test_y, pred)  # Evaluate accuracy
        resp = joblib.dump(Model, 'CNNmodel.pkl')
        global GlobalModel
        GlobalModel = Model
        y_true_numeric = [AttackEncodings.get(label, 0) for label in test_y]

        y_pred_numeric =  [AttackEncodings.get(label, 0) for label in pred] 
        cm = confusion_matrix(y_true_numeric, y_pred_numeric, labels=list(AttackEncodings.values()))
        plot_confusion_matrix(cm, normalize    = False, target_names = list(AttackEncodings.values()),
                      title        = "Confusion Matrix")
    return {"message":"The Model is trained ","accuracyScore":score}


AttackEncodings = {'processtable': 1, 'land': 2, 'neptune': 3, 'satan': 4, 'warezmaster': 5, 'back': 6, 'buffer_overflow': 7, 'snmpgetattack': 8, 'warezclient': 9, 'teardrop': 10, 'mailbomb': 11, 'normal': 12, 'multihop': 13, 'ps': 14, 'httptunnel': 15, 'imap': 16, 'xsnoop': 17, 'rootkit': 18, 'loadmodule': 19, 'portsweep': 20, 'pod': 21, 'perl': 22, 'nmap': 23, 'guess_passwd': 24, 'spy': 25, 'ftp_write': 26, 'ipsweep': 27, 'snmpguess': 28, 'xlock': 29, 'smurf': 30, 'saint': 31, 'apache2': 32, 'mscan': 33}







@app.post("/ParseCSVString")
async def ParseCSVString(csvString :str):
    data_dict = dict(zip(trainColumns, csvString.split(',')))
    df = pd.DataFrame([data_dict])
    return df

@app.post("/Test_CNN_Model")
async def Test_CNN_Model(csvString :str):
    global trainColumns
    global testColumns
    global columns
    df_test = await ParseCSVString(csvString)
    print(df_test.shape)
    df_train = pd.read_csv('train.csv')
    df_train.columns = trainColumns
    df_test.columns = trainColumns
    print(df_train.head())
    combined_data = pd.concat([df_test,df_train])
    print(combined_data.head())

    combined_data = one_hot(combined_data,columns)
    tmp = combined_data.pop('subclass') 
    new_df_train = combined_data #normalize(combined_data,combined_data.columns)
    new_df_train['class'] = tmp
    y_train = new_df_train['class']
    combined_data_X = new_df_train.drop('class', axis=1)
    ActualTestTrue = new_df_train['class']      # the actyual tru data 
    Model = load('CNNmodel.pkl')

    if Model:
        train_X, test_X, train_y, test_y = train_test_split(combined_data_X, y_train, test_size=0.8, shuffle=False , random_state=None)
        x_columns_test = new_df_train.columns.drop('class')
        x_test_array = train_X[x_columns_test].values   
        
        pred = Model.predict(x_test_array)
        pred = ConvertToLinearOutput(pred[0] ,train_y.values[0])

    return {"Predicted": pred[0], "Actual":train_y.values[0] }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


#Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy()  # do not touch the original df
    for feature_name in cols:
        if df[feature_name].dtype in ['int64', 'float64']:  # Check if the column is numerical
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if max_value > min_value:
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

#One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop([each], axis=1)  
    return df
columns = ['protocol_type','service','flag']



confusionMatrixPath = "analytics/confusionMatrix.png"

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(confusionMatrixPath)
