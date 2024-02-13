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
import keras
from utilities import SequentialModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.utils import to_categorical
# from tensorflow.keras.utils import get_file, plot_model
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from fastapi import FastAPI, File, HTTPException, Body, UploadFile



app = FastAPI()



@app.post("/Upload_DataSets")
async def Upload_DataSets(trainingFile:UploadFile = File(...) , testingFile:UploadFile = File(...)):
    with open("train.csv","wb") as f:
        content = await trainingFile.read()
        f.write(content) 
    with open("test.csv","wb") as f:
        content = await trainingFile.read()
        f.write(content)
    return {'message' : "Files Uploaded Successfully"}
    

@app.post("/Train_CNN_Model")
async def Train_CNN_Model():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_test.columns = testColumns
    df_train.columns = trainColumns
    combined_data = pd.concat([df_train, df_test])
    combined_data = one_hot(combined_data,columns)
    tmp = combined_data.pop('subclass') 
    new_df_train = normalize(combined_data,combined_data.columns)
    new_df_train['class'] = tmp
    y_train = new_df_train['class']
    combined_data_X = new_df_train.drop('class', axis=1)
    # kfold = StratifiedKFold(n_splits=2,shuffle=True,random_state=42)
    # kfold.get_n_splits(combined_data_X,y_train)
    # batch_size = 32
    model = Sequential()
    model.add(Convolution1D(64, kernel_size=122, activation="relu",input_shape=(122, 1)))
    model.add(MaxPooling1D(5,padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    for layer in model.layers:
        print(layer.output_shape)

    print(model.summary())
    #for train_index, test_index in kfold.split(combined_data_X, y_train):
    if model:
        train_X, test_X, train_y, test_y = train_test_split(combined_data_X, y_train, test_size=0.2, random_state=42)
        
        x_columns_train = new_df_train.columns.drop('class')
        x_train_array = train_X[x_columns_train].values
        x_columns_test = new_df_train.columns.drop('class')
        x_test_array = test_X[x_columns_test].values
        
        Model = SequentialModel()  # Refreshing model for each K-fold
        Model.fit(x_train_array, train_y)  # Train the SVM classifier
        
        pred = Model.predict(x_test_array)  # Perform predictions using the trained classifier
        
        score = metrics.accuracy_score(test_y, pred)  # Evaluate accuracy
        joblib.dump(Model, 'CNNmodel.pkl')

        confussion_matrix=confusion_matrix(test_y, pred, labels=[0,1,2,3,4])    
        
    return {"message":"The Model is trained ","accuracyScore":score}




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
        df = df.drop([each], axis=1)  # Corrected line
    return df
columns = ['protocol_type','service','flag']

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
    plt.show()