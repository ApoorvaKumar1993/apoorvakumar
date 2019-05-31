# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:22:47 2019

@author: tudurdod
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:05:13 2019

@author: tudurdod
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
import matplotlib
import scipy.signal as sp

#Importing datasets


dataset_1 = pd.read_csv('Frequency_Domain_Analysis\Differential_Mode\Training_Datas\Diffvia_poor_z_Mode_Conversion.csv')
dataset_1 = dataset_1[['Frequency','SDD21','Sqaure_feature','Rollingstd','Insertion_dip']]
                                     
dataset_1 = dataset_1.dropna()

dataset_5 = pd.read_csv('Frequency_Domain_Analysis\Differential_Mode\Training_Datas\MS_return_plane_removed.csv')
dataset_5 = dataset_5[['Frequency','SDD21','Sqaure_feature','Rollingstd','Insertion_dip']]
dataset_5 = dataset_5.dropna()


dataset_4 = pd.read_csv('Frequency_Domain_Analysis\Differential_Mode\Training_Datas\J37J38J39J40J41J42J43J44J45J46J47J48_MS3Pair_4W_12port_deembed.csv')
dataset_4 = dataset_4[['Frequency','SDD21','Sqaure_feature','Rollingstd','Insertion_dip']]
dataset_4 = dataset_4.dropna()

dataset_3 = pd.read_csv('Frequency_Domain_Analysis\Differential_Mode\Training_Datas\J1J2J3J4J5J6J7J8J9J10J11J12_MS3Pair_1W_12port_deembed.csv')
dataset_3 = dataset_3[['Frequency','SDD21','Sqaure_feature','Rollingstd','Insertion_dip']]
dataset_3 = dataset_3.dropna()


dataset_2= pd.read_csv('Frequency_Domain_Analysis\Differential_Mode\Training_Datas\J_m7_20inBkpl_train.csv')
dataset_2 = dataset_2[['Frequency','SDD21','Sqaure_feature','Rollingstd','Insertion_dip']]
dataset_2 = dataset_2.dropna()



dataset_6 = pd.read_csv('Frequency_Domain_Analysis\Differential_Mode\Training_Datas\Patho_80_120_machine_learning5.csv')
dataset_6 = dataset_6[['Frequency','SDD21','Sqaure_feature','Rollingstd','Insertion_dip']]
dataset_6 = dataset_6.dropna()




dataframes = [dataset_1,dataset_2,dataset_3,dataset_4,dataset_5,dataset_6]
Training_dataset = pd.concat(dataframes)
#Training_dataset = Training_dataset[['Frequency','SDD11','SDD21','SCD21','Sqaure_feature','Rolling_Std',
#                                     'Impedance_Threshold','ModeConversion_Threshold',
#                                     'Insertion_loss_dip_from_std']]

Correlation = Training_dataset.corr()

X = Training_dataset.iloc[:,0:4].values
y = Training_dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.20, shuffle =False, stratify =None)


'Standardization'
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'Saving Standardizer'


if 0:
    from joblib import dump, load
    dump(sc, 'Standardizer.joblib') 

'Modeling'



from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
X_train = np.reshape(X_train,[-1, 1, 4])
X_test = np.reshape(X_test,[-1, 1, 4])


y_train = np.reshape(y_train,[-1,1])
y_test = np.reshape(y_test,[-1,1])

epochs=10
batch_size=64
model = Sequential()
model.add(LSTM(64, input_shape=(1,4),return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.array(X_train), np.array(y_train), epochs= epochs, batch_size= batch_size, 
          verbose=2, validation_split=0.2)  
#print(model.summary())

Test_Loss = model.evaluate(np.array(X_test), np.array(y_test))
print("Test Loss", Test_Loss)
predict_deep = model.predict(np.array(X_test))  
predict_deep = np.where(predict_deep<0.5,0,1)



'Save Models'

from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




#def testing(two,More_than_two,test_link):
#
#    'Test data prediction'
#
#    testdataset = pd.read_csv(test_link, low_memory =False)
#    testdataset = testdataset.dropna()
#    
#    cols = testdataset.columns
#    cols = cols.map(lambda x: x.replace(' ', ''))
#    cols = cols.map(lambda x: x.replace('_', ''))
#    testdataset.columns = cols
#    testdataset.rename(columns={'!DATAFreq':'Frequency'},inplace=True)
#    #testdataset.convert_objects(convert_numeric=True).dtypes
#    
#    
#    'NUmber of Diff port'
#    
#   
#    
#    if two:
#        'Threshold value'
#        Returnloss_threshold = -13
#        Mode_Conversion_Threshold = -18
#        
#        inertion_loss = testdataset[['SDD21']]
#        Sqaure_Feature = inertion_loss * inertion_loss
#        testdataset['Sqaure_feature'] = Sqaure_Feature
#        Impedance_Threshold = np.where(testdataset['SDD11']>Returnloss_threshold, 1, 0)
#        Mode_Conversion_Threshold = np.where(testdataset['SCD11']>Mode_Conversion_Threshold, 1, 0)
#        Rollingstd= Sqaure_Feature.rolling(10).std()
#        Rollingstd = Rollingstd.fillna(0.00)
#        testdataset['Rollingstd'] = Rollingstd
#        testdataset = testdataset[['Frequency','SDD21','Sqaure_feature','Rollingstd']]
#        
#        
#        test_x = testdataset.iloc[:,0:4].values
#        test_x = sc.transform(test_x)
#        test_x = np.reshape(test_x,[-1, 1, 4])
#        Test_Prediction = model.predict(np.array(test_x))  
#        Test_Prediction = np.where(Test_Prediction<0.5,0,1)
#        
#        Impedance_issue = []
#        Mode_conversion_issue = []
#        
#        'Impedance issue check'
#        
#        for i, j in zip(Impedance_Threshold, Test_Prediction):
#            
#            if (i and j) == 1:
#                Impedance_issue.append(1)
#                
#        if 1 in Impedance_issue:
#            print('Impedance discontinuity issue')
#            
#        'Mode conversion check'
#        
#        for i, j in zip(Mode_Conversion_Threshold, Test_Prediction):
#            
#            if (i and j) == 1:
#                Mode_conversion_issue.append(1)
#                
#        if 1 in Mode_conversion_issue:
#            print('Mode conversion issue')
#            
#                
#                
#            
#            
#    if More_than_two:
#        
#        Returnloss_threshold = -13
#        Mode_Conversion_Threshold = -16
#        NEXT_Threshold = -13
#        FEXT_Threshold = -13
#        
#        inertion_loss = testdataset[['SDD34']]
#        Sqaure_Feature = inertion_loss * inertion_loss
#        testdataset['Sqaure_feature']= Sqaure_Feature
#        Impedance_Threshold = np.where(testdataset['SDD11']>Returnloss_threshold, 1, 0)
#        Mode_Conversion_Threshold = np.where(testdataset['SCD33']>Mode_Conversion_Threshold, 1, 0)
#        Rollingstd= Sqaure_Feature.rolling(10).std()
#        Rollingstd = Rollingstd.fillna(0.00)
#        NEXT_Threshold = np.where(testdataset['SDD31']>NEXT_Threshold , 1, 0)
#        FEXT_Threshold = np.where(testdataset['SDD41']>FEXT_Threshold, 1, 0)
#        #Rollingstd=(Rollingstd-Rollingstd.mean())/Rollingstd.std()
#        #Rollingmean= Sqaure_Feature.rolling(10).mean()
#        testdataset['Rollingstd'] = Rollingstd
#        testdataset = testdataset[['Frequency','SDD21','Sqaure_feature','Rollingstd']]
#        
#        
#        test_x = testdataset.iloc[:,0:4].values
#        test_x = sc.transform(test_x)
#        test_x = np.reshape(test_x,[-1, 1, 4])
#        Test_Prediction = model.predict(np.array(test_x))  
#        Test_Prediction = np.where(Test_Prediction<0.5,0,1)
#        
#        Impedance_issue = []
#        Mode_conversion_issue = []
#        NEXT_issue = []
#        FEXT_issue = []
#        
#        
#        'Impedance issue check'
#        
#        for i, j in zip(Impedance_Threshold, Test_Prediction):
#            
#            if (i and j) == 1:
#                Impedance_issue.append(1)
#                
#        if 1 in Impedance_issue:
#            print('Impedance discontinuity issue')
#            
#        'Mode conversion check'
#        
#        for i, j in zip(Mode_Conversion_Threshold, Test_Prediction):
#            
#            if (i and j) == 1:
#                Mode_conversion_issue.append(1)
#                
#        if 1 in Mode_conversion_issue:
#            print('Mode_conversion_issue')
#            
#        'Impedance issue check'
#        
#        for i, j in zip(NEXT_Threshold, Test_Prediction):
#            
#            if (i and j) == 1:
#                NEXT_issue.append(1)
#                
#        if 1 in NEXT_issue:
#            print('NEXT issue')
#            
#        'Mode conversion check'
#        
#        for i, j in zip(FEXT_Threshold, Test_Prediction):
#            
#            if (i and j) == 1:
#                FEXT_issue.append(1)
#                
#        if 1 in FEXT_issue:
#            print('FEXT issue')
#
#
#
#
#link = 'Frequency_Domain_Analysis\Differential_Mode\Test_Datas/X_Talk_Patho.csv'
#two = 0
#More_than_two = 1
#testing(two,More_than_two,link)
