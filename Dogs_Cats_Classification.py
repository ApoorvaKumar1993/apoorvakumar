# Dogs vs Cats classification using Convolutional Neural Network


# Part 1 - Building the CNN

    # your code here
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


final_test =  test_datagen.flow_from_directory('dataset/kaggle_test',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'binary', shuffle=False)

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)



temp_test = test_datagen.flow_from_directory('dataset/temp_test',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary', shuffle = False)

predict = classifier.predict_generator(temp_test)
#predict = classifier.predict_generator(final_test)
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Accuracy = classifier.evaluate_generator(final_test)
#print('Accuracy',Accuracy)


save_model = 0

if save_model == 1:

    model_json = classifier.to_json()
    with open("DogCatModel.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("DogCatModel.json.h5")
    print("Saved model to disk")

load_model = 0


if load_model == 1:


    json_file = open('DogCatModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("DogCatModel.json.h5")
    print("Loaded model from disk")
    temp_test = test_datagen.flow_from_directory('dataset/temp_test',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary',shuffle=False)
    
    predict = loaded_model.predict_generator(temp_test)
    predict_kaggle = loaded_model.predict_generator(final_test)
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    Accuracy = loaded_model.evaluate_generator(final_test)
    print('Accuracy',Accuracy)
    
    
    for i in range(len(predict)):
        if predict[i]<0.5:
            predict[i] = 0
        else:
            predict[i] = 1
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
        
    
    y_true = temp_test.classes
    y_pred = predict
    
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    print(cm)
    print('F1_Score',f1) 
    
import pandas as pd
import numpy as np
predict_kaggle = np.reshape(predict_kaggle,(-1,))

ids = np.arange(start = 1, stop =12501, step =1)
submissionfile = pd.DataFrame({'id':ids, 'label':predict_kaggle})

submissionfile.to_csv('Final_Prediction.csv', index=None)