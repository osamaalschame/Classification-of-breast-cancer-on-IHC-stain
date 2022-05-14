
import cv2
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model

# lists to store data
data = []
label = []

# folder where data is placed
BASE_FOLDER = '/Users/osama-mac/Desktop/master/sim/cells/'

folders = [i for i in os.listdir(BASE_FOLDER) if not i.startswith('.')]

# loading data to lists
i=0
for folder in folders:
    for file in (os.listdir(BASE_FOLDER + folder)):
        if (file.split('.')[1] == 'png'):
            img = cv2.imread(BASE_FOLDER + folder + '/' + file)
            img = Image.fromarray(img)
            img = img.resize((224, 224))
            data.append(np.array(img))
            label.append(np.array(i))
    
    i=i+1

data=np.array(data)
label=np.array(label)

# now split the data in to train and test with the help of train_test_split
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)

# convert train label to categorical 
train_label = to_categorical(train_label, 4)
test_label = to_categorical(test_label, 4)
#normalize 

train_data = train_data/255
test_data = test_data/255
num_classes = 4
# define the larger model
def leNet_model():
  # create model
  model = Sequential()
  
  model.add(Conv2D(30, (5, 5), input_shape=(224, 224,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(15, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  # Compile model
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = leNet_model()
print(model.summary())
# save the best
# checkpoint_filepath = 'weights/classification.h5'
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)

# load pre-trained weights
# model.load_weights('classification.h5')
# # run the training 
# history=model.fit(train_data, train_label, 
#                   epochs=50,  
#                   callbacks=[model_checkpoint_callback],
#                   validation_data = (test_data,test_label), 
#                   batch_size = 400,
#                   shuffle=False, 
#                   verbose = 1)
# save the model
# model.save('classification_100_epochs.h5')
# # plot the model
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training', 'validation'])
# plt.title('Loss')
# plt.xlabel('epoch')
# plt.show() 
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training','validation'])
# plt.title('Accuracy')
# plt.xlabel('epoch')
# plt.show()
# score = model.evaluate(test_data, test_label, verbose=0)
 


# # IOU
# y_pred=model.predict(test_data)
# y_pred_thresholded = y_pred > 0.5

# intersection = np.logical_and(test_label, y_pred_thresholded)
# union = np.logical_or(test_label, y_pred_thresholded)
# iou_score = np.sum(intersection) / np.sum(union)
# print("IoU socre is: ", iou_score)



#predict the image
model.load_weights('weights/classification.h5')

im=test_data[87]
im=np.expand_dims(im,axis=0)
print(im.shape)
prediction=model.predict(im)
prediction=prediction.argmax(axis=-1)
print("predicted digit: "+str(prediction))


#confusion matrix 
# predictions = model.predict(test_data)
# y_pred = (predictions > 0.5)
# matrix = confusion_matrix(test_label.argmax(axis=1), y_pred.argmax(axis=1))
# ax = sns.heatmap(matrix, annot=True, fmt='g')

# ax.set_title('Confusion Matrix ')
# ax.set_xlabel('Predicted images')
# ax.set_ylabel('Actual images')

# ax.xaxis.set_ticklabels(['Moderate', 'Negative', 'Strong','Weak'])
# ax.yaxis.set_ticklabels(['Moderate', 'Negative', 'Strong','Weak'])

# plt.show()