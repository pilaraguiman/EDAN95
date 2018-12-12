import numpy as np
import os
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import models
from keras.models import load_model
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
import pickle
from sklearn.metrics import confusion_matrix
#Convbase maybe use the other file

train_dir = "flowers_split/test"
validation_dir = "flowers_split/validation"
test_dir = "flowers_split/test"

batch_size=20
test_datagen = ImageDataGenerator(rescale=1./255)
conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape = (150,150,3))
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = batch_size,
    class_mode = 'categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = batch_size,
    class_mode = 'categorical'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = batch_size,
    class_mode = 'categorical')

#Extract features method
def extract_features(sample_count, generator):
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count,5))
    i = 0
    k = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)

        features[k:k+inputs_batch.shape[0]] = features_batch
        labels[k:k+inputs_batch.shape[0]] = labels_batch
        i += 1
        k += inputs_batch.shape[0]
        if (i+1) * batch_size >= sample_count:
            break
    return features, labels





#Extract features
print("extracting")
#train_features, train_labels = extract_features(2593, train_generator)
print("1")
#validation_features, validation_labels = extract_features(865, validation_generator)
print("2")
#test_features, test_labels = extract_features(865, test_generator)
print("3")

#Model the thingy
print("modeling")
#model = models.Sequential()
#model.add(layers.Flatten(input_shape=(3, 3, 2048)))
#model.add(layers.Dense(256, activation='relu',))
#model.add(layers.Dropout(0.4))
#model.add(layers.Dense(5, activation='softmax'))
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

#flatten the features
print("reshaping")
#Save features
#pickle.dump(train_features, open("train_features.sav", 'wb'))
#pickle.dump(validation_features,  open("validation_features.sav", 'wb'))
#pickle.dump(test_features, open("test_features.sav", 'wb'))

#pickle.dump(train_labels, open("train_labels.sav", 'wb'))
#pickle.dump(validation_labels,  open("validation_labels.sav", 'wb'))
#pickle.dump(test_labels, open("test_labels.sav", 'wb'))



#Load features
print("loading")

train_features = pickle.load(open("train_features.sav", 'rb'))
#validation_features = pickle.load(open("validation_features.sav", 'rb'))
test_features = pickle.load(open("test_features.sav", 'rb'))

train_labels = pickle.load(open("train_labels.sav", 'rb'))
test_labels = pickle.load(open("test_labels.sav", 'rb'))
#validation_labels = pickle.load(open("validation_labels.sav", 'rb'))
# Fitting the model

#print(train_features.shape)
print("fitting")
model = load_model("model.h5")
results = model.evaluate(test_features, test_labels)


y_predicted = model.predict(test_features)

y_test_non_category = [ np.argmax(t) for t in test_labels]
y_predict_non_category = [ np.argmax(t) for t in y_predicted]

cfm = confusion_matrix(y_test_non_category, y_predict_non_category)
print(cfm)


#print(results)
#history = model.fit(
    #  train_features,
    #train_labels,
    #epochs=20,
    #batch_size=20,
    #validation_data = (test_features,test_labels))

print("saving")

#model.save("model.h5")







