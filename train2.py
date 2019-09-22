
import matplotlib

matplotlib.use("Agg")

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
import numpy as np
from imutils import paths
import cv2


#tf.config.optimizer_set_jit(True)

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-1
BS = 20 #8 initially
EPOCHS = 50

number_categories = 4
im_width=64
im_height=64

imagePaths = sorted(list(paths.list_images('./dataset')))
images = []
labels = []

def load_images():
    # dummy function, implement this
    # Return a numpy array of shape (N, height, width, #channels)
    # or a list of (height, width, #channels) arrays (may have different image
    # sizes).
    # Images should be in RGB for colorspace augmentations.
    # (cv2.imread() returns BGR!)
    # Images should usually be in uint8 with values from 0-255.

    print("[INFO] Loading images in memory...")
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (im_width, im_height))
        images.append(image)#[:,:,::-1])#---> RGB to BGR
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    return images


images = load_images()
#images_aug = seq.augment_images(images)  # done by the library

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
images = np.array(images, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
print(labels)
labels = le.fit_transform(labels)

print("labels : ", labels)

labels = np_utils.to_categorical(labels, number_categories)
print("categories : ", labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.25, random_state=42)

# initialize an our data augmenter as an "empty" image data generator
aug = ImageDataGenerator()

# check to see if we are applying "on the fly" data augmentation, and
# if so, re-instantiate the object
print("[INFO] performing 'on the fly' data augmentation")
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

print("[INFO] creating model from VGG16")

prior = keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(im_width, im_height, 3)
)

model = Sequential()
model.add(prior)
model.add(Flatten())
model.add(Dense(256, activation='relu', name='Dense_Intermediate'))
model.add(Dropout(0.1, name='Dropout_Regularization'))
model.add(Dense(4, activation='sigmoid', name='Output'))

print("[INFO] Preventing weight modification from any initial layers of VGG16")

# Freeze the VGG16 model, e.g. do not train any of its weights.
# We will just use it as-is.
for cnn_block_layer in model.layers[0].layers:
    cnn_block_layer.trainable = False
model.layers[0].trainable = False


print("[INFO] compile the model")

# Compile the model. I found that RMSprop with the default learning
# weight worked fine.
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[INFO] Fitting")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format('cats.model'))
model.save('cats.model')

print("[INFO] Generating info graph")
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure(figsize=(15,15))
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('cat_model_perf_' + str(im_width) +'.jpg')