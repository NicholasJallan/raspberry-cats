import matplotlib

matplotlib.use("Agg")

from resnet import ResNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
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
labels = le.fit_transform(labels)

labels = np_utils.to_categorical(labels, number_categories)

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

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)
model = ResNet.build(im_width, im_height, 3, number_categories, (2, 3, 4), (32, 64, 128, 256), reg=0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
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