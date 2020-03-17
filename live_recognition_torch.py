# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
#from keras.applications import imagenet_utils

#import face_recognition
import imutils
import time
import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

with open('torch_cat.model', 'rb') as f:
    model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("retained device is ", device)
model.eval()
model.to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

classes = ("Billy", "Maya", "Nara", "Neko")

# loop over frames from the video file stream

current_cat = None
while True:
	frame = vs.read()
	img = Image.fromarray(frame)
	input_tensor = transform(img)
	input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
	input_batch = input_batch.to(device)
	with torch.no_grad():
		output = model(input_batch)

	softmax = torch.nn.functional.softmax(output[0], dim=0)
	pred = classes[np.argmax(softmax.cpu()).item()]

	print(pred)

	# display the image to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()