from imutils.video import VideoStream
from imutils.video import FPS
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

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# start the FPS counter
fps = FPS().start()

classes = ("Billy", "Maya", "Nara", "Neko")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        ret_val, img = cap.read()
        cv2.imshow("CSI Camera", img)
        input_tensor = transform(Image.fromarray(img))
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        input_batch = input_batch.to(device)
        with torch.no_grad():
            output = model(input_batch)

        print(output)
        softmax = torch.nn.functional.softmax(output[0], dim=0)
        pred = classes[np.argmax(softmax.cpu()).item()]
        print(pred)

        # update the FPS counter
        fps.update()
        
        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

