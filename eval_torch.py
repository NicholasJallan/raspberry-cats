import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix

with open('torch_cat.model', 'rb') as f:
    model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)

model.eval()
if torch.cuda.is_available():
    model.to('cuda')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])


test_path = './dataset/test/'
test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]

classes = ("Billy", "Maya", "Nara", "Neko")

y_true = []
y_pred = []
for test_file in test_files:
    input_image = Image.open(test_path + test_file)
    y_true.append(test_file[:-5].lower())

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    input_batch = input_batch.to('cuda')

    # move the input and model to GPU for speed if available
    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    softmax = torch.nn.functional.softmax(output[0], dim=0)
    pred = classes[np.argmax(softmax.cpu()).item()]
    y_pred.append(pred.lower())
    if pred.lower() != test_file[:-5].lower():
        print('missclassified : ', test_file)

labels = ['billy', 'maya', 'nara', 'neko']
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(include_values=True)
plt.show()