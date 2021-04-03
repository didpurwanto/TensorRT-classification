
import torch as t 
import cv2
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import  ToTensor
from albumentations.augmentations.transforms import Normalize

from torchvision import models
model = models.resnet50(pretrained=True)

def preprocess_image(img_path):
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ])
    
    input_img = cv2.imread(img_path)

    input_data = transforms(image=input_img)["image"]
    batch_data = t.unsqueeze(input_data, 0)
    return batch_data

def calculate_score(output_data):
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    
    confidences = t.nn.functional.softmax(output_data, dim=1)[0] * 100
    _, indices = t.sort(output_data, descending=True)
    
    # print('indices ', indices)
    # print('confidences: ', confidences.shape)
    # print('classes: ', classes)

    for i in  range(1000):
        score = confidences[i].cpu().detach().numpy()
        if score > 0.5:
            print(score, classes[i])



input = preprocess_image("/media/didpurwanto/DiskL/disertation_ex/turkish_coffee.jpg").cuda()    

model.eval()
model.cuda()
output = model(input)

calculate_score(output)

