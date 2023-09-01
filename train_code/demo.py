import torch
import argparse
import cv2
import numpy as np
import os
from torchvision import models

weight_path = "../train_logs/last.pth"
input_width = 384
input_height = 384
input_channels = 1

with torch.no_grad():
    model = torch.load(weight_path)
    model.eval()
    model.cpu()
    img = cv2.imdecode(np.fromfile("../demo_images/1.jpg",np.uint8),0)
    img = img[0:400,0:400]
    img = cv2.resize(img,(input_width,input_height))
    img = np.float32(img/255)
    tens = np.expand_dims(img, 2)
    tens = np.transpose(tens,(2,1,0))
    tens = np.expand_dims(tens,0)
    tens = torch.Tensor(tens)
    out = model(tens)
    id = np.argmax(out,1)

    print(id)