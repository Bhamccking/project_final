import torch
import argparse
import cv2
import numpy as np
import os
from train import buildNet

weight_path = "../train_logs/best.pth"
save_path = "../weights/model.onnx"

os.makedirs(os.path.dirname(save_path),exist_ok=True)

input_width = 224
input_height = 224
input_channels = 1

with torch.no_grad():
    model = torch.load(weight_path)
    # model = buildNet(2,"",input_channels)
    # model.load_state_dict(checkpoint)
    model.eval()
    img = torch.zeros((1, input_channels,input_width,input_height))
    model.cpu()
    img.cpu()

    torch_out = torch.onnx._export(model, img,save_path,
                                   opset_version=12,
                                   input_names=['input'],
                                   output_names=['output'],
                                   verbose=False,
                                   export_params=True
                                   )
    try:
        net = cv2.dnn.readNet(save_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        cv_mat = np.zeros((input_width, input_height,input_channels),np.uint8)
        blob = cv2.dnn.blobFromImage(cv_mat, 1 / 255.0, (input_width, input_height), crop=False)
        net.setInput(blob)
        out = net.forward()[0]
        print(out)
        print("onnx succeed")
    except:
        print("onnx fail")

print("save model", os.path.abspath(save_path))

