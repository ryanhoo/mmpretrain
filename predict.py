import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from torchvision import models    

checkpoint_file='work_dirs/efficientnetv2_s/epoch_100.pth'
model = torch.load(checkpoint_file)
model.eval()
print('Finished loading model!')
print(model)
device = torch.device("cpu" if args.cpu else "cuda")
model = model.to(device)

# ------------------------ export -----------------------------
output_onnx = 'super_resolution.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input0"]
output_names = ["output0","output1"]
inputs = torch.randn(1, 3, 1080, 1920).to(device)

torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)