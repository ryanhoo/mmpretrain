import torch

# model_path = '/data1/docker-lab/TF/mj/mix_cls_471700.pth'
model_path = 'work_dirs/efficientnetv2_s_2nd/best.pth'
# 加载权重文件
checkpoint = torch.load(model_path)

print(checkpoint['meta'])