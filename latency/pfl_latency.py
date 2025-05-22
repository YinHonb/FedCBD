import torch
import torchvision.models as models
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 2000
# alexnet vgg16 resnet50 mobilenet_v2
torch.cuda.synchronize()
start_time = time.time()
models_list = [models.mobilenet_v2(pretrained=False).to(device)]*n
torch.cuda.synchronize()
end_time = time.time()
print('add:', end_time - start_time)
torch.cuda.synchronize()
start_time = time.time()
S = []
for model in models_list:
    flattened_params = []
    for param in model.parameters():
        flattened_params.append(param.flatten())
    S.append(torch.cat(flattened_params))
torch.cuda.synchronize()
end_time = time.time()
print('flatten:', end_time - start_time)
del models_list

S_pp = []
torch.cuda.synchronize()
start_time = time.time()
for i in range(n):
    temp = torch.zeros_like(S[i], device=device)
    a = 0
    for j in random.sample(range(n), 1000):
        temp += S[j] * 1.0
        a += 1.0
    temp /= a
torch.cuda.synchronize()
end_time = time.time()
print('pfl:', end_time - start_time)
