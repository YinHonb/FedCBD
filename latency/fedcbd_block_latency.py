import torch
import torchvision.models as models
import time
import multiprocessing as mp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 2000
# alexnet vgg16 resnet50 mobilenet_v2
start_time = time.time()
models_list = [models.mobilenet_v2(pretrained=False).to(device)]*n
end_time = time.time()
print('add:', end_time - start_time)
start_time = time.time()
S = []
for model in models_list:
    flattened_params = []
    for param in model.parameters():
        flattened_params.append(param.flatten())
    S.append(torch.cat(flattened_params))
end_time = time.time()
print('flatten:', end_time - start_time)
del models_list
S = torch.stack(S)
I = torch.eye(n, device=device)
A = torch.ones((n, n), device=device) - I
Sig = torch.rand((n, n), device=device)
D = torch.mul(torch.matmul(A, Sig.T), I)
Q = torch.inverse(torch.mul(I, torch.matmul((I + A), Sig.T)))
H = torch.matmul(Q, (torch.mul(Sig, A) - D)) + I
H = H.to(device)
print(H)
torch.cuda.synchronize()
start_time = time.time()
for i in range(20):
    S_temp = torch.matmul(H[i*100:(i+1)*100][:], S)
torch.cuda.synchronize()
end_time = time.time()
print('fedcbd:', end_time - start_time)

