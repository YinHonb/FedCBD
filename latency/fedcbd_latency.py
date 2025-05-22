import torch
import torchvision.models as models
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 1000
# alexnet vgg16 resnet50 mobilenet_v2
torch.cuda.synchronize()
start_time = time.time()
models_list = [models.mobilenet_v2(pretrained=False).to(device) for _ in range(n)]
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
S = torch.stack(S)
I = torch.eye(n, device=device)
A = torch.ones((n, n), device=device) - I
Sig = torch.ones((n, n), device=device)
D = torch.mul(torch.matmul(A, Sig.T), I)
Q = torch.inverse(torch.mul(I, torch.matmul((I + A), Sig.T)))
H = torch.matmul(Q, (torch.mul(Sig, A) - D)) + I
H = H.to(device)
print(H)
torch.cuda.synchronize()
start_time = time.time()
S = torch.torch.matmul(H, S)
torch.cuda.synchronize()
end_time = time.time()
print('fedcbd:', end_time - start_time)

