import random
import numpy as np
import torch
from FedCBD import FedCBD
from config import get_args


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    args, cfg = get_args()
    fedcbd = FedCBD(args, cfg)

    for now_time in range(args.comm_round):
        fedcbd.step(now_time)
        if now_time == 0:
            print('Weight matrix: ')
            for i in range(args.n_parties):
                print(fedcbd.W[i])
            print('Similarity matrix: ')
            for i in range(args.n_parties):
                print(fedcbd.cos_matrix[i])





