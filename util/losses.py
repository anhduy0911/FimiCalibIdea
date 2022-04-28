import torch
import numpy as np

class ConstrastiveLoss():
    def __init__(self, temperature=1, metrics='cosine'):
        self.temperature = temperature
        self.metrics = metrics
        
    def __call__(self, flt_inps, eps=1e-6):
        if self.metrics == 'cosine':
            B, N, H = flt_inps.shape
            
            # calculate negative similarity
            # shape N, B * H
            flt_negs = flt_inps.permute(1, 0, 2).contiguous()
            flt_negs = flt_negs.view(N, -1).contiguous()
            
            dot_inp = torch.matmul(flt_negs, flt_negs.t())
            norm_inp = torch.norm(flt_negs, dim=1) + eps
            norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
            cosine_inp = dot_inp / norm_mtx_inp
            cosine_inp = 1/2 * (cosine_inp + 1)
            cosine_inp -= torch.eye(N, dtype=torch.float32).to(flt_negs.device)
            similarity_negative = torch.sum(cosine_inp, dim=1) / (N - 1)
            # print(similarity_negative.shape)
            # print('______')

            similarity_positives = []
            for i in range(N):
                # shape B, H
                input = flt_inps[:, i, :]
                dot_inp = torch.matmul(input, input.t())
                norm_inp = torch.norm(input, dim=1) + eps
                norm_mtx_inp = torch.matmul(norm_inp.unsqueeze(1), norm_inp.unsqueeze(0))
                cosine_inp = dot_inp / norm_mtx_inp
                cosine_inp = 1/2 * (cosine_inp + 1)
                cosine_inp -= torch.eye(B, dtype=torch.float32).to(input.device)
                similarity_positive = torch.sum(cosine_inp, dim=1) / (B - 1)
                similarity_positive = torch.mean(similarity_positive)
                similarity_positives.append(similarity_positive)
            
            similarity_positives = torch.stack(similarity_positives, dim=0)
            # print(similarity_positives)
            # print(similarity_positives.shape)
            losses = - torch.log(torch.exp(similarity_positives / self.temperature) / (torch.exp(similarity_negative / self.temperature) + torch.exp(similarity_positives / self.temperature)))
        elif self.metrics == 'dtw':
            B, N, L, H = flt_inps.shape
            negative_scores_mtx = torch.zeros((N, N), dtype=torch.float32).to(flt_inps.device)
            similarity_positives = torch.zeros(N, dtype=torch.float32).to(flt_inps.device)
            for i in range(N):
                # calculate positive similarity
                similarity_positives[i] = dynamic_time_warping(flt_inps[:,i,:,:], flt_inps[:,i,:,:])
                for j in range(i, N):
                    # calculate negative similarity
                    dtw = dynamic_time_warping(flt_inps[:, i, :, :], flt_inps[:, j, :, :])
                    negative_scores_mtx[i,j] = dtw
                    negative_scores_mtx[j,i] = dtw
            
            similarity_negative = torch.sum(negative_scores_mtx, dim=1) / (N - 1)

            losses = - torch.log(torch.exp(similarity_positives / self.temperature) / (torch.exp(similarity_negative / self.temperature) + torch.exp(similarity_positives / self.temperature)))
        return torch.mean(losses)

def dynamic_time_warping(s, t):
    s = s.flatten().contiguous()
    t = t.flatten().contiguous()
    
    n, m = len(s), len(t)
    dtw_matrix = torch.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = torch.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]

if __name__ == '__main__':
    import time
    import torch.nn as nn

    loss = ConstrastiveLoss(metrics='dtw')
    loss2 = nn.MSELoss()
    inp = torch.randn(128, 5, 7, 64)
    inp2 = torch.randn(128, 5, 7, 64)

    start = time.time()
    print
    print(loss(inp))
    print(time.time() - start)
    # start = time.time()
    # print(loss2(inp, inp2))
    # print(time.time() - start)
    
    for i in range(100):
        inp = torch.randn(128, 5, 64)
        assert(loss(inp) > 0)
