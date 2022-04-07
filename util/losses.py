import torch

class ConstrastiveLoss():
    
    def __call__(self, flt_inps, eps=1e-6):
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
        
        # print(similarity_negative)
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

        losses = - torch.log(torch.exp(similarity_negative) / (torch.exp(similarity_negative) + torch.exp(similarity_positives)))
        # print(losses)

        return torch.mean(losses)

if __name__ == '__main__':
    import time
    import torch.nn as nn

    loss = ConstrastiveLoss()
    loss2 = nn.MSELoss()
    inp = torch.randn(128, 5, 64)
    inp2 = torch.randn(128, 5, 64)

    start = time.time()
    print(loss(inp))
    print(time.time() - start)
    start = time.time()
    print(loss2(inp, inp2))
    print(time.time() - start)
    