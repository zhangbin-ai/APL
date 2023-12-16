import torch
import torch.nn.functional as F

def AVContrastive_loss_100(obj_feature:torch.Tensor(), audio_feature:torch.Tensor(), tau=0.4):
    norm_e = 1e-3
    B, T, d = audio_feature.shape

    x =  obj_feature.reshape(B, T, -1, d)
    y = audio_feature.reshape(B, T, 1, d)
    obj_feature = obj_feature.reshape(B*T, -1, d)
    audio_feature = audio_feature.reshape(B*T, 1, d)

    z1 = obj_feature
    z2 = audio_feature
    cos_matrix = F.cosine_similarity(z1, z2, dim=2, eps=1e-6).reshape(B, T, -1)
    cos_matrix_F = F.softmax(cos_matrix, dim=-1)

    cos_matrix_exp = torch.exp(cos_matrix_F / tau)
    
    thread = 0.0110
    zero = torch.zeros_like(cos_matrix)
    positive_matrix = torch.where(cos_matrix_F < thread, zero, cos_matrix_exp)

    second = torch.sum(cos_matrix_exp, dim=-1)
    first = torch.sum(positive_matrix, dim=-1)
    first[first == 0] = norm_e
    
    other_neg_list = []
    for i in range(B):
        obj = torch.cat((x[:i, :, :, :], x[i+1:, :, :, :]), dim=0)
        aud = y[i, :, :, :]
        cos_mat_other = F.cosine_similarity(obj, aud, dim=-1)
        other_neg_list.append(cos_mat_other.unsqueeze(0))

    neg_01 = torch.cat(other_neg_list, dim=0)
    neg_02 = torch.mean(neg_01, dim=1)
    neg_03 = torch.mean(neg_02, dim=-1)

    res = torch.mean((1 / T) *torch.sum(-torch.log(first / (second + neg_03)), dim=-1))
    
    return res if res > norm_e else norm_e