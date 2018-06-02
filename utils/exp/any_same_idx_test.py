import torch

def any_same_idx(idxes):
    idxes_list = list(idxes)
    same_idxes = list()
    unique_idx_list = list()
    same_idxes_idxes = list()
    for i, idx in enumerate(idxes_list):
        if idx not in unique_idx_list:
            unique_idx_list.append(idx)
        elif idx not in same_idxes:
            same_idxes.append(idx)
            same_idxes_idxes.append([idxes_list.index(idx), i,])
        else:
            same_idxes_idxes[same_idxes.index(idx)] += [i,]
    same_idxes_idxes = [torch.LongTensor(o) for o in same_idxes_idxes]
    return same_idxes, same_idxes_idxes


similarity_max = torch.randn(8)
a = torch.LongTensor([1,2,3,4,5,2,4, 2])
same_idx, same_idxes_idxes = any_same_idx(a)

for same_idx_idxes in same_idxes_idxes:
    simi = similarity_max[same_idx_idxes]
    _, max_idx = torch.max(similarity_max[same_idx_idxes], dim=0)
    mask = torch.ByteTensor(similarity_max.size()).fill_(0)
    mask[same_idx_idxes] = 1
    mask[same_idx_idxes[max_idx]] = 0
    similarity_max[mask] = 0.

