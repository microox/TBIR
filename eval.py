import torch

def mapk2(a, b, eps=1e-8, k=10):
    """
    Given embedding spaces F and G for all samples of validation / test set, calculate map@10
    :param a: shared feature space of image
    :param b: shared feature space of caption
    :param k: map@k
    :return: map@k
    """
    # compute cosine similarity matrix of all pairs in a and b
    a_n, b_n = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    cossim = torch.mm(a_norm, b_norm.transpose(-2, -1))

    # rank cosine similarity matrix --> rank images for each caption according to rank of cosine similarity matrix
    values_rank, indices_rank = torch.topk(cossim, k)

    # similarity scores of corresponding pairs
    diag = torch.diag(cossim)

    # mask out similarity scores of corresponding pairs
    mask = torch.diag(torch.ones_like(diag))

    # calculate map@k based on indices_rank and cossim
    map = sort_tensor_by_indices(mask, indices_rank)

    return map


def sort_tensor_by_indices(a, b):
    ap = []
    for i in range(b.shape[0]):
        # change the order of the ith tensor of a with the indexes of b[i]
        c = a[i][b[i]]
        for j in range(c.shape[0]):
            c[j] = c[j] * 1/(j+1)
        ap.append(sum(c).item())
    map = sum(ap) / len(ap)
    return map


def mapk4(b1, b2, eps=1e-8, k=10):
    """
    Given two binary spaces b1 and b2, rank all captions in the binary space
    """ 
    mult = torch.mm(b1, b2.transpose(-2, -1))
    sim = (torch.ones_like(mult) * b1.shape[1] + mult) * 0.5

    # rank cosine similarity matrix --> rank images for each caption according to rank of cosine similarity matrix
    values_rank, indices_rank = torch.topk(sim, k)

    # similarity scores of corresponding pairs
    diag = torch.diag(sim)

    # mask out similarity scores of corresponding pairs
    mask = torch.diag(torch.ones_like(diag))

    # calculate map@k based on indices_rank and cossim
    map = sort_tensor_by_indices(mask, indices_rank)

    return map
