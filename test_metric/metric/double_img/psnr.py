import torch
import torchvision.transforms.functional as TF


def calc_psnr(tar_img, prd_img):
    tar_img = TF.to_tensor(tar_img)
    prd_img = TF.to_tensor(prd_img)
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps.item()
