import torch
from PIL import Image
import os
from collections import OrderedDict
import torchvision.transforms as transforms
from skimage import img_as_ubyte
from model.arch.DSFNet import DSFNet

import cv2
import torch.nn.functional as F

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["model"])
    except:
        state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

model = DSFNet()
model.cuda().eval()
print('restoring images......')

def pad_to_multiple(x, multiple=16):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, h, w

def predict(prd_root: str, out_dir: str):
    load_checkpoint(model, weights_dir)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    for filename in os.listdir(prd_root):
        img = Image.open(os.path.join(prd_root,filename)).convert('RGB')
        input_ = train_transform(img)
        input = input_.unsqueeze(0)
        input = input.cuda()
        input, orig_h, orig_w = pad_to_multiple(input, multiple=16)
        with torch.no_grad():
            restored = model(input)
            restored = restored[:, :, :orig_h, :orig_w]
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        os.makedirs(os.path.join(out_dir,os.path.basename(prd_root)),exist_ok=True)
        restored = img_as_ubyte(restored[0])
        save_img((os.path.join(out_dir,os.path.basename(prd_root), filename)),restored)
    print('finished')

def predict256(prd_root: str, out_dir: str):
    load_checkpoint(model, weights_dir)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    for filename in os.listdir(prd_root):
        img = Image.open(os.path.join(prd_root,filename)).convert('RGB')
        input_ = train_transform(img)
        input = input_.unsqueeze(0)
        input = input.cuda()
        input, orig_h, orig_w = pad_to_multiple(input, multiple=16)
        with torch.no_grad():
            restored = model(input)
            restored = restored[:, :, :orig_h, :orig_w]
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        os.makedirs(os.path.join(out_dir,os.path.basename(prd_root) + '256'),exist_ok=True)
        restored = img_as_ubyte(restored[0])
        save_img((os.path.join(out_dir,os.path.basename(prd_root) + '256', filename)),restored)
    print('256finished')

if __name__ == '__main__':
    import sys
    prd_root = sys.argv[2] if len(sys.argv)>2 else '/root/autodl-tmp/UIEB/test'
    out_dir = sys.argv[3] if len(sys.argv)>3 else './result'
    weights_dir = sys.argv[1] if len(sys.argv)>1 else './checkpoint/DSFNet/DSFNet/ckpt/best_psnr.pth'

    predict(prd_root,out_dir)
