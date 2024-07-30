import os
import numpy as np
import random
import torch
import argparse
from tqdm import tqdm
from torchvision.transforms import *

from model.deepten import DeepTen
from metric import AverageMeter, accuracy

from PIL import Image

    
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_to_idx = {'brick': 0, 'carpet': 1, 'ceramic': 2, 'fabric': 3, 'foliage': 4, 'food': 5, 'glass': 6, 'hair': 7, 'leather': 8, 'metal': 9, 'mirror': 10, 'other': 11, 'painted': 12, 'paper': 13, 'plastic': 14, 'polishedstone': 15, 'skin': 16, 'sky': 17, 'stone': 18, 'tile': 19, 'wallpaper': 20, 'water': 21, 'wood': 22}
idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))

def main(args):
    # Load Model
    n_class = 23
    model = DeepTen(n_class, pretrained=False).cuda()
    model.eval()

    # Load Checkpoint
    # checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/model/deepten_resnet50_minc-1225f149.pth")
    checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/checkpoints/2/24.tar")['model_state_dict']
    model.load_state_dict(checkpoint, strict=False)

    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([# Resize(base_size),
                                         CenterCrop(352),
                                         ToTensor(),
                                         normalize])
    
    img = Image.open(args.img_path).convert('RGB')
    img = transform_test(img)
    _dirname = os.path.split(os.path.dirname(args.img_path))[1]
    label = class_to_idx[_dirname]

    data = img.cuda().unsqueeze(0)
    output = model(data.float())

    _, pred = output.topk(5, 1, True, True)
    pred_list = pred[0].cpu().numpy().tolist()
    p = torch.softmax(output, dim=1)[0].detach().cpu().numpy().tolist()
    for i in range(5):
        no = int(pred_list[i])
        print(idx_to_class[no], p[no])


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # data settings
    parser = argparse.ArgumentParser(description='Deep Encoding')
    parser.add_argument('--img_path', type=str, 
                        default='/data/zhangpengjie/zhangpengjie/Workspace/Datasets/minc-2500/images/stone/stone_000015.jpg')
    # cuda, seed and logging
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
