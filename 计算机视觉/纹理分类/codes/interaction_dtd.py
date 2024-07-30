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

class_to_idx ={'banded': 0, 'blotchy': 1, 'braided': 2, 'bubbly': 3, 'bumpy': 4, 'chequered': 5, 'cobwebbed': 6, 'cracked': 7, 'crosshatched': 8, 'crystalline': 9, 'dotted': 10, 'fibrous': 11, 'flecked': 12, 'freckled': 13, 'frilly': 14, 'gauzy': 15, 'grid': 16, 'grooved': 17, 'honeycombed': 18, 'interlaced': 19, 'knitted': 20, 'lacelike': 21, 'lined': 22, 'marbled': 23, 'matted': 24, 'meshed': 25, 'paisley': 26, 'perforated': 27, 'pitted': 28, 'pleated': 29, 'polka-dotted': 30, 'porous': 31, 'potholed': 32, 'scaly': 33, 'smeared': 34, 'spiralled': 35, 'sprinkled': 36, 'stained': 37, 'stratified': 38, 'striped': 39, 'studded': 40, 'swirly': 41, 'veined': 42, 'waffled': 43, 'woven': 44, 'wrinkled': 45, 'zigzagged': 46}
idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))

def main(args):
    # Load Model
    n_class = 47
    model = DeepTen(n_class, pretrained=False).cuda()
    model.eval()

    # Load Checkpoint
    # checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/model/deepten_resnet50_minc-1225f149.pth")
    checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/checkpoints/3_dtd/119.tar")['model_state_dict']
    model.load_state_dict(checkpoint, strict=False)

    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([# Resize(base_size),
                                         CenterCrop(320),
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
                        default="/data/zhangpengjie/zhangpengjie/Workspace/Datasets/dtd/images/waffled/waffled_0108.jpg")
    # cuda, seed and logging
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
