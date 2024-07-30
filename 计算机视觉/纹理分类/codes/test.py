import os
import numpy as np
import random
import torch
import argparse
from tqdm import tqdm
from torchvision.transforms import *

from model.deepten import DeepTen
from datasets.minc import MINCDataset
from metric import AverageMeter, accuracy
    
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Load Model
    n_class = 23
    model = DeepTen(n_class, pretrained=False).cuda()
    model.eval()
    # for name,params in model.named_parameters():
    #     print(name)

    # Load Checkpoint
    # checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/model/deepten_resnet50_minc-1225f149.pth")
    checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/checkpoints/2/24.tar")['model_state_dict']
    model.load_state_dict(checkpoint, strict=False)

    # Load Data
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([# Resize(base_size),
                                         CenterCrop(352),
                                         ToTensor(),
                                         normalize])
    dataset = MINCDataset(root='/data/zhangpengjie/zhangpengjie/Workspace/Datasets', split='test', transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    tbar = tqdm(dataloader, desc='\r')
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (data, target) in enumerate(tbar):
        data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            output = model(data.float())

        # _, pred = output.topk(5, 1, True, True)
        # print(pred[0])
        # p = torch.softmax(output, dim=1)[0].cpu().numpy().tolist()
        # for i in range(5):
        #     no = int(pred[0].cpu().numpy().tolist()[i])
        #     print(p[no])
        # import cv2
        # image = data[0].cpu().numpy().transpose(1,2,0)
        # print(image.shape)
        # cv2.imwrite('./Test.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # import sys
        # sys.exit(0)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0])
        top5.update(acc5[0])
            
        tbar.set_description('Top1: %.3f | Top5: %.3f'%(top1.avg, top5.avg))

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
    parser.add_argument('--dataset', type=str, default='minc',
                        help='training dataset (default: minc)')
    # cuda, seed and logging
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
