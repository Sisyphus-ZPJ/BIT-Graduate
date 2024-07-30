import os
import numpy as np
import math
import random
import torch
import argparse
from tqdm import tqdm
import logging
from torchvision.transforms import *

from model.deepten import DeepTen
from datasets.dtd import DTDDataset
from metric import AverageMeter, accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
    
_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
    
def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logging.basicConfig(filename=os.path.join(args.save_path, 'log.txt'), level=logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info(args)
    
    # Load Model
    n_class = 47
    model = DeepTen(n_class, pretrained=args.pretrained).cuda()

    # Load Checkpoint
    # checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/model/deepten_resnet50_minc-1225f149.pth")
    # checkpoint = torch.load("/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/checkpoints/1/20.tar")['model_state_dict']
    # model.load_state_dict(checkpoint, strict=False)

    # Load Data
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([# Resize(base_size),
                                          RandomResizedCrop(320),
                                          RandomHorizontalFlip(),
                                          RandomVerticalFlip(),
                                          # ColorJitter(0.4, 0.4, 0.4),
                                          ToTensor(),
                                          # Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                                          normalize])
    transform_validate = transforms.Compose([CenterCrop(320),
                                             ToTensor(),
                                             normalize])
    train_dataset = DTDDataset(root=args.data_path, split='train', transform=transform_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validate_dataset = DTDDataset(root=args.data_path, split='validate', transform=transform_validate)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    # Load optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    
    # Loop
    top1 = AverageMeter()
    top5 = AverageMeter()
    rloss = AverageMeter()
    for epoch in range(args.epochs):
        tbar = tqdm(train_dataloader, desc='\r')
        for batch_idx, (data, target) in enumerate(tbar):
            data, target = data.cuda(), target.cuda()
            
            output = model(data.float())

            # _, pred = output.topk(1, 1, True, True)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0])
            top5.update(acc5[0])
            rloss.update(float(loss))
            
            tbar.set_description('Epoch: %d| Batch: %d| Top1: %.3f| Top5: %.3f| Loss: %.4f'%(epoch, batch_idx, top1.avg, top5.avg, rloss.avg))
            if batch_idx % 50 == 0:
                logging.info('(Train) Epoch: %d| Batch: %d| Top1: %.3f| Top5: %.3f | Loss: %.4f| LR: %.8f'%(epoch, batch_idx, top1.avg, top5.avg, rloss.avg, scheduler.get_last_lr()[0]))
                top1.reset()
                top5.reset()
                rloss.reset()

        scheduler.step()
        model.eval()
        tbar_validate = tqdm(validate_dataloader, desc='\r')
        top1_validate = AverageMeter()
        top5_validate = AverageMeter()
        for batch_idx, (data, target) in enumerate(tbar_validate):
            data, target = data.cuda(), target.cuda()
            
            with torch.no_grad():
                output = model(data.float())
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_validate.update(acc1[0])
            top5_validate.update(acc5[0])
        
        logging.info('Epoch: %d| Top1: %.3f| Top5: %.3f'%(epoch, top1_validate.avg, top5_validate.avg))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(args.save_path, str(epoch)+'.tar'))
        model.train()
        
    return

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
    # model
    parser.add_argument('--pretrained', type=bool, default=True)
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.95)
    # seed
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    # Dataset
    parser.add_argument('--data_path', type=str, default='/data/zhangpengjie/zhangpengjie/Workspace/Datasets')
    # checking point
    parser.add_argument('--save_path', type=str, 
                        default='/data/zhangpengjie/zhangpengjie/Workspace/Experiments/Texture/checkpoints/4_dtd')
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
