from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils.utils import *
import argparse
import numpy as np
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

        
seed = 1234
set_random_seed(seed)

DATASET = 'DFC22'     # ['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15']
SPLIT = '1-4'     # ['1-4', '1-8', '100', '300']
DFC22_DATASET_PATH = 'Your local path'
iSAID_DATASET_PATH = 'Your local path'
MER_DATASET_PATH = 'Your local path'
MSL_DATASET_PATH = 'Your local path'
Vaihingen_DATASET_PATH = 'Your local path'
GID15_DATASET_PATH = 'Your local path'

PERCENT = 20
LAMBDA = 1
AUG_PROCESS_MODE = 'SDC'   # ['SC', 'DS', 'DC', 'SDC']
NUM_CLASSES = {'DFC22': 12, 'iSAID': 15, 'MER': 9, 'MSL': 9, 'Vaihingen': 5, 'GID-15': 15}

def parse_args():
    parser = argparse.ArgumentParser(description='WSCL Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['GID-15', 'iSAID', 'DFC22', 'MER', 'MSL', 'Vaihingen'], default=DATASET)
    parser.add_argument('--percent', type=float, default=PERCENT, help='0~100, the low-entropy percent r')
    parser.add_argument('--lamb', type=int, default=LAMBDA, help='the trade-off weight to balance the supervised loss and the unsupervised loss')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv2')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/labeled.txt')
    parser.add_argument('--unlabeled-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/unlabeled.txt')
    parser.add_argument('--save-path', type=str, default='./exp/' + DATASET + '/' + SPLIT + '_' + str(PERCENT) + '_' + str(LAMBDA) + '/' + AUG_PROCESS_MODE)

    args = parser.parse_args()
    return args


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    create_path(args.save_path)

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=8, shuffle=False, pin_memory=False, num_workers=8, drop_last=False)

    trainset_u = SemiDataset(args.dataset, args.data_root, 'train_u', args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiDataset(args.dataset, args.data_root, 'train_l', args.crop_size, args.labeled_id_path, nsample=len(trainset_u.ids))

    trainloader_u = DataLoader(trainset_u, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=8, drop_last=True)
    trainloader_l = DataLoader(trainset_l, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=8, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))
    train(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, args)


def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, NUM_CLASSES[args.dataset])
    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)
    model = DataParallel(model).cuda()

    return model, optimizer


def train(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0
    weight_u = args.lamb

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))
        for i, ((img, mask), (img_u_w, img_u_s1, img_u_s2)) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w = model(img_u_w)
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)

            if np.random.uniform(0, 1) < 0.5:
                if AUG_PROCESS_MODE == 'SC':
                    conf_u_w, mask_u_w, img_u_s1 = generate_unsup_aug_sc(conf_u_w, mask_u_w, img_u_s1)
                elif AUG_PROCESS_MODE == 'DS':
                    img_u_s1 = generate_unsup_aug_ds(img_u_s1, img_u_s2)
                elif AUG_PROCESS_MODE == 'DC':
                    conf_u_w, mask_u_w, img_u_s1 = generate_unsup_aug_dc(conf_u_w, mask_u_w, img_u_s1, img_u_s2)
                else:
                    conf_u_w, mask_u_w, img_u_s1 = generate_unsup_aug_sdc(conf_u_w, mask_u_w, img_u_s1, img_u_s2)

            model.train()

            num_lb, num_ulb = img.shape[0], img_u_w.shape[0]
            preds = model(torch.cat((img, img_u_s1)))
            pred, pred_u_s = preds.split([num_lb, num_ulb])

            prob_u_s = pred_u_s.clone()
            em_u_s = entropy_map(prob_u_s.softmax(dim=1), 1)
            em_threshold = np.percentile(em_u_s.detach().cpu().numpy().flatten(), args.percent)

            loss_l = criterion(pred, mask)

            loss_u = criterion(pred_u_s, mask_u_w)
            loss_u = loss_u * (em_u_s <= em_threshold)
            loss_u = torch.mean(loss_u)

            loss = loss_l + weight_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u += loss_u.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u: %.3f' % (total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1)))

        if (epoch + 1) % 5 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            model.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)

                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()

                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            mIOU *= 100.0
            IOU *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))



if __name__ == '__main__':
    args = parse_args()
    if args.epochs is None:
        args.epochs = {'GID-15': 100, 'iSAID': 100, 'MER': 100, 'MSL': 100, 'Vaihingen': 100, 'DFC22': 100}[args.dataset]
    if args.lr is None:
        args.lr = {'GID-15': 0.001, 'iSAID': 0.001, 'MER': 0.001, 'MSL': 0.001,
                   'Vaihingen': 0.001, 'DFC22': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'GID-15': 321, 'iSAID': 321, 'MER': 321, 'MSL': 321, 'Vaihingen': 321, 'DFC22': 321}[args.dataset]
    if args.data_root is None:
        args.data_root = {'GID-15': GID15_DATASET_PATH,
                          'iSAID': iSAID_DATASET_PATH,
                          'MER': MER_DATASET_PATH,
                          'MSL': MSL_DATASET_PATH,
                          'Vaihingen': Vaihingen_DATASET_PATH,
                          'DFC22': DFC22_DATASET_PATH}[args.dataset]

    print(args)

    main(args)
