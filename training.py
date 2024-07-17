import time, torchvision, argparse, sys, os
import torch, random
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import functools
from torchmetrics.functional import structural_similarity_index_measure as ssim

from networks.NAFNet_arch import NAFNet_wDetHead
from networks.UFPNet_code_uncertainty_arch import UFPNet_code_uncertainty, wnet
from networks.Uformer_arch import Uformer
from networks.denoising_diffusion_pytorch import GaussianDiffusion, Unet
from datasets.datasets_pairs import my_dataset, my_dataset_eval, my_dataset_wTxt, FusionDataset
from data.reflect_dataset import CEILDataset, CEILTestDataset, CEILTrainDataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from utils.UTILS import compute_psnr, MixUp_AUG, rand_bbox, compute_ssim
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from loss.perceptual import LossNetwork
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loss.contrastive_loss import HCRLoss
from networks.network_RefDet import RefDet, RefDetDual
from networks.cofeNet import Encode_Decode_seg_2scale_share_atten_res
from networks.Unet_arch import build_unet as UNet
import hashlib
from data.image_folder import read_fns

sys.path.append(os.getcwd())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.device_count() == 8:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3,4, 5,6,7"
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
if torch.cuda.device_count() == 4:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3"
    device_ids = [0, 1, 2, 3]
if torch.cuda.device_count() == 2:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_ids = [0, 1]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(20)

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str,
                    default="train")  # modify the experiments name-->modify all save path
parser.add_argument('--writer_dir', type=str, default='logs/')
parser.add_argument('--training_data_path', type=str, default='datasets/training_data/')
parser.add_argument('--eval_in_path_wild55', type=str, default='datasets/eval_data/')
parser.add_argument('--eval_gt_path_wild55', type=str,
                    default='datasets/eval_data/')
# training setting
parser.add_argument('--EPOCH', type=int, default=500)
parser.add_argument('--T_period', type=int, default=50)  # CosineAnnealingWarmRestarts
parser.add_argument('--BATCH_SIZE', type=int, default=2)
parser.add_argument('--Crop_patches', type=int, default=320)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--learning_rate_Det', type=float, default=0.0002)
parser.add_argument('--print_frequency', type=int, default=20)
parser.add_argument('--SAVE_Inter_Results', type=bool, default=True)
parser.add_argument('--SAVE_test_Results', type=bool, default=True)
# during training
parser.add_argument('--max_psnr', type=int, default=0)
parser.add_argument('--fix_sample', type=int, default=100000)
parser.add_argument('--lam_addition', type=float, default=0.1)
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--addition_loss', type=str, default='VGG')  # VGG
parser.add_argument('--hue_loss', type=str2bool, default=False)
parser.add_argument('--lam_hueLoss', type=float, default=0.1)
parser.add_argument('--others_loss', type=str, default='none')
parser.add_argument('--lam_othersLoss', type=float, default=0.1)
parser.add_argument('--Aug_regular', type=str2bool, default=False)
parser.add_argument('--MixUp_AUG', type=str2bool, default=False)

# training setting
parser.add_argument('--base_channel', type=int, default=32)
parser.add_argument('--base_channel_refineNet', type=int, default=24)
parser.add_argument('--num_block', type=int, default=6)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28], help='List of integers')
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1], help='List of integers')
parser.add_argument('--middle_blk_num', type=int, default=1)
parser.add_argument('--fusion_ratio', type=float, default=0.7)
parser.add_argument('--load_pre_model', type=str2bool, default=True)
parser.add_argument('--pre_model', type=str, default='')
parser.add_argument('--pre_model1', type=str, default='')
parser.add_argument('--pre_model_strict', type=str2bool, default=False)  # VGG
parser.add_argument('--eval_freq', type=int, default=27)

# network structure
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--hyper', type=str2bool, default=False)
parser.add_argument('--drop_flag', type=str2bool, default=True)
parser.add_argument('--drop_rate', type=float, default=0.4)
parser.add_argument('--augM', type=str2bool, default=False)
parser.add_argument('--in_norm', type=str2bool, default=False)
parser.add_argument('--pyramid', type=str2bool, default=False)
parser.add_argument('--global_skip', type=str2bool, default=True)
parser.add_argument('--adjust_loader', type=str2bool, default=False)

# syn data
parser.add_argument('--low_sigma', type=float, default=2, help='min sigma in synthetic dataset')
parser.add_argument('--high_sigma', type=float, default=5, help='max sigma in synthetic dataset')
parser.add_argument('--low_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
parser.add_argument('--high_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
parser.add_argument('--syn_mode', type=int, default=3)
parser.add_argument('--low_A', type=float, default=2, help='min sigma in synthetic dataset')
parser.add_argument('--high_A', type=float, default=5, help='max sigma in synthetic dataset')
parser.add_argument('--low_beta', type=float, default=1.3, help='max gamma in synthetic dataset')
parser.add_argument('--high_beta', type=float, default=1.3, help='max gamma in synthetic dataset')

# DDP
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--loss_char', type=str2bool, default=False)

# cutmix
parser.add_argument('--cutmix', type=str2bool, default=False, help='max gamma in synthetic dataset')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='max gamma in synthetic dataset')

parser.add_argument('--Det_model', type=str, default='None')  # VGG

parser.add_argument('--concat', type=str2bool, default=True, help='merge manner')
parser.add_argument('--merge_manner', type=int, default=0)

parser.add_argument('--TV_weights', type=float, default=0.00001, help='max gamma in synthetic dataset')
parser.add_argument('--save_pth_model', type=str2bool, default=True)

parser.add_argument('--s1_loss', type=str, default='None')  # VGG

parser.add_argument('--load_model_flag', type=int, default=0)

#  --in_norm   --pyramid
args = parser.parse_args()

if args.debug == True:
    fix_sampleA = 50
    fix_sampleB = 50
    fix_sampleC = 50
    print_frequency = 5

else:
    fix_sampleA = args.fix_sample
    fix_sampleB = args.fix_sample
    fix_sampleC = args.fix_sample
    print_frequency = args.print_frequency

exper_name = args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
# if not os.path.exists(args.writer_dir):
#     os.mkdir(args.writer_dir)
os.makedirs(args.writer_dir, exist_ok=True)

unified_path = args.unified_path
SAVE_PATH = unified_path + exper_name + '/'
if not os.path.exists(SAVE_PATH):
    # os.mkdir(SAVE_PATH,)
    os.makedirs(SAVE_PATH, exist_ok=True)

if args.SAVE_Inter_Results:
    SAVE_Inter_Results_PATH = unified_path + exper_name + '__inter_results/'
    if not os.path.exists(SAVE_Inter_Results_PATH):
        # os.mkdir(SAVE_Inter_Results_PATH)
        os.makedirs(SAVE_Inter_Results_PATH, exist_ok=True)

trans_eval = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])


def check_dataset(in_path, gt_path, name='RD'):
    print("Check {} length({}) len(in)==len(gt)?: {} ".format(name, len(os.listdir(in_path)),
                                                              os.listdir(in_path) == os.listdir(gt_path)))


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

MAX_PSNR = 0.0


def test(net, net_Det, eval_loader, epoch=1, iters=100, max_psnr_val=MAX_PSNR, Dname='S', SAVE_test_Results=False):
    global MAX_PSNR
    net.eval()
    net_Det.eval()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        eval_input_ssim = 0.0
        eval_output_ssim = 0.0
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            mask = wnet(inputs)
            sparse_out = net_Det(inputs)
            outputs = net(inputs, sparse_out)
            out_v = wnet(outputs)

            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)
            eval_input_ssim += compute_ssim(inputs, labels)
            eval_output_ssim += compute_ssim(outputs, labels)

            if SAVE_test_Results:
                SAVE_Test_Results_PATH = unified_path + exper_name + '_test_results/'
                os.makedirs(SAVE_Test_Results_PATH, exist_ok=True)
                Final_SAVE_Test_Results_PATH = SAVE_Test_Results_PATH + Dname + '/'
                os.makedirs(Final_SAVE_Test_Results_PATH, exist_ok=True)
                save_imgs_for_visual6(
                    Final_SAVE_Test_Results_PATH + name[0] + '-' + str(epoch) + '_' + str(iters) + '.jpg',
                    inputs, labels, outputs, sparse_out.repeat(1, 3, 1, 1), mask.repeat(1, 3, 1, 1),
                    out_v.repeat(1, 3, 1, 1))

        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        Final_input_SSIM = eval_input_ssim / len(eval_loader)
        Final_output_SSIM = eval_output_ssim / len(eval_loader)
        writer.add_scalars(exper_name + '/testing', {'eval_PSNR_Output': eval_output_psnr / len(eval_loader),
                                                     'eval_PSNR_Input': eval_input_psnr / len(eval_loader), }, epoch)
        if Final_output_PSNR > max_psnr_val:
            max_psnr_val = Final_output_PSNR
            MAX_PSNR = Final_output_PSNR
            print(
                "test-epoch:{},Dname:{},[Num_eval:{} In_PSNR:{}  Out_PSNR:{} In_SSIM:{}  Out_SSIM:{}]"
                "--------max_psnr_val:{}, cost time: {}".format(
                    epoch, Dname, len(eval_loader), round(Final_input_PSNR, 2),
                    round(Final_output_PSNR, 2), round(Final_input_SSIM, 3),
                    round(Final_output_SSIM, 3), round(max_psnr_val, 2), time.time() - st))


def save_imgs_for_visual(path, inputs, labels, outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path, nrow=3, padding=0)


def save_imgs_for_visual6(path, inputs, labels, outputs, sparse_out, mask, out_v):
    torchvision.utils.save_image(
        [inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0], sparse_out.cpu()[0], mask.cpu()[0], out_v.cpu()[0]], path,
        nrow=6, padding=0)


def save_imgs_for_visualR2(path, inputs, labels, outputs, inputs1, labels1, outputs1):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0],
                                  inputs1.cpu()[0], labels1.cpu()[0], outputs1.cpu()[0]], path, nrow=3, padding=0)


def get_training_data():
    train_pre_datasets = CEILTrainDataset("datasets/training_data/real_train", enable_transforms=True, )
    train_loader = DataLoader(dataset=train_pre_datasets, batch_size=args.BATCH_SIZE, num_workers=8, shuffle=True)
    print('len(train_loader):', len(train_loader))
    return train_loader


def get_eval_data(val_in_path=args.eval_in_path_nature20, val_gt_path=args.eval_gt_path_nature20
                  , trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label=val_gt_path, transform=trans_eval, fix_sample=500)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader


def get_erosion_dilation(input_mask, flag=True, kernel_size=3):
    if flag:
        temp_mask = -F.max_pool2d(-input_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return temp_mask
    else:
        return F.max_pool2d(input_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def obtain_sparse_reprentation(tensorA, tensorB):
    maxA = tensorA.max(dim=1)[0]
    maxB = tensorB.max(dim=1)[0]

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    A_grad_x = F.conv2d(maxA.unsqueeze(1), sobel_x, padding=1)
    A_grad_y = F.conv2d(maxA.unsqueeze(1), sobel_y, padding=1)
    grad1 = torch.sqrt(A_grad_x ** 2 + A_grad_y ** 2)

    B_grad_x = F.conv2d(maxB.unsqueeze(1), sobel_x, padding=1)
    B_grad_y = F.conv2d(maxB.unsqueeze(1), sobel_y, padding=1)
    grad2 = torch.sqrt(B_grad_x ** 2 + B_grad_y ** 2)

    mask = (grad1 > grad2).float()
    return mask


def ssim_loss(pred, target):
    ssim_value = ssim(pred, target)
    return 1 - ssim_value


def dice(pred, target, smooth=1e-5):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice_score
    return loss


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("==" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    if args.hyper:
        img_channel = args.img_channel + 1472
    else:
        img_channel = args.img_channel

    wnet = wnet(in_c=3, n_classes=1, layers=[8, 16, 32], conv_bridge=True, shortcut=True)
    checkpoint = torch.load('ckpt/lwnet.pth', map_location=device)
    wnet.load_state_dict(checkpoint['model_state_dict'])
    print('sucess!  load wnet(v_loss)')
    wnet = wnet.to(device)
    for param in wnet.parameters():
        param.requires_grad = False
    wnet.eval()

    net = UFPNet_code_uncertainty(img_channel=args.img_channel, width=args.base_channel,
                                  middle_blk_num=args.middle_blk_num,
                                  enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks,
                                  drop_flag=args.drop_flag, drop_rate=args.drop_rate)

    net_Det = RefDet(backbone='efficientnet-b3',
                     proj_planes=16,
                     pred_planes=32,
                     use_pretrained=True,
                     fix_backbone=False,
                     has_se=False,
                     num_of_layers=6,
                     expansion=4)

    if args.load_pre_model and (args.load_model_flag == 0):
        checkpoint = torch.load(args.pre_model)
        net.load_state_dict(checkpoint, strict=True)
        print('sucess!  load pre-model (removal)')

        checkpoint1 = torch.load(args.pre_model1)
        net_Det.load_state_dict(checkpoint1, strict=True)
        print('sucess!  load pre-model (detection)')

    net_Det.to(device)
    net.to(device)

    train_loader1_re = get_training_data()
    check_dataset(args.eval_in_path_wild55, args.eval_gt_path_wild55, 'val-wild55')
    eval_loader_wild55 = get_eval_data(val_in_path=args.eval_in_path_wild55, val_gt_path=args.eval_gt_path_wild55)

    net_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizerG = optim.Adam([{'params': net_params, 'lr': args.learning_rate},
                             {'params': net_Det.parameters(), 'lr': args.learning_rate / 2}],
                            betas=(0.9, 0.999))  # lr=args.learning_rate,
    scheduler = CosineAnnealingWarmRestarts(optimizerG, T_0=args.T_period,
                                            T_mult=1)  # ExponentialLR(optimizerG, gamma=0.98)

    # Losses
    loss_char = losses.CharbonnierLoss()
    if args.hue_loss:
        criterion_hue = losses.HSVLoss()
    if args.others_loss.lower() == 'hue':
        criterion = losses.HSVLoss()
    elif args.others_loss.lower() == 'ssim':
        criterion = losses.SSIMLoss()
    elif args.others_loss.lower() == 'contrast':
        criterion = HCRLoss()

    if args.addition_loss == 'VGG':
        vgg = models.vgg16()
        vgg.load_state_dict(torch.load('ckpt/vgg16-397923af.pth'))
        vgg_model = vgg.features[:16]
        vgg_model = vgg_model.to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        loss_network = LossNetwork(vgg_model)
        loss_network.eval()

    step = 0

    max_psnr_val_wild55 = args.max_psnr

    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    input_PSNR_all = 0
    train_PSNR_all = 0

    training_results = {'total_loss': 0.0, 'total_loss1': 0.0, 'total_loss2': 0.0,
                        'total_loss3': 0.0, 'input_PSNR_all': 0.0, 'train_PSNR_all': 0.0,
                        'total_S1_loss': 0.0, 'S1_loss': 0.0, 'S1_TV_loss': 0.0, }

    iter_nums = 0
    MAX_TRAIN_PSNR = 0

    for epoch in range(args.EPOCH):
        EACH_TRAIN_PSNR = 0
        st = time.time()
        if args.adjust_loader:
            if epoch < int(args.EPOCH * 0.7):
                train_loader = train_loader1_re
            else:
                train_loader = train_loader1_re
        else:
            train_loader = train_loader1_re
        scheduler.step(epoch)

        for i, train_data in enumerate(tqdm(train_loader), 0):
            save_test_results = False
            inputs, label, img_name = train_data['input'], train_data['target_t'], train_data['fn']
            data_in = inputs
            if i == 0:
                print("Check data: data.size: {} ,in_GT_mask: {}".format(data_in.size(), label.size()))
            iter_nums += 1
            net.train()
            net.zero_grad()
            optimizerG.zero_grad()

            net_Det.train()
            net_Det.zero_grad()

            data_sparse = get_erosion_dilation(obtain_sparse_reprentation(data_in, label))
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            mask = wnet(inputs)
            v_label = wnet(labels)

            labels_sparse = Variable(data_sparse).to(device)

            r = np.random.rand()
            if args.cutmix and (r < args.cutmix_prob):
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(inputs.size()[0]).to(device)  # cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1: bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                labels[:, :, bbx1: bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]

                sparse_out = net_Det(inputs)
                train_output = net(inputs, sparse_out.detach())
            else:
                sparse_out = net_Det(inputs)
                train_output = net(inputs, sparse_out.detach(), mask)
            v_out = wnet(train_output)

            input_PSNR = compute_psnr(inputs, labels)
            trian_PSNR = compute_psnr(train_output, labels)

            if args.s1_loss.lower() == 'char':
                S1_loss1 = loss_char(sparse_out, labels_sparse)
            else:
                S1_loss1 = losses.sigmoid_mse_loss(sparse_out, labels_sparse)
            S1_loss2 = losses.TVLoss(args.TV_weights)(sparse_out)
            S1_g_loss = S1_loss1 + S1_loss2

            if args.loss_char:
                loss1 = loss_char(train_output, labels)  # F.smooth_l1_loss(train_output, labels)
            else:
                loss1 = F.smooth_l1_loss(train_output, labels)
            if args.addition_loss == 'VGG':
                loss2 = args.lam_addition * loss_network(train_output, labels)
            elif args.addition_loss == 'FFT':
                loss2 = args.lam_addition * losses.fftLoss()(train_output, labels)
            else:
                loss2 = 0.01 * loss1

            if args.others_loss.lower() == 'none':
                loss3 = loss1
                g_loss = loss1 + loss2
            elif args.others_loss.lower() == 'contrast':
                loss3 = args.lam_othersLoss * HCRLoss(train_output, labels, inputs)
                g_loss = loss1 + loss2 + loss3
            else:
                loss3 = args.lam_othersLoss * criterion(train_output, labels)
                g_loss = loss1 + loss2 + loss3

            v_loss = ssim_loss(train_output, labels)
            dice_loss = dice(v_out, v_label)

            g_loss = g_loss + S1_g_loss

            training_results['total_loss'] += g_loss.item()
            training_results['total_loss1'] += loss1.item()
            training_results['total_loss2'] += loss2.item()
            training_results['total_loss3'] += loss3.item()

            training_results['total_S1_loss'] += S1_g_loss.item()
            training_results['S1_loss'] += S1_loss1.item()
            training_results['S1_TV_loss'] += S1_loss2.item()

            training_results['input_PSNR_all'] += input_PSNR
            training_results['train_PSNR_all'] += trian_PSNR

            g_loss.backward()
            optimizerG.step()

            writer.add_scalars(exper_name + '/training',
                               {'PSNR_Output': training_results['train_PSNR_all'] / iter_nums,
                                'PSNR_Input': training_results['input_PSNR_all'] / iter_nums, }, iter_nums)
            writer.add_scalars(exper_name + '/training',
                               {'total_loss': training_results['total_loss'] / iter_nums,
                                'loss1_char': training_results['total_loss1'] / iter_nums,
                                'loss2': training_results['total_loss2'] / iter_nums,
                                'loss3': training_results['total_loss3'] / iter_nums, }, iter_nums)
            print(
                "train-epoch:%d,iter:[%d / %d], [lr: %.7f ],[in_PSNR: %.3f, out_PSNR: %.3f],"
                "[final_loss:%.5f,RR:%.5f,v_loss:%.5f,dice_loss:%.5f, || "
                "RD_loss:%.5f],time:%.3f" %
                (epoch, i + 1, len(train_loader), optimizerG.param_groups[0]["lr"], input_PSNR, trian_PSNR,
                 g_loss.item(), loss1.item(),
                 v_loss.item(), dice_loss.item(), S1_loss1.item(), time.time() - st))
            st = time.time()
            EACH_TRAIN_PSNR += trian_PSNR

        EACH_TRAIN_PSNR /= len(train_loader)
        if EACH_TRAIN_PSNR >= MAX_TRAIN_PSNR and args.save_pth_model:
            save_test_results = True
            MAX_TRAIN_PSNR = EACH_TRAIN_PSNR
            save_model = SAVE_PATH + 'Net_epoch_{}_psnr_{}.pth'.format(epoch, round(EACH_TRAIN_PSNR, 4))
            torch.save(net.state_dict(), save_model)
            save_model_Det = SAVE_PATH + 'Net_Det_epoch_{}_psnr_{}.pth'.format(epoch, round(EACH_TRAIN_PSNR, 4))
            torch.save(net_Det.state_dict(), save_model_Det)

        test(net=net, net_Det=net_Det, eval_loader=eval_loader_wild55, epoch=epoch,
             max_psnr_val=MAX_PSNR, iters=iter_nums, Dname='wild55', SAVE_test_Results=save_test_results)
