import time, torchvision, argparse, sys, os
import torch, random
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim

from datasets.datasets_pairs import my_dataset, my_dataset_eval, my_dataset_wTxt, FusionDataset
import torchvision.transforms as transforms

from networks.UFPNet_code_uncertainty_arch import UFPNet_code_uncertainty
from utils.UTILS import compute_psnr, MixUp_AUG, rand_bbox, compute_ssim
import matplotlib.image as img

from networks.network_RefDet import RefDet, RefDetDual
from networks.Unet_arch import build_unet as UNet
from networks.NAFNet_arch import NAFNet_wDetHead  # NAFNetLocal
from networks.Uformer_arch import Uformer

sys.path.append(os.getcwd())

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
                    default="testing_NAFNet_512")  # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str, default='')
# parser.add_argument('--model_save_dir', type=str, default= )#required=True
parser.add_argument('--training_data_path', type=str,
                    default='/gdata1/zhuyr/Deref/training_data/')
# parser.add_argument('--training_data_path_Txt', type=str, default='/gdata1/zhuyr/Deref/training_data/Ref_HZ1.txt')
parser.add_argument('--training_data_path_Txt', nargs='*', help='a list of strings')
parser.add_argument('--training_data_path_Txt1', nargs='*', help='a list of strings')
# --experiment_name SIRRwPreD --EPOCH 150 --T_period 50 --Crop_patches 320 --training_data_path_Txt '/mnt/data_oss/ReflectionData/SIRR_USTC/DeRef_USTC_wPreD.txt'

parser.add_argument('--writer_dir', type=str, default='/gdata1/zhuyr/Deref/Deref_RW_writer_logs/')

parser.add_argument('--eval_in_path_nature20', type=str, default='/gdata1/zhuyr/Deref/training_data/nature20/blended//')
parser.add_argument('--eval_gt_path_nature20', type=str,
                    default='/gdata1/zhuyr/Deref/training_data/nature20/transmission_layer/')

parser.add_argument('--eval_in_path_real20', type=str, default='/gdata1/zhuyr/Deref/training_data/real20/blended/')
parser.add_argument('--eval_gt_path_real20', type=str,
                    default='/gdata1/zhuyr/Deref/training_data/real20/transmission_layer/')

parser.add_argument('--eval_in_path_wild55', type=str, default='datasets/eval_data/wild55/blended/')
parser.add_argument('--eval_gt_path_wild55', type=str,
                    default='datasets/eval_data/wild55/transmission_layer/')

parser.add_argument('--eval_in_path_soild200', type=str, default='/gdata1/zhuyr/Deref/training_data/solid200/blended/')
parser.add_argument('--eval_gt_path_soild200', type=str,
                    default='/gdata1/zhuyr/Deref/training_data/solid200/transmission_layer/')

parser.add_argument('--eval_in_path_postcard199', type=str,
                    default='/gdata1/zhuyr/Deref/training_data/postcard199/blended/')
parser.add_argument('--eval_gt_path_postcard199', type=str,
                    default='/gdata1/zhuyr/Deref/training_data/postcard199/transmission_layer/')

parser.add_argument('--eval_in_path_SIR', type=str, default='/gdata1/zhuyr/Deref/training_data/SIR/blended/')
parser.add_argument('--eval_gt_path_SIR', type=str, default='/gdata1/zhuyr/Deref/training_data/SIR/transmission_layer/')

parser.add_argument('--eval_in_path_RW', type=str, default='/gdata1/zhuyr/Deref/training_data/real_wogt/blended/')

parser.add_argument('--SAVE_test_Results', type=bool, default=True)

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

parser.add_argument('--load_pre_model', type=str2bool, default=True)  # VGG
parser.add_argument('--pre_model', type=str, default='./train/NAFNet/qianyi_Net_epoch_1_iters_112_psnr_53.2157.pth')
parser.add_argument('--pre_model1', type=str,
                    default='./train/NAFNet/qianyi_Net_Det_epoch_1_iters_112_psnr_53.2157.pth')

parser.add_argument('--pre_model_strict', type=str2bool, default=False)  # VGG

parser.add_argument('--eval_freq', type=int, default=2000)

# network structure

parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--hyper', type=str2bool, default=False)
parser.add_argument('--drop_flag', type=str2bool, default=False)
parser.add_argument('--drop_rate', type=float, default=0.4)

parser.add_argument('--augM', type=str2bool, default=False)
parser.add_argument('--in_norm', type=str2bool, default=False)
parser.add_argument('--pyramid', type=str2bool, default=False)
parser.add_argument('--global_skip', type=str2bool, default=False)

parser.add_argument('--adjust_loader', type=str2bool, default=False)

parser.add_argument('--Det_model', type=str, default='None')  # VGG

parser.add_argument('--concat', type=str2bool, default=True, help='merge manner')
parser.add_argument('--merge_manner', type=int, default=0)

parser.add_argument('--TV_weights', type=float, default=0.00001, help='max gamma in synthetic dataset')
parser.add_argument('--save_pth_model', type=str2bool, default=True)

parser.add_argument('--s1_loss', type=str, default='None')  # VGG

parser.add_argument('--load_model_flag', type=int, default=0)

#  --in_norm   --pyramid
args = parser.parse_args()

exper_name = args.experiment_name
unified_path = args.unified_path

trans_eval = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize((256, 256))
    ])

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test(net, net_Det, eval_loader, Dname='S', SAVE_test_Results=False):
    net.eval()
    net_Det.eval()
    with torch.no_grad():
        eval_results = {'eval_input_psnr': 0.0, 'eval_output_psnr': 0.0,
                        'eval_input_ssim': 0.0, 'eval_output_ssim': 0.0,
                        'infer_time': 0.0}
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):  # enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            infer_st = time.time()
            sparse_out = net_Det(inputs)
            outputs = net(inputs, sparse_out)

            # labels_gbr = labels[:, [2, 1, 0], :, :]
            # outputs_gbr = outputs[:, [2, 1, 0], :, :]
            # v_label = v_net(labels_gbr)
            # v_output = v_net(outputs_gbr)
            #
            # import matplotlib.pyplot as plt
            # img = v_label[0, 0].detach().cpu().numpy()
            # plt.imshow(img, cmap='gray')
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            #
            # img = v_output[0, 0].detach().cpu().numpy()
            # plt.imshow(img, cmap='gray')
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()

            eval_results['infer_time'] += time.time() - infer_st
            eval_results['eval_input_psnr'] += compute_psnr(inputs, labels)
            eval_results['eval_output_psnr'] += compute_psnr(outputs, labels)
            eval_results['eval_input_ssim'] += compute_ssim(inputs, labels)
            eval_results['eval_output_ssim'] += compute_ssim(outputs, labels)

            if SAVE_test_Results:
                SAVE_Test_Results_PATH = unified_path + exper_name + '__test_results/'
                os.makedirs(SAVE_Test_Results_PATH, exist_ok=True)

                Final_SAVE_Test_Results_PATH = SAVE_Test_Results_PATH + Dname + '/'
                SINGLE_IMAGE_SAVE_PATH = Final_SAVE_Test_Results_PATH + "/noCompare/"
                os.makedirs(Final_SAVE_Test_Results_PATH, exist_ok=True)
                os.makedirs(SINGLE_IMAGE_SAVE_PATH, exist_ok=True)
                save_imgs_for_single(SINGLE_IMAGE_SAVE_PATH + name[0], outputs)
                save_imgs_for_visual4(
                    Final_SAVE_Test_Results_PATH + name[0],
                    inputs, labels, outputs, sparse_out.repeat(1, 3, 1, 1))

        Final_output_PSNR = eval_results['eval_output_psnr'] / len(eval_loader)
        Final_input_PSNR = eval_results['eval_input_psnr'] / len(eval_loader)
        Final_output_SSIM = eval_results['eval_output_ssim'] / len(eval_loader)
        Final_input_SSIM = eval_results['eval_input_ssim'] / len(eval_loader)

        print(
            "Dname:{}-------[Num_eval:{} In_PSNR:{}  Out_PSNR:{} , In_SSIM:{}  Out_SSIM:{}], [total cost time: {} || total infer time:{} avg infer time:{} ]".format(
                Dname, len(eval_loader), round(Final_input_PSNR, 4),
                round(Final_output_PSNR, 4), round(Final_input_SSIM, 4),
                round(Final_output_SSIM, 4), time.time() - st, eval_results['infer_time'],
                                             eval_results['infer_time'] / len(eval_loader)))


def save_imgs_for_visual(path, inputs, labels, outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path, nrow=3, padding=0)


def save_imgs_for_single(path, outputs):
    torchvision.utils.save_image(outputs.cpu()[0], path)


def save_imgs_for_visual4(path, inputs, labels, outputs, sparse_out):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0], sparse_out.cpu()[0]], path,
                                 nrow=4, padding=0)


def save_imgs_for_visualR2(path, inputs, labels, outputs, inputs1, labels1, outputs1):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0],
                                  inputs1.cpu()[0], labels1.cpu()[0], outputs1.cpu()[0]], path, nrow=3, padding=0)


def get_eval_data(val_in_path=args.eval_in_path_nature20, val_gt_path=args.eval_gt_path_nature20
                  , trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label=val_gt_path, transform=trans_eval, fix_sample=500)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader


def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))


def check_dataset(in_path, gt_path, name='RD'):
    print("Check {} length({}) len(in)==len(gt)?: {} ".format(name, len(os.listdir(in_path)),
                                                              os.listdir(in_path) == os.listdir(gt_path)))


'''
python  /ghome/zhuyr/Deref_RW/testing_reflection_wNAFNetwDetEnc_wDDP_wJointDetSparse_V3_saveImg.py  --experiment_name testing-final   --enc_blks 1 1 1 28  --middle_blk_num 1  --dec_blks 1 1  1  1  --concat True  --merge_manner 0   --pre_model /gdata1/zhuyr/Deref/Deref_RW/DeRWref_wFusion-wFT-V3_10_5/SIRR_best-postcard199-wEMA__24.1_Epoch-0_iters-190.pth   --pre_model1  /gdata1/zhuyr/Deref/Deref_RW/DeRWref_wFusion-wFT-V3_10_3/SIRR_Det_best-real20__23.8_Epoch-0_iters-50.pth  --load_pre_model True    
'''
if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("==" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    net = NAFNet_wDetHead(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.middle_blk_num,
                          enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=args.global_skip,
                          drop_flag=args.drop_flag, drop_rate=args.drop_rate,
                          concat=args.concat, merge_manner=args.merge_manner)
    # net = Uformer(img_size=256, embed_dim=16, win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff',
    #               modulator=True)
    # net = UFPNet_code_uncertainty(img_channel=args.img_channel, width=args.base_channel,
    #                               middle_blk_num=args.middle_blk_num,
    #                               enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks,
    #                               drop_flag=args.drop_flag, drop_rate=args.drop_rate)
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

    # wnet = UNet()
    # checkpoint = torch.load('ckpt/unet.pth', map_location=device)
    # wnet.load_state_dict(checkpoint)
    # print('sucess!  load wnet(v_loss)')
    # wnet = wnet.to(device)
    # for param in wnet.parameters():
    #     param.requires_grad = False  # 冻结 wnet 的所有参数
    # wnet.eval()

    check_dataset(args.eval_in_path_wild55, args.eval_gt_path_wild55, 'val-wild55')
    eval_loader_wild55 = get_eval_data(val_in_path=args.eval_in_path_wild55, val_gt_path=args.eval_gt_path_wild55)

    test(net=net, net_Det=net_Det, eval_loader=eval_loader_wild55, Dname='wild55', SAVE_test_Results=True)
