import torch.nn.functional as F
import torch
from PIL import Image
import torchvision.transforms as transforms


def get_erosion_dilation(input_mask, flag=True, kernel_size=3):
    # B*3*H*W, B*1*H*W
    if flag:  # flag True 腐蚀操作
        temp_mask = -F.max_pool2d(-input_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return temp_mask
    else:  # Flase 膨胀操作
        return F.max_pool2d(input_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


# 该函数的作用是通过计算图像A和图像B的边缘强度梯度，并生成一个掩码来表示哪些位置的边缘强度在图像A中更强
def obtain_sparse_reprentation(tensorA, tensorB):
    maxA = tensorA.max(dim=1)[0]
    maxB = tensorB.max(dim=1)[0]
    # 定义 sobel 滤波器
    # sobel_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    A_grad_x = F.conv2d(maxA.unsqueeze(1), sobel_x, padding=1)
    A_grad_y = F.conv2d(maxA.unsqueeze(1), sobel_y, padding=1)
    grad1 = torch.sqrt(A_grad_x ** 2 + A_grad_y ** 2)

    B_grad_x = F.conv2d(maxB.unsqueeze(1), sobel_x, padding=1)
    B_grad_y = F.conv2d(maxB.unsqueeze(1), sobel_y, padding=1)
    grad2 = torch.sqrt(B_grad_x ** 2 + B_grad_y ** 2)

    # 比较 grad1 和 grad2 的值，如果 grad1 大于 grad2 的位置记为 1，其他设置为 0
    mask = (grad1 > grad2).float()
    return mask


# 读取图像A
image_A_path = 'path_to_image_A.jpg'  # 替换为图像A的文件路径
image_A = Image.open(image_A_path).convert('RGB')

# 读取图像B
image_B_path = 'path_to_image_B.jpg'  # 替换为图像B的文件路径
image_B = Image.open(image_B_path).convert('RGB')

# 定义图像预处理的转换
transform = transforms.ToTensor()

# 将图像转换为张量
tensor_A = transform(image_A).unsqueeze(0)  # 添加一个维度表示批次大小
tensor_B = transform(image_B).unsqueeze(0)  # 添加一个维度表示批次大小

# 调用 obtain_sparse_reprentation 函数并获取结果
data_sparse = get_erosion_dilation(obtain_sparse_reprentation(tensor_A, tensor_B))
