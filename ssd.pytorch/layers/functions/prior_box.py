from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    """
    1、计算先验框，根据feature map的每个像素生成box;
    2、框的中个数为： 38×38×4+19×19×6+10×10×6+5×5×6+3×3×4+1×1×4=8732
    3、 cfg: SSD的参数配置，字典类型
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # 300
        self.image_size = cfg['img_size']
        # number of priors for feature map location (either 4 or 6)

        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        # 特征图的大小 [38, 19, 10, 5, 3, 1]
        self.feature_maps = cfg['feature_maps']
        # 相对于原图来说框实际的大小
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        # 相当在原图上每个框的中心的跨度
        # 比如在38×38特征图上，每个框的中心相距为8（这个8相当于原图来说）
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # 遍多尺度的 map: [38, 19, 10, 5, 3, 1]
        # 遍历每个特征图
        for k, f in enumerate(self.feature_maps):
            # 第k层的futuremap大小
            f_k = self.image_size / self.steps[k]
            # f_k 其实就是相当于futuremap的大小，感觉没差别
            # 遍历图每个像素点
            for i, j in product(range(f), repeat=2):
                # product为笛卡尔积，比如0-1，repeat=2就是生成(0,0),(0,1),(1,0),(1,1),
                # 二维各不相同
                # unit center x,y
                # 相对于特征图的偏移量，也是相对于原图的偏移量，因为都是相对长度
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # 框的长宽，也是相对于原图的大小而言
                # 长宽比为1的小框
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]
                # 长宽比为1的大框
                # rel size: sqrt(s_k * s_(k+1))
                # s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                s_k_prime = sqrt(self.min_sizes[k]*self.max_sizes[k])/self.image_size
                mean += [cx, cy, s_k_prime, s_k_prime]
                # 长宽比为其他的框
                # rest of aspect ratios
                # 因为 1,5,6，特征图只有4个框，2,3,4有6个框
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            # 归一化
            # 加了下划线是inplace方法
            output.clamp_(max=1, min=0)
        return output

# # 调试代码
# if __name__ == "__main__":
#     # SSD300 CONFIGS
#     voc = {
#         'num_classes': 21,
#         'lr_steps': (80000, 100000, 120000),
#         'max_iter': 120000,
#         'feature_maps': [38, 19, 10, 5, 3, 1],
#         'img_size': 300,
#         'steps': [8, 16, 32, 64, 100, 300],
#         'min_sizes': [30, 60, 111, 162, 213, 264],
#         'max_sizes': [60, 111, 162, 213, 264, 315],
#         # 特征图的长宽比
#         'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#         'variance': [0.1, 0.2],
#         'clip': True,
#         'name': 'VOC',
#     }
#     box = PriorBox(voc)
#     print('Priors box shape:', box.forward().shape)
#     print('Priors box:\n',box.forward())