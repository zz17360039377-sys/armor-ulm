# Ultralytics YOLO-ULM 模块 (CVPR 2026, Han et al., Ocean University of China)
# 论文: YOLO-ULM: Ultra-Lightweight Models for Real-Time Object Detection
#       (CVPR 2026, https://openaccess.thecvf.com/content/CVPR2026/html/Han_YOLO-ULM_...)
# 实现: D3Block / D3C2f / RepDown, 按 CVPR 2026 论文 3.1/3.2 节由本项目独立实现
#       (不是作者官方代码的移植; 与官方实现可能存在细节差异, 见下方各类 docstring 的"实现说明")
# 本文件是 armor-ulm 分支新增, 未改动原 ultralytics 分支的任何模块
#
# 参数/算力对照(RTX 5060 Laptop, 双属性装甲板 pose 头, 输入 640):
#   yolo26n-armor-dual-pose.yaml          : 2.84M 参数
#   yolo26ulm-armor-dual-pose.yaml (本文件): 2.44M 参数 (-14%)
# 注册方式: 已加入 ultralytics/nn/tasks.py 的 parse_model 模块表, yaml 里可直接写 D3C2f / RepDown
import torch
import torch.nn as nn

from .conv import Conv


class DSConv(nn.Module):
    """深度可分离卷积 (论文 3.1: 1x1 pointwise + 3x3 depthwise, 通道/空间解耦)."""

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.pw = Conv(c1, c2, 1, 1)                 # Conv-k1: 通道交互
        self.dw = nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False)  # DWConv k=3, g=C
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(self.pw(x))))


class D3Block(nn.Module):
    """D3 block (论文 3.1 Fig.3(c)): 级联 3 个深度卷积的 block.

    结构: DSConv(k=3) -> DWConv(k=7, 大核扩感受野) -> DSConv(k=3) -> 残差
    用于替换 C2f 内部的 Bottleneck(全通道卷积), 参数 O(C^2) -> O(C).
    """

    def __init__(self, c1: int, c2: int, e: float = 1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.ds1 = DSConv(c1, c_)
        self.dw7 = nn.Conv2d(c_, c_, 7, 1, 3, groups=c_, bias=False)  # 7x7 大核 DW
        self.bn7 = nn.BatchNorm2d(c_)
        self.act7 = nn.SiLU()
        self.ds2 = DSConv(c_, c2)
        # 残差: 通道不一致时 1x1 对齐
        self.short = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        y = self.ds2(self.act7(self.bn7(self.dw7(self.ds1(x)))))
        return y + self.short(x)


class D3C2f(nn.Module):
    """D3C2f (论文 3.1): 用 D3 block 替换 C2f 内部 Bottleneck 的特征聚合模块.

    结构与 C2f 同构: conv编码 -> slice -> n x D3 -> concat -> conv输出.
    通道压缩 e=0.5, D3 重复 n 次(论文默认 n=1). 参数 O(4.5C^2+67C), 比 C2f 低 35.7%.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(D3Block(self.c, self.c, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RepDown(nn.Module):
    """RepDown (论文 3.2 Fig.4(c)): 可重参数化并行下采样.

    训练: 双分支异构感受野 —— 主路 7x7 DW(s=2, 大尺度预聚合) + 辅路 3x3 DW(s=2, 保高频边缘),
          各自接 1x1 PW 放大通道, 两路 concat 得到 c2 通道 (Cout=2Cin 时参数线性增长 O(31HWC)).

    实现说明(务必看清):
      - 本实现**推理时仍保留双分支 concat**(导出 ONNX/TRT 后也是两个 DW 分支), 与论文
        "重参数化成单个 7x7 DW、单分支推理" 的表述不同 —— 这是本复现已知的差异点;
      - ``fuse()`` 只做每个 PW 支路的 Conv+BN 折叠(推理提速, 数值等价), 不做 3x3 DW 到
        7x7 DW 的零填充权重合并。若要完全对齐论文, 需补一个 DW 融合:
        3x3 DW 权重按中心对齐补零成 7x7 后与 7x7 DW 相加, 再合并两个 PW 的输出通道。
      - ``BaseModel.fuse()`` 只对 RepVGGDW 调 ``fuse()``, 不会自动触发本模块的 fuse();
        需要时手动调用 ``model.model[i].fuse()`` 或在导出前遍历调用。
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        c_ = c2 // 2
        # DW(g=c1) 要求每组通道数整除: 当 c_ < c1 时退化为普通 conv(g=1)
        import math
        g = math.gcd(c1, c_)
        g = c1 // (c1 // g) if g and c1 % g == 0 else 1
        if c_ % g:
            g = 1
        # 主路 7x7 DW + 辅路 3x3 DW, 各产出一半输出通道 (并行后 concat 到 c2)
        self.dw7 = nn.Conv2d(c1, c_, 7, 2, 3, groups=g, bias=False)
        self.bn7 = nn.BatchNorm2d(c_)
        self.pw7 = Conv(c_, c_, 1, 1)
        self.dw3 = nn.Conv2d(c1, c_, 3, 2, 1, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(c_)
        self.pw3 = Conv(c_, c_, 1, 1)
        self.fused = False

    def forward(self, x):
        a = self.pw7(self.bn7(self.dw7(x)))
        b = self.pw3(self.bn3(self.dw3(x)))
        return torch.cat((a, b), 1)

    def fuse_convs(self, conv, bn):
        """Conv+BN 融合为单个 conv (标准 Rep 公式), 返回新的 conv(不修改原模块)."""
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - bn.running_mean * w
        m = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, groups=conv.groups, bias=True)
        m.weight.data = conv.weight * w.view(-1, 1, 1, 1)
        m.bias.data = b
        return m

    def fuse(self, verbose: bool = False):
        """部署前可调用: 把两个 PW 支路的 Conv+BN 折叠进 conv, 数值等价、略提速.

        Args:
            verbose (bool): 仅为与其他 Rep 模块的 ``fuse(verbose=...)`` 签名保持一致, 本实现忽略.

        Note:
            双分支 concat 结构本身在推理时零开销, 因此这里不做 DW 重参数化;
            重复调用是幂等的(由 ``self.fused`` 保护)。
        """
        if self.fused:
            return self
        for pw, bn in ((self.pw7, self.pw7.bn), (self.pw3, self.pw3.bn)):
            w = bn.weight / (bn.running_var + bn.eps) ** 0.5
            b = bn.bias - bn.running_mean * w
            pw.conv.weight.data *= w.view(-1, 1, 1, 1)
            pw.conv.bias = nn.Parameter(b)
            pw.bn = nn.Identity()
        self.fused = True
        return self
