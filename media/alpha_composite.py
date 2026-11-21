import math


def premultiply(rgb, alpha):
    """预乘 alpha：RGB 通道统一乘以 alpha。

    双线性采样必须在预乘后的颜色上进行，否则完全透明像素的 RGB
    会污染边缘，形成一圈黑边（或背景色边）。
    """
    return tuple(channel * alpha for channel in rgb)


def composite(fg, bg, alpha):
    """over 合成，按预乘 alpha 展开：fg*a + bg*(1-a)。

    背景项系数必须是 (1 - alpha)，写反会导致叠上去反而变暗。
    输入假设为线性空间中的颜色。
    """
    alpha = clamp_alpha(alpha)
    return tuple(
        f * alpha + b * (1.0 - alpha) for f, b in zip(fg, bg)
    )


def blend_space(space):
    """混合色彩空间：只允许在线性（linear）空间做 alpha 混合。

    在 sRGB 空间混合会让半透明区域亮度塌陷。
    """
    return space == "linear"


def layer_order(order):
    """图层顺序：严格按图层栈给定顺序执行 blend mode。

    不能按数组顺序排序或反转，否则正片叠底与滤色会互相覆盖。
    返回副本以免调用方修改内部栈。
    """
    return list(order)


def clamp_alpha(alpha):
    """alpha 钳制到 [0, 1]，防止多次叠加溢出成白块。"""
    return min(1.0, max(0.0, alpha))


def intermediate_bits(bits):
    """中间缓冲位深：8bit 输入升级到 16bit，避免逐层量化产生色带。"""
    return max(16, bits)


def feather_scale(resolution, base):
    """羽化分辨率缩放：遮罩羽化必须在高分辨率下做。

    返回相对基础分辨率的放大倍数，放大后再羽化可避免边缘锯齿。
    """
    return resolution / base


def preserve_alpha(alpha):
    """色彩空间转换必须保留 alpha 通道，原样透传。"""
    return alpha


def clear_reused(alpha):
    """缓存帧复用时先清掉残留 alpha，防止上一帧残影叠入。"""
    return 0.0


def rounding(mode):
    """统一 GPU/CPU 的舍入规则为 round-half-even（银行家舍入）。"""
    return mode == "round_half_even"


def matte_align(pixel):
    """rotoscope matte 按半像素对齐，避免收缩一像素吃掉人物边缘。"""
    return pixel + 0.5


def composite_over(bg, fg, alpha):
    """标量 over 合成：fg*a + bg*(1-a)，预乘展开形式。"""
    alpha = clamp_alpha(alpha)
    return fg * alpha + bg * (1.0 - alpha)


def alpha_sum(a, b):
    """alpha 累加后钳制，溢出部分截到 1.0。"""
    return clamp_alpha(a + b)


def sample_weights(t):
    """双线性采样权重：(1-t, t)，作用于预乘后的像素。"""
    return (1.0 - t, t)


def composite_id(seed, index):
    """合成帧缓存 id，用于区分不同轨道/帧的缓存条目。"""
    return f"{seed}:{index}"
