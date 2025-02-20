import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from thop import profile

NEG_INF = -1000000


# ##############------DropOut------###############
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# ##############------DropOut------###############


# ##############------Multi_Layer_Perceptron------###############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ##############------Multi_Layer_Perceptron------###############


# ##################---------Window_Attention_D---------###################
class WindowAttention_D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        # Arguments
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape

        # #########------q, k, v------##########
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # #########------q, k, v------##########

        # #########------q*k------##########
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale,
                                  max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn = attn * logit_scale
        # #########------q*k------##########

        # #########--------Relative_Position_Bias--------##########
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        # #########--------Relative_Position_Bias--------##########

        # #########------masking+softmax------##########
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        # #########------masking+softmax------##########

        attn = self.attn_drop(attn)

        # #########------(qk)*v+Linear------##########
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        # #########------(qk)*v+Linear------##########

        x = self.proj_drop(x)

        return x


# ##################---------Window_Attention_D---------###################


# ##################---------Window_Attention_S---------###################
class WindowAttention_S(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()

        # Arguments
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, rpi, mask=None, sp_mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape

        # #########------q, k, v------##########
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # #########------q, k, v------##########

        # #########------q*k------##########
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale,
                                  max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn = attn * logit_scale
        # #########------q*k------##########

        # #########--------Relative_Position_Bias--------##########
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        # #########--------Relative_Position_Bias--------##########

        # #########------masking+softmax------##########
        if sp_mask is not None:
            nP = sp_mask.shape[0]
            attn = attn.view(b_ // nP, nP, self.num_heads, n, n) + sp_mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, n, n)
            if mask is not None:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        # #########------masking+softmax------##########

        attn = self.attn_drop(attn)

        # #########------(qk)*v+Linear------##########
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        # #########------(qk)*v+Linear------##########

        x = self.proj_drop(x)

        return x

# ##################---------Window_Attention_S---------###################


# ##############------Rectangular_Window_Partition------###############
def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size,
                                                            c)
    return windows

# ##############------Rectangular_Window_Partition------###############


# ##############------Rectangular_Window_Reverse------###############
def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

# ##############------Rectangular_Window_Reverse------###############


# ##############------Triangular_Window_Partition------###############
def window_partition_triangular(x, window_size, masks):
    b, h, w, c = x.shape
    m = len(masks)  # 4
    ws = window_size  # 2*win
    h_ws = h // ws
    w_ws = w // ws
    x = x.view(b, h_ws, ws, w_ws, ws, c)  # b, h/ws, ws, w/ws, ws, c
    windows = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, ws, ws)
    # b, h/ws, w/ws, c, ws, ws-->b*(h_ws)*(w_ws)*c, ws, ws
    # window_mask=torch.zeros((len(masks), windows.shape[0], ws//2 * ws//2), dtype=windows.dtype).to(x.device)
    window_masks = []
    for mask in masks:
        mask = mask.expand(windows.shape[0], -1, -1)  # b*(h_ws)*(w_ws)*c, ws, ws
        window_mask = windows[mask]
        window_masks.append(window_mask.unsqueeze(0))  # 4, b*(h_ws)*(w_ws)*c, ws, ws
    window_masks = torch.cat(window_masks, dim=0)
    window_masks = window_masks.view(m, windows.shape[0], -1)  # 4, b*(h_ws)*(w_ws)*c, ws*ws
    m, _, n = window_masks.shape
    window_masks = window_masks.view(m, -1, c, n).permute(1, 0, 3, 2).contiguous()
    # [m, b*(h_ws)*(w_ws)*c, n]->[b*(h_ws)*(w_ws), m, n, c]
    return window_masks

# ##############------Triangular_Window_Partition------###############


# ##############------Triangular_Window_Reverse------###############
def window_reverse_triangular(x, window_size, masks):
    b_, m, n, c = x.shape  # [b*(h_ws)*(w_ws), m, n, c]
    x = x.permute(1, 0, 3, 2).contiguous().view(m, -1)  # [m, b*(h_ws)*(w_ws)*c, n]
    reconstructed = torch.zeros((b_ * c, window_size, window_size), dtype=x.dtype).to(x.device)
    for mask, x_ in zip(masks, x):
        mask = mask.expand(b_ * c, -1, -1)
        reconstructed[mask] = x_  # [b*(h_ws)*(w_ws)*c, ws, ws]
    return reconstructed

# ##############------Triangular_Window_Reverse------###############


# ##################---------DAB---------###################
class DAB(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=16,
                 shift_size=0,
                 interval=0,
                 triangular_flag=0,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.interval = interval
        self.mlp_ratio = mlp_ratio
        self.triangular_flag = triangular_flag
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_D(dim,
                                      window_size=to_2tuple(self.window_size),
                                      num_heads=num_heads,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      attn_drop=attn_drop,
                                      proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask, triangular_masks):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"
        shortcut = x

        # ###--------####
        x = x.view(b, h, w, c)
        # ###--------####

        # #######################-------Start_(S)W-MSA-------########################

        # ######----Cyclic_Shift + Mask----########
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.shift_size == self.window_size // 2:
                attn_mask = attn_mask[0]
            if self.shift_size == self.window_size:
                attn_mask = attn_mask[1]
            if self.shift_size == self.window_size // 2 * 3:
                attn_mask = attn_mask[2]
        else:
            shifted_x = x
            attn_mask = None
        # ######----Cyclic_Shift + Mask----########

        # ######----Partition_Windows----########
        if not self.triangular_flag:
            x_windows = window_partition(shifted_x,
                                         self.window_size)  # nw*b, window_size, window_size, c
            x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                       c)  # nw*b, window_size*window_size, c
        else:
            x_windows = window_partition_triangular(shifted_x, 2 * self.window_size,
                                                    triangular_masks)  # nw*b, window_size, window_size, c
            _, m, n, _ = x_windows.shape  # [b*(h_ws)*(w_ws), m, n, c]
            x_windows = x_windows.view(-1, n, c)  # [b*(h_ws)*(w_ws)*m, n, c]
        # ######----Partition_Windows----########

        # ######----W-MSA/SW-MSA----########
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)
        # ######----W-MSA/SW-MSA----########

        # ######----Merge_Windows----########
        if self.triangular_flag:
            attn_windows = attn_windows.view(-1, m, n, c)  # [b*(h_ws)*(w_ws), m, n, c]
            shifted_x = window_reverse_triangular(attn_windows, 2 * self.window_size,
                                                  triangular_masks)  # nw*b, window_size, window_size, c
            shifted_x = shifted_x.view(b, h // (2 * self.window_size), w // (2 * self.window_size), c,
                                       2 * self.window_size, 2 * self.window_size)
            shifted_x = shifted_x.permute(0, 1, 4, 2, 5, 3).contiguous().view(b, h, w, c)
        else:
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_x = window_reverse(attn_windows, self.window_size, h,
                                       w)  # b h' w' c
        # ######----Merge_Windows----########

        # ######----Reverse_Cyclic_Shift----########
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        # ######----Reverse_Cyclic_Shift----########

        attn_x = attn_x.view(b, h * w, c)

        # #######################--------End_(S)W-MSA--------########################

        # ###--------####
        x = shortcut + self.drop_path(self.norm1(attn_x))
        # ###--------####

        # ###----LN+MLP----####
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        # ###----LN+MLP----####

        return x

# ##################---------DAB---------###################


# ##################---------SAB---------###################
class SAB(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=16,
                 shift_size=0,
                 interval=2,
                 triangular_flag=0,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.interval = interval
        self.mlp_ratio = mlp_ratio
        self.triangular_flag = triangular_flag
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention_S(dim,
                                      window_size=to_2tuple(self.window_size),
                                      num_heads=num_heads,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      attn_drop=attn_drop,
                                      proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask, triangular_masks):
        h, w = x_size
        b, l, c = x.shape

        # ###--------####
        assert l == h * w, "input feature has wrong size %d, %d, %d" % (l, h, w)
        if min(h, w) <= self.window_size:
            self.window_size = min(h, w)  # Won't partition, if window size is larger than input resolution
        # ###--------####

        shortcut = x

        # ###--------####
        x = x.view(b, h, w, c)
        # ###--------####

        # ###----padding----####
        size_par = self.interval
        pad_l = pad_t = 0
        pad_r = (size_par - w % size_par) % size_par
        pad_b = (size_par - h % size_par) % size_par
        x = F.pad(x,
                  (0, 0, pad_l, pad_r, pad_t, pad_b))
        # ###----padding----####

        _, Hd, Wd, _ = x.shape

        # ###----Masking----####
        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1
        # ###----Masking----####

        # #######################-------Start_(S)W-MSA-------########################

        # ######----Cyclic_Shift + Mask----########
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.shift_size == self.window_size // 2:
                attn_mask = attn_mask[0]
            elif self.shift_size == self.window_size:
                attn_mask = attn_mask[1]
            elif self.shift_size == self.window_size // 2 * 3:
                attn_mask = attn_mask[2]
        else:
            shifted_x = x  # [1, 64, 64, 180]
            attn_mask = None
        # ######----Cyclic_Shift + Mask----########

        # #######-----Sparse_Attention-----#########
        I, Gh, Gw = self.interval, Hd // self.interval, Wd // self.interval
        shifted_sparse_x = shifted_x.reshape(b, Gh, I, Gw, I, c).permute(0, 2, 4, 1, 3, 5).contiguous()
        shifted_sparse_x = shifted_sparse_x.reshape(b * I * I, Gh, Gw, c)
        nP = I ** 2  # number of partitioning groups
        # attn_mask_sp
        if pad_r > 0 or pad_b > 0:
            mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
            mask = mask.reshape(nP, 1, Gh * Gw)
            attn_mask_sp = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x.device)
            attn_mask_sp = attn_mask_sp.masked_fill(mask < 0, NEG_INF)
        else:
            attn_mask_sp = None
        # #######-----Sparse_Attention-----#########

        # ######----Partition_Windows----########
        if not self.triangular_flag:
            # _, h_s, w_s, _ = shifted_sparse_x.shape  #Here: h_s=Gh and w_s=Gw
            x_windows = window_partition(shifted_sparse_x,
                                         self.window_size)  # nw*b, window_size, window_size, c
            x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                       c)  # nw*b, window_size*window_size, c  #[16, 16*16, 180]
        else:
            # _, h_s, w_s, _ = shifted_sparse_x.shape  #Here: h_s=Gh and w_s=Gw
            assert Gh >= (2 * self.window_size) and Gw >= (2 * self.window_size), "input feature has wrong size"
            x_windows = window_partition_triangular(shifted_sparse_x, 2 * self.window_size,
                                                    triangular_masks)  # nw*b, window_size, window_size, c
            _, m, n, _ = x_windows.shape  # [b*(h_ws)*(w_ws), m, n, c]
            x_windows = x_windows.view(-1, n, c)  # [b*(h_ws)*(w_ws)*m, n, c]
        # ######----Partition_Windows----########

        # ######----W-MSA/SW-MSA----########
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask, sp_mask=attn_mask_sp)
        # ######----W-MSA/SW-MSA----########

        # ######----Merge_Windows----########
        if self.triangular_flag:
            attn_windows = attn_windows.view(-1, m, n, c)  # [b*(h_ws)*(w_ws), m, n, c]
            shifted_sparse_x = window_reverse_triangular(attn_windows, 2 * self.window_size,
                                                         triangular_masks)  # nw*b, window_size, window_size, c
            shifted_sparse_x = shifted_sparse_x.view(-1, Gh // (2 * self.window_size), Gw // (2 * self.window_size), c,
                                                     2 * self.window_size, 2 * self.window_size)
            shifted_sparse_x = shifted_sparse_x.permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, Gh, Gw,
                                                                                            c)
        else:
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_sparse_x = window_reverse(attn_windows, self.window_size, Gh,
                                              Gw)  # b h' w' c
        # ######----Merge_Windows----########

        # ######----Reverse_Sparse----########
        shifted_sparse_x = shifted_sparse_x.reshape(b, I, I, Gh, Gw, c).permute(0, 3, 1, 4, 2,
                                                                                5).contiguous()  # b, Gh, I, Gw, I, c
        shifted_x = shifted_sparse_x.reshape(b, Hd, Wd, c)
        # ######----Reverse_Sparse----########

        # ######----Reverse_Cyclic_Shift----########
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        # ######----Reverse_Cyclic_Shift----########

        # ######----Remove_Padding----########
        if pad_r > 0 or pad_b > 0:
            attn_x = attn_x[:, :h, :w, :].contiguous()
        attn_x = attn_x.view(b, h * w, c)
        # ######----Remove_Padding----########

        # #######################--------End_(S)W-MSA--------########################

        # ###--------####
        x = shortcut + self.drop_path(self.norm1(attn_x))
        # ###--------####

        # ###----LN+MLP----####
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        # ###----LN+MLP----####

        return x

# ##################---------SAB---------###################


# ##################---------Atten_Blocks---------###################
class AttenBlocks(nn.Module):
    """ A series of attention blocks.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 sparse_flag,
                 depth,
                 num_heads,
                 window_size,
                 shift_size,
                 interval,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.sparse_flag = sparse_flag
        self.depth = depth

        if not sparse_flag:
            self.blocks = nn.ModuleList([DAB(dim=dim,
                                             input_resolution=input_resolution,
                                             num_heads=num_heads,
                                             window_size=window_size,
                                             shift_size=shift_size[i],
                                             interval=interval,
                                             triangular_flag=0 if (i % 2 == 0) else 1,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop,
                                             attn_drop=attn_drop,
                                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                             norm_layer=norm_layer) for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([SAB(dim=dim,
                                             input_resolution=input_resolution,
                                             num_heads=num_heads,
                                             window_size=window_size,
                                             shift_size=shift_size[i],
                                             interval=interval,
                                             triangular_flag=0 if (i % 2 == 0) else 1,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop,
                                             attn_drop=attn_drop,
                                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                             norm_layer=norm_layer) for i in range(depth)])

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'], params['triangular_masks'])
        return x


# ##################---------Atten_Blocks---------###################


# ##################---------DC---------###################
class DC(nn.Module):
    def __init__(self):
        super(DC, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)

    def forward(self, rec, x_under, mask):
        rec = torch.fft.fftn(rec)
        rec = torch.fft.fftshift(rec)

        k_under = torch.fft.fftn(x_under)
        k_under = torch.fft.fftshift(k_under)

        result = mask * (rec * self.w / (1 + self.w) + k_under * 1 / (self.w + 1)) + 0.0
        result = result + (1 - mask) * rec  # non-sampling point
        result = torch.fft.ifftshift(result)
        result = torch.fft.ifftn(result)
        result = torch.abs(result)
        return result


# ##################---------DC---------###################


# @ARCH_REGISTRY.register()
# ##################---------DSAB---------###################
class DSAB(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        img_size: Input image size.
        patch_size: Patch size.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 sparse_flag,
                 depth,
                 num_heads,
                 window_size,
                 shift_size,  # tuple
                 interval,  # tuple added
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 img_size=224,
                 patch_size=4):
        super(DSAB, self).__init__()

        self.window_size = window_size

        self.residual_group = AttenBlocks(dim=dim,
                                          input_resolution=input_resolution,
                                          sparse_flag=sparse_flag,
                                          depth=depth,
                                          num_heads=num_heads,
                                          window_size=window_size,
                                          shift_size=shift_size,
                                          interval=interval,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          drop=drop,
                                          attn_drop=attn_drop,
                                          drop_path=drop_path,
                                          norm_layer=norm_layer)

        self.dc = DC()

        self.conv = nn.Conv2d(dim, 1, 3, 1, 1)
        self.conv_back = nn.Conv2d(1, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
                                      norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
                                          norm_layer=None)

        # relative position index
        # #####----Relative_Position_Index----######
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        # #####----Relative_Position_Index----######

    # #####----Relative_Position_Index----######
    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    # #####----Relative_Position_Index----######

    # #####----Attention_Mask(SW-MSA)----######
    def calculate_mask(self, x_size, shift_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    # #####----Attention_Mask----######

    # ##############------Triangular_Window_Mask------###############
    def triangle_masks(self, x):
        ws = 2 * self.window_size
        rows = torch.arange(ws).unsqueeze(1).repeat(1, ws)
        cols = torch.arange(ws).unsqueeze(0).repeat(ws, 1)

        upper_triangle_mask = (cols > rows) & (rows + cols < ws)
        right_triangle_mask = (cols >= rows) & (rows + cols >= ws)
        bottom_triangle_mask = (cols < rows) & (rows + cols >= ws - 1)
        left_triangle_mask = (cols <= rows) & (rows + cols < ws - 1)

        return [upper_triangle_mask.to(x.device), right_triangle_mask.to(x.device), bottom_triangle_mask.to(x.device),
                left_triangle_mask.to(x.device)]

    # ##############------Triangular_Window_Mask------###############

    def forward(self, x, x_size, x_copy, mask):
        # Calculate attention mask and relative position index in advance to speed up inference.
        # The original code is very time-cosuming for large window size.
        attn_mask = tuple([self.calculate_mask(x_size, shift_size).to(x.device) for shift_size in
                           (self.window_size // 2, self.window_size, self.window_size // 2 * 3)])
        triangular_masks = tuple(self.triangle_masks(x))

        params = {'attn_mask': attn_mask, 'triangular_masks': triangular_masks,
                  'rpi_sa': self.relative_position_index_SA}

        x_init = x
        x_trans = self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))
        x_dc = self.dc(x_trans, x_copy, mask)
        x = self.patch_embed(self.conv_back(x_dc)) + x_init

        return x


# ##################---------DSAB---------###################


# ##################---------Patch_Embedding---------###################
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size   (int | tuple): Image size.
        patch_size (int): Patch token size.
        in_chans   (int): Number of input image channels.
        embed_dim  (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=192, patch_size=1, in_chans=1, embed_dim=120, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


# ##################---------Patch_Embedding---------###################


# ##################---------Patch_Unembedding---------###################
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size   (int | tuple): Image size.
        patch_size (int): Patch token size.
        in_chans   (int): Number of input image channels.
        embed_dim  (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=192, patch_size=1, in_chans=1, embed_dim=120, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


# ##################---------Patch_Unembedding---------###################


# @ARCH_REGISTRY.register()
class MHWT(nn.Module):
    r"""
    Args:
        img_size       (int | tuple(int)): Input image size.
        patch_size     (int | tuple(int)): Patch size. Default: 1
        in_chans       (int): Number of input image channels.
        embed_dim      (int): Patch embedding dimension.
        depths         (tuple(int)): Depth of each Swin Transformer layer.
        num_heads      (tuple(int)): Number of attention heads in different layers.
        window_size    (tuple(int)): Window size.
        interval       (tuple(int)): Dilation in Sparse Attention
        mlp_ratio      (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias       (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale       (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate      (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer     (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape            (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm     (bool): If True, add normalization after patch embedding. Default: True
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size,
                 patch_size=1,
                 in_chans=1,
                 out_chans=1,
                 embed_dim=120,
                 depths=(8, 8, 8, 8, 8, 8),
                 num_heads=(4, 4, 4, 4, 4, 4),
                 window_size=(16, 16, 8, 8, 4, 4),
                 interval=(0, 2, 0, 2, 0, 2),
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 resi_connection='1conv'
                 ):
        super(MHWT, self).__init__()

        # Arguments
        self.window_size = window_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        num_in_ch = in_chans
        num_out_ch = out_chans

        # #####----Shallow_Feature_Extraction----######
        self.conv_fusion = nn.Conv2d(num_in_ch, embed_dim // 2, 3, 1, 1)
        self.conv_fusion_complement = nn.Conv2d(num_in_ch, embed_dim // 2, 3, 1, 1)
        self.conv_first = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        # #####----Shallow_Feature_Extraction----######

        # #####----Patch_Embedding----######
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=embed_dim,
                                      embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # #####----Patch_Embedding----######

        # #####----Patch_Unembedding----######
        self.patch_unembed = PatchUnEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=embed_dim,
                                          embed_dim=embed_dim,
                                          norm_layer=norm_layer if self.patch_norm else None)
        # #####----Patch_Unembedding----######

        # #####----Absolute_Position_Embedding----######
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # #####----Absolute_Position_Embedding----######

        # #####----DropOut_with_stochastic----######
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # #####----DropOut_with_stochastic----######

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            window = window_size[i_layer]
            shift = (0, 0, window // 2, window // 2, window, window, window // 2 * 3, window // 2 * 3)
            layer = DSAB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         sparse_flag=0 if (i_layer % 2 == 0) else 1,
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size[i_layer],
                         shift_size=shift,  # tuple added
                         interval=interval[i_layer],  # tuple added
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         drop=drop_rate,
                         attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         img_size=img_size,
                         patch_size=patch_size)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # #####----Last_Convolution+Reconstruction----######
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        # #####----Last_Convolution+Reconstruction----######

        self.apply(self._init_weights)

    # #####----Weight_Initialization----######
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # #####----Weight_Initialization----######

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # #########-------forward_features-------##########
    def forward_features(self, x, x_copy, mask):
        x_size = (x.shape[2], x.shape[3])

        # #Embed$$Unembed
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, x_copy, mask)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        return x

    # #########-------forward_features-------##########

    # #########-------forward-------##########
    def forward(self, x, complement, mask):
        x_copy = x
        x_fusion = self.conv_fusion(x)
        c_fusion = self.conv_fusion_complement(complement)
        x_first = torch.cat((x_fusion, c_fusion), dim=1)
        x_first = self.conv_first(x_first)

        x_trans = self.forward_features(x_first, x_copy, mask)

        x_trans = self.conv_after_body(x_trans) + x_first

        x = x_copy + self.conv_last(x_trans)
        return x
    # #########-------forward-------##########


def define_G(opt):
    opt_net = opt['MODEL']

    netG = MHWT(img_size=opt_net['IMG_SIZE'],
                in_chans=opt_net['INPUT_DIM'],  # 1
                embed_dim=opt_net['HEAD_HIDDEN_DIM'],
                depths=opt_net['DEPTHS'],
                num_heads=opt_net['NUM_HEADS'],
                window_size=opt_net['WINDOW_SIZE'],
                interval=opt_net['INTERVAL'],
                mlp_ratio=opt_net['MLP_RATIO'],  # 2
                resi_connection=opt_net['RESI_CONNECTION'])  # '1conv'
    return netG


if __name__ == '__main__':
    x = torch.rand(1, 1, 192, 192)
    y = torch.rand(1, 1, 192, 192)
    z = torch.rand(1, 1, 192, 192)
    model = MHWT(img_size=(192, 192),
                 in_chans=1,
                 embed_dim=120,
                 depths=(8, 8, 8, 8, 8, 8),
                 num_heads=(4, 4, 4, 4, 4, 4),
                 window_size=(16, 16, 8, 8, 4, 4),
                 interval=(0, 2, 0, 2, 0, 2),
                 mlp_ratio=2,
                 resi_connection='1conv')
    flops, params = profile(model, inputs=(x, y, z))
    # params = sum(param.nelement() for param in model.parameters())
    print(flops / 1e6)
    print(params / 1e6)
    out = model(x)
    print(out.shape)
