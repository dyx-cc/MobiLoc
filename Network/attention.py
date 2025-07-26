import torch
from torch import nn

from torchinfo import summary


def _normalize_depth_vars(depth_k, depth_v, filters):
    """
    Accepts depth_k and depth_v as either floats or integers
    and normalizes them to integers.

    Args:
        depth_k: float or int.
        depth_v: float or int.
        filters: number of output filters.

    Returns:
        depth_k, depth_v as integers.
    """

    if type(depth_k) == float:
        depth_k = int(filters * depth_k)
    else:
        depth_k = int(depth_k)

    if type(depth_v) == float:
        depth_v = int(filters * depth_v)
    else:
        depth_v = int(depth_v)

    return depth_k, depth_v


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1, padding='same', bias=True, kernel_initializer='he_normal'):
        super(Conv2dLayer, self).__init__()
        if padding == 'same':
            self.padding = (kernel_size//2)*stride if stride > 1 else kernel_size // 2
        elif padding == 'valid':
            self.padding = 0

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=bias)

        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,x):
        return self.conv(x)


class AttnAugmentation2d(nn.Module):
    def __init__(self, depth_k, depth_v, num_heads, relative=True):

        """
        Applies attention augmentation on a convolutional layer output.

        Args:
            depth_k: float or int. Number of filters for k.
                Computes the number of filters for `v`.
                If passed as float, computed as `filters * depth_k`.
            depth_v: float or int. Number of filters for v.
                Computes the number of filters for `k`.
                If passed as float, computed as `filters * depth_v`.
            num_heads: int. Number of attention heads.
                Must be set such that `depth_k // num_heads` is > 0.
            relative: bool, whether to use relative encodings.

        Raises:
            ValueError: if depth_v or depth_k is not divisible by num_heads.

        Returns:
            Output tensor of shape
            -   [Batch, Height, Width, Depth_V] if channels_last data format.
            -   [Batch, Depth_V, Height, Width] if channels_first data format.
        """

        super(AttnAugmentation2d, self).__init__()
        if depth_k % num_heads != 0:
            raise ValueError('`depth_k` (%d) is not divisible by `num_heads` (%d)' % (
                depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError('`depth_v` (%d) is not divisible by `num_heads` (%d)' % (
                depth_v, num_heads))

        if depth_k // num_heads < 1.:
            raise ValueError('depth_k / num_heads cannot be less than 1 ! '
                             'Given depth_k = %d, num_heads = %d' % (
                             depth_k, num_heads))

        if depth_v // num_heads < 1.:
            raise ValueError('depth_v / num_heads cannot be less than 1 ! '
                             'Given depth_v = %d, num_heads = %d' % (
                                 depth_v, num_heads))


        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative = relative

        self.initialized = False

    def forward(self,x):
        device = x.device
        in_shape = x.size()
        batch, channels, height, width = in_shape
        if not self.initialized:
            self.depth_k, self.depth_v = _normalize_depth_vars(self.depth_k, self.depth_v, in_shape)
            if self.relative:
                dk_per_head = self.depth_k // self.num_heads
                if dk_per_head == 0:
                    print('dk per head', dk_per_head)

                self.key_relative_w = nn.Parameter(torch.randn(2 * width - 1, dk_per_head)).to(device)
                nn.init.normal_(self.key_relative_w, std=dk_per_head ** -0.5)
                self.key_relative_h = nn.Parameter(torch.randn(2 * height - 1, dk_per_head)).to(device)
                nn.init.normal_(self.key_relative_h, std=dk_per_head ** -0.5)
            else:
                self.key_relative_w = None
                self.key_relative_h = None

            self.initialized = True

        x = x.permute(0,2,3,1)

        q, k, v = torch.split(x, [self.depth_k, self.depth_k, self.depth_v], dim=-1)
        q = self.split_heads_2d(q)
        k = self.split_heads_2d(k)
        v = self.split_heads_2d(v)

        depth_k_heads = self.depth_k / self.num_heads
        q = q * (depth_k_heads ** -0.5)

        qk_shape = (batch, self.num_heads, height*width, self.depth_k // self.num_heads)
        v_shape = (batch, self.num_heads, height * width, self.depth_v // self.num_heads)

        # Reshape for attention computation
        flat_q = q.reshape(qk_shape)
        flat_k = k.reshape(qk_shape)
        flat_v = v.reshape(v_shape)

        # Dot product attention -> [Batch, Num_heads, HW, HW]
        logits = torch.matmul(flat_q, flat_k.transpose(-2,-1))

        # Apply relative encodings
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = nn.functional.softmax(logits, dim=-1)
        attn_out = torch.matmul(weights, flat_v)

        # Reshape attention output
        attn_out_shape = (batch, self.num_heads, height, width, self.depth_v // self.num_heads)
        attn_out = attn_out.view(*attn_out_shape)
        attn_out = self.combine_heads_2d(attn_out)

        attn_out = attn_out.permute(0,3,1,2)

        return attn_out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.depth_v
        return tuple(output_shape)

    def split_heads_2d(self, ip):
        tensor_shape = list(ip.size())

        batch = tensor_shape[0]
        height = tensor_shape[1]
        width = tensor_shape[2]
        channels = tensor_shape[3]

        # Save the spatial tensor dimensions
        self._batch = batch
        self._height = height
        self._width = width

        ret_shape = (batch, height, width,  self.num_heads, channels // self.num_heads)
        split = ip.view(ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = split.permute(transpose_axes)

        return split

    def relative_logits(self, q):
        shape = list(q.size())

        height = shape[2]
        width = shape[3]

        rel_logits_w = self.relative_logits_1d(q, self.key_relative_w, height, width,
                                               transpose_mask=[0, 1, 2, 4, 3, 5])

        rel_logits_h = self.relative_logits_1d(
            q.permute([0, 1, 3, 2, 4]),
            self.key_relative_h, width, height,
            transpose_mask=[0, 1, 4, 2, 5, 3])

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = rel_logits.view(-1, self.num_heads * H, W, 2 * W - 1)
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = rel_logits.view(-1, self.num_heads, H, W, W)
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, H, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        rel_logits = rel_logits.reshape(-1, self.num_heads, H * W, H * W)
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L = x.shape[:3]
        col_pad = torch.zeros(B, Nh, L, 1, device=x.device, dtype=x.dtype)
        x = torch.cat([x, col_pad], dim=3)
        flat_x = x.view(B, Nh, L * 2 * L)
        flat_pad = torch.zeros(B, Nh, L - 1, device=x.device, dtype=x.dtype)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = flat_x_padded.view(B, Nh, L + 1, 2 * L - 1)
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def combine_heads_2d(self, inputs):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = torch.permute(inputs, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = list(transposed.shape)
        a, b = shape[-2:]
        ret_shape = shape[:-2] + [a * b]
        # [batch, height, width, depth_v]
        reshaped = transposed.reshape(ret_shape)
        return reshaped

    def get_config(self):
        config = {
            'depth_k': self.depth_k,
            'depth_v': self.depth_v,
            'num_heads': self.num_heads,
            'relative': self.relative,
        }
        base_config = super(AttnAugmentation2d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Augmented_Conv2d(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, strides=1, depth_k=0.2, depth_v=0.2, num_heads=8, relative_encodings=True):
        """
        Builds an Attention Augmented Convolution block.

        Args:
            input: keras tensor.
            filters: number of output filters.
            kernel_size: convolution kernel size.
            strides: strides of the convolution.
            depth_k: float or int. Number of filters for k.
                Computes the number of filters for `v`.
                If passed as float, computed as `filters * depth_k`.
            depth_v: float or int. Number of filters for v.
                Computes the number of filters for `k`.
                If passed as float, computed as `filters * depth_v`.
            num_heads: int. Number of attention heads.
                Must be set such that `depth_k // num_heads` is > 0.
            relative_encodings: bool. Whether to use relative
                encodings or not.

        Returns:
            a keras tensor.
        """

        super(Augmented_Conv2d, self).__init__()
        depth_k, depth_v = _normalize_depth_vars(depth_k, depth_v, filters)
        self.conv_out = Conv2dLayer(in_channels, filters-depth_v, kernel_size=kernel_size, stride=strides)

        self.qkv_conv = Conv2dLayer(in_channels, 2*depth_k+depth_v, kernel_size=1, stride=strides)
        self.attn_out1 = AttnAugmentation2d(depth_k, depth_v, num_heads, relative_encodings)
        self.attn_out2 = Conv2dLayer(depth_v, depth_v, kernel_size=1)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self,x):
        y = self.conv_out(x)
        y1 = self.qkv_conv(x)
        y1 = self.attn_out1(y1)
        y1 = self.attn_out2(y1)

        y = torch.concat([y,y1],dim=1)
        y = self.bn(y)
        return y


if __name__ == '__main__':
    input = torch.rand(1,64,3,3)
    model = Augmented_Conv2d(64, filters=64, kernel_size=3, depth_k=4, depth_v=4, num_heads=1, relative_encodings=True)

    print(model)
    summary(model, input_size=(1, 64, 3, 3))

    y = model(input)

    print("Attention Augmented Conv out shape : ", y.shape)