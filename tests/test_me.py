import torch
from fft_conv_pytorch import fft_conv, FFTConv2d

# Create dummy data.  
#     Data shape: (batch, channels, length)
#     Kernel shape: (out_channels, in_channels, kernel_size)
#     Bias shape: (out channels, )
# For ordinary 1D convolution, simply set batch=1.
# signal = torch.randn(3, 3, 3, 3)
# kernel = torch.randn(2, 3, 2, 2)
# bias = torch.randn(2)
signal = torch.tensor([[[[0, 1, 2],
                       [3, 4, 5],
                       [6, 7, 8]]]], dtype=float)
kernel = torch.tensor([[[[0, 1],
                       [2, 3]]]], dtype=float)
bias = torch.zeros(1)

# Functional execution.  (Easiest for generic use cases.)
out = fft_conv(signal, kernel, bias=bias)


# Object-oriented execution.  (Requires some extra work, since the 
# defined classes were designed for use in neural networks.)
fft_conv = FFTConv2d(1, 1, 2, bias=True)#3,2,2
fft_conv.weight = torch.nn.Parameter(kernel)
fft_conv.bias = torch.nn.Parameter(bias)
# out = fft_conv(signal)
print('out =\n', out)