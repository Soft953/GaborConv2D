import cv2
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec


def _gabor_filter(shape, sigma=1.0, theta=20, lambd=15.0, gamma=0.5):
    """Return a gabor filter."""
    params = {
        'ksize': shape,
        'sigma': sigma,
        'theta': theta,
        'lambd': lambd,
        'gamma': gamma
    }
    gabor_filter = cv2.getGaborKernel(**params)
    return gabor_filter


class GaborConv2D(Conv2D):
    """Class GaborConv2D

       Custom Conv2D with constant gabor filter included.
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(GaborConv2D, self).__init__(filters, kernel_size, **kwargs)
        print(self.kernel_size)
        if np.size(kernel_size) == 1:
            self.kernelB_init_weight = _gabor_filter(shape=(kernel_size, kernel_size))
        else:
            self.kernelB_init_weight = _gabor_filter(kernel_size)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernelA = self.add_weight(
            name='kernelA',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        self.kernelB = K.constant(self.kernelB_init_weight)
        self.kernel = K.transpose(K.dot(K.transpose(self.kernelA), self.kernelB))

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        self._build_conv_op_input_shape = input_shape
        self._build_input_channel = input_channel
        self._padding_op = self._get_padding_op()
        self._conv_op_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self._padding_op,
            data_format=self._conv_op_data_format)
        self.built = True
