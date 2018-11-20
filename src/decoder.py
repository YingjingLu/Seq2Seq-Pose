import math
import numpy as np 
import tensorflow as tf 
from dc_gan_util import *

def resnet_18_decoder( inputs, u_net_input, output_width, output_height ):
    with tf.variable_scope( 'decoder', reuse = tf.AUTO_REUSE ) as scope:

        s_w, s_h =  output_width, output_height
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_16, 2)