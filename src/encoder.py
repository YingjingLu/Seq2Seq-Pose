import math
import numpy as np 
import tensorflow as tf 
from dc_gan_util import *


def resnet_18_encoder( inputs, args ):
    u_net_list = []
    with tf.variable_scope( "encoder", reuse = tf.AUTO_REUSE ) as scope:
        l = leaky_relu( conv2d( inputs, 64, k_h = 7, k_w = 7, d_h = 2, d_w = 2, name = "l0_conv" ) )
        ld = tf.layers.max_pooling2d( l, 3, 2, padding = "same", name = "l0_max_pool" )

        """ 64 filters * 2 """
        l = leaky_relu( conv2d( ld, 64,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b1_l0" ) )
        l = conv2d( l, 64,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b1_l1" )

        l = leaky_relu( ld + l )
        ld = leaky_relu( l )

        l = leaky_relu( conv2d( l, 64,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b2_l0" ) )
        l = conv2d( l, 64,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b2_l1" )

        l = leaky_relu( ld + l )
        ld = leaky_relu( l )
        u_net_list.append( ld )

        l = leaky_relu( conv2d( l, 128,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b3_l0" ) )
        l = conv2d( l, 128,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b3_l1" )

        l = leaky_relu( ld + l )
        ld = leaky_relu( l )

        l = leaky_relu( conv2d( l, 128,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b4_l0" ) )
        l = conv2d( l, 128,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b4_l1" )

        l = leaky_relu( ld + l )
        ld = leaky_relu( l )
        u_net_list.append( ld )

        l = leaky_relu( conv2d( l, 256,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b5_l0" ) )
        l = conv2d( l, 256,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b5_l1" )

        l = leaky_relu( ld + l )
        ld = leaky_relu( l )

        l = leaky_relu( conv2d( l, 256,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b6_l0" ) )
        l = conv2d( l, 256,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b6_l1" )

        l = leaky_relu( ld + l )
        ld = leaky_relu( l )
        u_net_list.append( ld )

        l = leaky_relu( conv2d( l, 512,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b7_l0" ) )
        l = conv2d( l, 512,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b7_l1" )

        l = leaky_relu( ld + l )
        ld = leaky_relu( l )

        l = leaky_relu( conv2d( l, 512,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b8_l0" ) )
        l = conv2d( l, 512,  k_h = 3, k_w = 3, d_h = 2, d_w = 2, name = "b8_l1" )

        l = leaky_relu( ld + l )
        ld = leaky_relu( l )
        u_net_list.append( ld )

        print( "Encoder shape", l.get_shape().as_list() )

        l = tf.layers.flatten( l, name = "encode_flatten" )
        l = leaky_relu( tf.layers.dense( l, args.hidden_size, name = "encode_fc" ) )

        return l

