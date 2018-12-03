import tensorflow as tf 
from dc_gan_util import conv3d, deconv3d, batch_norm, conv2d, deconv2d, conv_out_size_same

def img_encoder3d( inputs, start_filter_num = 16 ):
    with tf.variable_scope( "img_encoder3d", reuse = tf.AUTO_REUSE ) as scope:
        conv = tf.nn.relu( conv3d( inputs, start_filter_num, name = "conv1a" ) )
        conv = tf.nn.max_pool3d( conv, [ 1, 1, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], name = "pool1" )

        conv = tf.nn.relu( conv3d( conv, start_filter_num * 2, name = "conv2a" ) )
        conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], name = "pool2" )

        conv = tf.nn.relu( conv3d( conv, start_filter_num * 4, name = "conv3a" ) )
        conv = tf.nn.relu( conv3d( conv, start_filter_num * 4, name = "conv3b" ) )
        conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], name = "pool3" )

        conv = tf.nn.relu( conv3d( conv, start_filter_num * 8, name = "conv4a" ) )
        conv = tf.nn.relu( conv3d( conv, start_filter_num * 8, name = "conv4b" ) )
        conv = tf.nn.max_pool3d( conv, [ 1, 2, 2, 2, 1 ], [ 1, 1, 1, 1, 1 ], name = "pool4" )

        return conv

def img_decoder3d( inputs, out_depth, out_h, out_w, out_c, 
                     start_filter_num = 16 ):
    with tf.variable_scope( 'img_decoder3d', reuse = tf.AUTO_REUSE ) as scope:
        sd, sh, sw = out_depth, out_h, out_w
        sd2, sh2, sw2 = conv_out_size_same( sd, 2 ), conv_out_size_same( sh, 2 ),conv_out_size_same( sw, 2 )
        sd4, sh4, sw4 = conv_out_size_same( sd2, 2 ), conv_out_size_same( sh2, 2 ),conv_out_size_same( sw2, 2 )
        sd8, sh8, sw8 = conv_out_size_same( sd4, 2 ), conv_out_size_same( sh4, 2 ),conv_out_size_same( sw4, 2 )

        deconv = tf.nn.relu( deconv3d( inputs, [ -1, sd8, sh8, sw8, start_filter_num * 8 ], name = "deconv1" ) )
        deconv = tf.nn.relu( deconv3d( deconv, [ -1, sd4, sh4, sw4, start_filter_num * 4 ], name = "deconv2" ) )
        deconv = tf.nn.relu( deconv3d( deconv, [ -1, sd2, sh2, sw2, start_filter_num * 2 ], name = "deconv3" ) )
        deconv = tf.nn.relu( deconv3d( deconv, [ -1, sd, sh, sw, start_filter_num * 1 ], name = "deconv4" ) )

        return tf.nn.tanh( deconv )

