import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.nn as nn 

class LSTM_Transformer( object ):

    def __init__( self, config, initial_state ):
        self.config = config
        with tf.variable_scope( "trans" ) as scope:
            self.inputs = tf.placeholder( tf.float32, 
                                        [ self.config.trans_in_l, None, self.config.trans_seq_l, 
                                          self.config.trans_in_w, self.config.trans_in_h ], 
                                        name = "trans_inputs"
            )
            self.targets = tf.placeholder( tf.float32,
                                        [ self.config.trans_in_l, None, self.config.trans_seq_l, 
                                          self.config.trans_in_w, self.config.trans_in_h ],
                                        name = "trans_target"  
            )
            self.initial_state_input = tf.placeholder( 
                                        tf.float32,
                                        [ None, self.config.trans_seq_l, self.config.trans_in_w, 
                                          self.config.trans_in_h ]
            )
            self.rnn = rnn.ConvLSTMCell( 
                                        self.config.trans_conv_ndims,
                                        [ self.config.trans_in_w, self.config.trans_in_h, self.config.trans_in_l ],
                                        self.config.trans_output_channel,
                                        self.config.trans_kernel_shape,
                                        name = "trans_conv_lstm" 
            )
            self.trans_output, self.trans_output_state = tf.nn.dynamic_rnn( self.rnn, 
                                                                            inputs = self.inputs,
                                                                            initial_state = self.initial_state_input
            )

