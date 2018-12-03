import os
class Config( object ):

	def __init__( self ):

		## Autoencoder configuration ##
		self.ae_in_w = 64
		self.ae_in_h = 64
		self.ae_in_c = 1
		# self.ae_in_l = 16
		self.ae_seq_l = 3

		self.ae_loss = "mmd"

		## Transformer Configuration ##
		self.trans_in_w = 16
		self.trans_in_h = 16
		self.trans_in_c = self.ae_in_c
		self.trans_in_l = 16 # max length of a given sequence
		self.trans_seq_l = self.ae_seq_l # length of each unit being input at each state

		self.trans_conv_ndim = 3
		self.trans_output_channel = 3 # 3 pictures as a group to be input to advance a state
		self.trans_kernel_shape = [ 3,3,3 ]


		### Training ###
		self.batch_size = 16 # batch size to be used to sample a batch of video frame for training
		self.frame_len = 16 # number of frames for each video clip to be passed to training

		### Data Preprocessing ###
		self.data_path = "../data/mnist_test_seq.npy"
		self.train_test_split = 0.75 # percentage of training samples within the entire dataset
		self.train_iter = 600000
		self.ckpt_dir = "mmd_ae"
		self.res_dir = "mmd_res"
		self.save_every = 300
		self.load_model = ""

	def init( self ):
		os.mkdir( self.ckpt_dir ) if not os.path.exists( self.ckpt_dir ) else print()
		os.mkdir( self.res_dir ) if not os.path.exists( self.res_dir ) else print()



