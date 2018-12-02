import numpy as np 
import gc
gc.enabled()

class Data_Source( object ):

	def __init__( self, config ):
		
		self.config = config


	def init( self ):

		self.data = np.load( self.config.data_path )
		np.random.shuffle( self.data )

		self.num_data = self.data.shape[  1 ]
		self.num_train = int( self.num_data * self.configg.train_test_split )
		self.train = self.data[ :, :self.num_train, :, : ]
		self.test = self.data[ :, self.num_train:, :, : ]
		self.num_test = test.shape[ 1 ]

		self.cur_train_index = 0
		self.cur_test_index = 0



	def get_batch( self, batch_size = -1, frame_len = -1 ):

		if batch_size == -1:
			batch_size = self.config.batch_size 
		if frame_len == -1:
			frame_len = self.config.frame_len


