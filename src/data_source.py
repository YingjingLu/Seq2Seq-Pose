import numpy as np 
import gc
gc.enable()

class Data_Source( object ):

	def __init__( self, config ):
		
		self.config = config
		self.init()


	def init( self ):

		self.data = np.load( self.config.data_path ).reshape( -1, -1, -1, -1, 1 )
		np.random.shuffle( self.data )

		self.num_data = self.data.shape[  1 ]
		self.num_train = int( self.num_data * self.config.train_test_split )
		self.train = self.data[ :, :self.num_train, :, :, : ]
		self.test = self.data[ :, self.num_train:, :, :, : ]
		self.num_test = self.test.shape[ 1 ]

		self.cur_train_index = 0
		self.cur_test_index = 0

		self.seq_len = self.train.shape[ 0 ]
		self.cur_seq_index = 0

		del self.data
		gc.collect()

	def get_train_batch_ae( self, batch_size = -1, frame_len = -1 ):

		if batch_size == -1:
			batch_size = self.config.batch_size 
		if frame_len == -1:
			frame_len = self.config.frame_len
		
		if self.cur_train_index + batch_size >= self.num_train:
			np.random.shuffle( self.train )
			self.cur_seq_index += 1
			if self.cur_seq_index + self.config.ae_seq_l >= self.seq_len:
				self.cur_seq_index = 0
		batch = self.train[ self.cur_seq_index: self.cur_seq_index + self.config.ae_seq_l,
		                    self.cur_train_index: self.cur_train_index + batch_size,
							:,
							:,
							: ]		
		self.cur_train_index += batch_size
		return batch
		
		

		
		


		

		

		


