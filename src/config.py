
class Config( object ):

	def __init__( self ):

		### Training ###
		self.batch_size = 16 # batch size to be used to sample a batch of video frame for training

		self.frame_len = 16 # number of frames for each video clip to be passed to training

		### Data Preprocessing ###
		self.data_path = "../data/mnist_test_seq.npy"
		self.train_test_split = 0.75 # percentage of training samples within the entire dataset



