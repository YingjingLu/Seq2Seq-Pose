from autoencoder import Autoencoder
from config import Config 
from data_source import Data_Source

config = Config()
config.init()
model = Autoencoder( config )
data_source = Data_Source( config )
model.train( data_source )

