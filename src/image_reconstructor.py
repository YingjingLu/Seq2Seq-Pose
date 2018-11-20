import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image 


# [None, width, height, 3]
def array_to_single_rgb(image_arrays, width, height, directory, file_name_start):
	assert(len(image_arrays.shape) == 4, "Must be of shape len 4")
	assert(image_arrays.shape[-1] == 3, "Must be RGB image")
	num_image = image_arrays.shape[0]
	for i in range(num_image):
		arr = image_arrays[i,:,:,:]
		img = Image.fromarray(arr, 'RGB')
		file_name = directory + '\\' + file_name_start + "_"+str(i)+'.png'
		img.save(file_name)

	print("Finish Image saving....")
		
def mnist_pair(samples, width, height, directory, file_name_start):
	for i, sample in enumerate(samples):
		fig = plt.figure(figsize=(10, 10))
		plt.imshow(sample.reshape(28, 56), cmap='Greys_r')
		file_name = directory + '\\' + file_name_start + "_"+str(i)+'.png'
		plt.savefig(file_name)
		plt.close(fig)
	print("Finish Image saving....")


def unnormalize(img, cdim):
    img_out = np.zeros_like(img)
    for i in range(cdim):
        # img_out[:, :, i] = 255.* ((img[:, :, i] + 1.) / 2.0)
        img_out[:, :, i] = 255.* (np.clip(img[:, :, i], 0, 1))
    img_out = img_out.astype(np.uint8)
    return img_out

def display_mnist(img, file_start):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5,5)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        k = unnormalize(img[i,:], 3)
        plt.imshow(k)
    plt.savefig('img_out\\'+ file_start +'.png', bbox_inches='tight')
    plt.close(fig)

def display_single_mnist(img, file_start):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5,5)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    plt.savefig('img_out\\'+ file_start +'.png', bbox_inches='tight')
    plt.close(fig)
