import os
import numpy as np
import shutil
from tqdm import tqdm


def sample_dataset():
	sample = 200000


	data_list = os.listdir("Data/Imagenet64")


	if not os.path.exists("FID_Images/Images"):
		os.mkdir("FID_Images/Images")

	print(len(data_list))


	choices = np.random.choice(len(data_list), size=sample)
	print(choices.shape)
	count = 0

	for index in tqdm(choices):
		image = data_list[index]


		shutil.copyfile(src="Data/Imagenet64/{}".format(image), dst="FID_Images/Images/{}".format(image))
		count += 1

	print("{} image copied".format(count))

def main():
	sample()

if __name__ == '__main__':
	main()