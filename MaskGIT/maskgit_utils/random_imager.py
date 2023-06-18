from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
from mpl_toolkits.axes_grid1 import ImageGrid


image_names = os.listdir("../Output/Images/")

choices = np.random.choice(len(image_names), size=25)


fig = plt.figure(figsize=(4., 4.))

grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 share_all=True
                 )

grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])

for ax, index in zip(grid, choices):

	name = image_names[index]

	im = Image.open("../Output/Images/{}".format(name))
	ax.imshow(im)

plt.axis("off")
plt.savefig("../figures/outputs.png")
plt.show()

