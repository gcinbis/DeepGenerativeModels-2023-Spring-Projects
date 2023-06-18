import numpy as np
import torch

def sample_from_dataset(dataset, n=1, mask_x=16, mask_y=16, mask_size=32):
	
	org_temp = []
	mask_temp = []

	mask = torch.zeros(size=(64, 64), dtype=bool)

	mask[mask_x:mask_x+mask_size, mask_y:mask_y+mask_size] = True

	mask = torch.stack((mask, mask, mask))
	print(mask.shape)


	for i in range(n):
		idx = np.random.randint(low=0, high=len(dataset))
		img = dataset[idx]

		masked = img.clone()
		masked[mask] = 0

		org_temp.append(img)
		mask_temp.append(masked)

	originals = torch.stack(org_temp)
	masked = torch.stack(mask_temp)

	return originals, masked



import torch
import numpy as np

def add_predicted_to_originals(originals, masked, out_images, mask_x=16, mask_y=16, mask_size=32):
    # Replace the masked region in originals with the corresponding part from masked_resized

    # Define mask for blending
    n = originals.size(2)  # Assuming originals is of shape [batch_size, channels, height, width]
    y = torch.linspace(-1, 1, n).view(-1, 1).repeat(1, n).to(originals.device)
    x = torch.linspace(-1, 1, n).repeat(n, 1).to(originals.device)
    dist = torch.sqrt(x*x + y*y)  # This creates a 2D grid where value at each point is distance to center

    # Create a gaussian window for blending
    sigma = 0.5  # This value might need adjusting, depending on your needs
    window = torch.exp(-dist / (2*sigma*sigma))

    # The mask indicating the cropped region
    mask = torch.zeros_like(originals)
    mask[:, :, mask_x:mask_x+mask_size, mask_y:mask_y+mask_size] = 1

    # Construct opacity matrix for blending
    mask_blend = torch.where(mask == 1, window, torch.ones_like(mask))

    mask_real = torch.where(mask == 0, mask.type(torch.float), mask_blend)
    mask_fake = torch.where(mask == 0, (1 - mask).type(torch.float), mask_blend)

    # Blending
    blended_image = mask_real * originals + mask_fake * out_images

    # Replace the cropped region
    originals[:, :, mask_x:mask_x+mask_size, mask_y:mask_y+mask_size] = blended_image[:, :, mask_x:mask_x+mask_size, mask_y:mask_y+mask_size]

    return originals





def cosine_scheduler(r):
	return np.cos(r * np.pi / 2)