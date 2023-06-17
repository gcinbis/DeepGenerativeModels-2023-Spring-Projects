import torch

from models.MaskGIT import MaskGIT
#from maskgit_utils.ImageDirectory import ImageDirectory
from maskgit_utils.TinyImageNetDirectory import ImageDirectory
import numpy as np
from torchvision import transforms

from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from maskgit_utils.utils import sample_from_dataset
from PIL import Image


MODELPATH = "SavedModels"
OUTPUT_PATH = "Output"

class MaskGITWrapper(object):
	"""docstring for MaskGITWrapper"""
	def __init__(self, args):
		super(MaskGITWrapper, self).__init__()
		self.args = args
		
		self.model_ready = False

		if self.args.device is None:
			self.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			print("No device is selected! Switching to {}".format(self.args.device))

		self.model = MaskGIT(self.args).to(self.args.device)


	def train(self):

		batch_size = self.args.batch_size
		epochs = self.args.epochs


		transformList = transforms.Compose([
			transforms.ToTensor(),
			])

		transform_to_image = transforms.ToPILImage()

		train_set = ImageDirectory(self.args.dataset_path, transform=transformList)
		train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)




		if self.args.start_epoch != 0:
			print("Resuming training. Starting model: {}".format(self.args.model_path))
			self.load_pretrained_model(self.args.model_path)
		else:
			print("Training from scratch")

		optimizer = torch.optim.Adam(self.model.transformer.parameters())
		losses = []

		self.model_ready = True
		for epoch in range(self.args.start_epoch+1, epochs):
			acc_loss = 0
			counter = 0
			for batch in tqdm(train_loader):
				optimizer.zero_grad()
				batch = batch.to(self.args.device)
				
				logits, labels = self.model(batch)

				loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

				loss.backward()
				
				acc_loss += loss.item() 
				#print(loss.item())
				optimizer.step()
				counter += 1

			acc_loss = acc_loss/(counter+1)
			print(acc_loss)
			losses.append(acc_loss)

			if (epoch+1) % self.args.ckpt_interval == 0:
				np.save("losses.npy", losses)
				torch.save(self.model.state_dict(), "{}/epoch_{}_model.pt".format(MODELPATH, epoch))
				plt.plot(losses)
				plt.xlabel("Epoch")
				plt.ylabel("Loss")
				plt.title("Loss vs. Epoch graph")
				plt.savefig("figures/loss_{}.png".format(epoch))


	def infer_image(self, samples=10, iters=8):

#		if not self.model_ready:
#			print("First, you should load a pretrained model or train a model")
#			return

		self.model.eval()


		transformList = transforms.Compose([
			transforms.ToTensor(),
		 ])
		tensortoImage = transforms.ToPILImage()
		dataset = ImageDirectory(self.args.dataset_path, transform=transformList)

		originals, masked = sample_from_dataset(dataset, samples)

		originals = originals.to(self.args.device)
		masked = masked.to(self.args.device)

		originals_tmp = originals.clone()

		out_images = self.model.inpaint_image(originals_tmp, originals, iterations=iters)


		for i in range(originals.shape[0]):
			org_img = tensortoImage(originals[i])
			masked_img = tensortoImage(masked[i])
			out_img = tensortoImage(out_images[i])

			fig, ax = plt.subplots(1, 3)


			ax[0].imshow(org_img)
			ax[0].set_title("Original Image")
			ax[1].imshow(masked_img)
			ax[1].set_title("Masked Image")
			ax[2].imshow(out_img)
			ax[2].set_title("Output of the Masked Image")

			plt.show()

	def save_images(self, samples=100000, iters=8):
		batch_size=100
		self.model.eval()

		transformList = transforms.Compose([
			transforms.ToTensor(),
		 ])
		tensortoImage = transforms.ToPILImage()
		dataset = ImageDirectory(self.args.dataset_path, transform=transformList)

		image_count = 0

		train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
		print(len(train_loader))
		train_loader_iter = iter(train_loader)

		with tqdm(total=samples) as pbar:

			for i in range(samples // batch_size):
				batch = next(train_loader_iter).to(self.args.device)
				out_images = self.model.inpaint_image(batch, batch, iterations=iters)
				pbar.update(batch_size)


				for out_img in out_images:
					out_img = tensortoImage(out_img).convert("RGB")
					out_img.save("{}/{}.png".format(OUTPUT_PATH, image_count))
					image_count += 1



	def load_pretrained_model(self, model_path):

		self.model.load_state_dict(torch.load(model_path, map_location=self.args.device))
		self.model.eval()
		self.model_ready = True
		print("Model loaded!")
