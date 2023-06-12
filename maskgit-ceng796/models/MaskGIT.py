import math
import torch
import torch.nn as nn
from VQGAN.taming_transformers.vqImporter import vqImporter
from models.models import BidirectionalTransformer
import numpy as np
from maskgit_utils import utils
import random

class MaskGIT(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.args = args
		self.tokenizer = vqImporter(args).eval()
		self.transformer = BidirectionalTransformer(args)

	def forward(self, x):

		with torch.no_grad():
			quants, _, (_, _, min_encoding_indices) = self.tokenizer.encode(x)

		min_encoding_indices = min_encoding_indices.view(quants.shape[0], -1)

		orgs = min_encoding_indices.clone()

		# get the mask
		mask = self.create_mask(min_encoding_indices.shape[1])

		# Use broadcasting to apply the mask to all instances in the batch
		mask = mask.unsqueeze(0) # add a singleton dimension to align the dimensions for broadcasting
		mask = mask.expand(min_encoding_indices.shape[0], -1) # expand the mask to the size of the batch

		# Apply mask, if m = 0, do not change, if m = 1, mask it!
		min_encoding_indices[~mask] = self.args.mask_token


		sos_token = torch.full(size=(quants.shape[0], 1), fill_value=self.args.mask_token+1, dtype=torch.long, device=self.args.device)


		min_encoding_indices = torch.cat((sos_token, min_encoding_indices), dim=1)
		orgs = torch.cat((sos_token, orgs), dim=1)

		logits = self.transformer(min_encoding_indices)

		return logits, orgs


	def create_mask(self, sequence_length):
		r = utils.cosine_scheduler(np.random.uniform())		
		num_tokens_to_mask = math.ceil(r * sequence_length) # get the # of tokens to mask

		mask = torch.zeros(sequence_length, dtype=torch.bool) # Initialize a mask with all False

		# get the indices to be masked
		mask_indices = torch.randperm(sequence_length)[:num_tokens_to_mask]
		
		# set these indices to True in mask tensor
		mask[mask_indices] = True

		return mask


	def get_mask_inference(self, confidence_scores, t, T):
		ratio = (t+1) / T
		num_tokens_to_mask = math.floor( utils.cosine_scheduler(ratio) * confidence_scores.shape[1]) 

		undecided_count = torch.sum(confidence_scores < 1.0, dim=-1)

		num_tokens_to_mask = torch.ones_like(undecided_count, dtype=torch.int) * num_tokens_to_mask


		num_tokens_to_mask = torch.minimum(undecided_count, num_tokens_to_mask)

		# Create the initial mask
		mask = torch.zeros_like(confidence_scores, dtype=torch.bool)

		# Sort the tokens by their confidence scores
#		sorted_indices = torch.argsort(confidence_scores, dim=-1)
#		print("Sorted Confidence indices: ", sorted_indices)
		# Get the indices of the tokens with the lowest confidence scores
#		mask_indices = sorted_indices[:, :num_tokens_to_mask]
#		print(mask_indices)
		# Set these indices to True in the mask tensor
#		mask[:, mask_indices] = True

		for i in range(mask.shape[0]):

			sorted_indices = torch.argsort(confidence_scores[i], dim=-1)
			mask_indices = sorted_indices[:num_tokens_to_mask[i]]


			mask[i, mask_indices] = True


		return mask




	def inpaint_image(self, originals, masked, iterations=8):
		# Get empty canvas for input images
		with torch.no_grad():
			quants, _, (_, _, input_tokens) = self.tokenizer.encode(masked)
#		print("Input token shape: ", input_tokens.shape)
		input_tokens = input_tokens.view(quants.shape[0], -1)
#		print("Quants shape: ", quants.shape)
#		print("Input token shape: ", input_tokens.shape)

#		print(input_tokens)

#		input_tokens = torch.ones(size=input_tokens.shape, dtype=torch.int)*self.args.mask_token

		for i in range(input_tokens.shape[0]):
			input_tokens[i][5] = 1024
#			input_tokens[i][6] = 1024
#			input_tokens[i][9] = 1024
#			input_tokens[i][10] = 1024

			for j in range(input_tokens.shape[1]):
				if j < input_tokens.shape[1] // 2:
					continue
#				input_tokens[i][j] = 1024


		# Generate SOS tokens and concatenate with the encoded images
		sos_tokens = torch.full(size=(quants.shape[0], 1), fill_value=self.args.mask_token+1, dtype=torch.long).to(input_tokens.device)
		input_tokens = torch.cat([sos_tokens, input_tokens], dim=1)

#		temp = input_tokens.clone()

		for t in range(iterations):

			# Get logits from the transformer
			logits = self.transformer(input_tokens)


			# Sample ids with their probability score
			sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()
			sampled_ids[sampled_ids > 1023] = random.randint(0,1024)

			# Get the number of unknown pixels

			sampled_ids = torch.where(input_tokens == self.args.mask_token, sampled_ids, input_tokens)

			# Calculate probabilities
			probs = nn.Softmax(dim=-1)(logits)

			confidence_scores = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)
			confidence_scores = torch.where(input_tokens == self.args.mask_token, confidence_scores, 1.0)

			# Get the mask using dynamic masking strategy
			mask = self.get_mask_inference(confidence_scores, t, iterations)


			input_tokens = torch.where(mask, self.args.mask_token, sampled_ids)

		# Return the filled canvas


#		print("Check dis: ", torch.eq(input_tokens, temp))
		vectors = self.tokenizer.quantize.embedding(input_tokens[:, 1:]).reshape(input_tokens.shape[0], 4, 4, 256)

		vectors = vectors.permute(0, 3, 1, 2)

		out_images = None

		with torch.no_grad():
			out_images = self.tokenizer.decode(vectors)


#		out_images = utils.add_predicted_to_originals(originals, masked, out_images)

		return out_images