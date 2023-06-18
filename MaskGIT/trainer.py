import argparse
from MaskGITWrapper import MaskGITWrapper

def main(args):
	maskGIT = MaskGITWrapper(args)

	if args.mode == "train":
		maskGIT.train()
	elif args.mode == "infer":
		maskGIT.load_pretrained_model(args.model_path)
		maskGIT.infer_image()


	elif args.mode == "generate_images":
		maskGIT.load_pretrained_model(args.model_path)
		maskGIT.save_images()

if __name__ == '__main__':


	parser = argparse.ArgumentParser()


	parser.add_argument("mode", help="One of the following train/infer")
	
	parser.add_argument("--model_path", help="Path of the pretrained model")
	parser.add_argument("--iters", type=int, default=8, help="Number of iterations for inference loop")
	parser.add_argument("--device", type=str, help="Which device to run")

	parser.add_argument("--start_epoch", type=int, default=0, help="The starting epoch if not training from scratch")

	parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training")
	parser.add_argument("--batch_size", type=int, default=256, help="Size of batches for training")
	parser.add_argument("--ckpt_interval", type=int, default=1, help="Model save intervals")
	parser.add_argument("--dim", type=int, default=768)
	parser.add_argument("--hidden_dim", type=int, default=3072)
	parser.add_argument("--n_layers", type=int, default=24)
	parser.add_argument("--num_codebook_vectors", type=int, default=1024)
	parser.add_argument("--num_img_tok", type=int, default=256)
	parser.add_argument("--mask_token", type=int, default=1024)

	parser.add_argument("--dataset_path", default="Data/Imagenet64")

	args = parser.parse_args()

	main(args)
