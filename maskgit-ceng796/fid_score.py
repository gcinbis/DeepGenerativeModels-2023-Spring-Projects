from pytorch_fid import fid_score
import argparse
import torch



def calculate(fid)



def main(args):

	if args.device is None:
		args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	paths = [args.target_path, args.out_path]
	batch_size = args.batch_size

	score = fid_score.calculate_fid_given_paths(paths=paths, batch_size=batch_size, device=args.device, dims=2048)	
	print("FID score: {}".format(score))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("out_path", help="Path of the output image folder")
	parser.add_argument("target_path", help="Path of the target image folder")
	parser.add_argument("--device", default=None, help="Device to compute")
	parser.add_argument("--batch_size", default=50)
	args = parser.parse_args()

	main(args)


