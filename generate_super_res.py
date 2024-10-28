import os
import torch
from config import *
from glob import glob
from PIL import Image
from generator import Generator
import torchvision.transforms as trf
from train import PATH_G, xavier_init_weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR_DIR = "./results/sr_img_test/"
SR_VAL_DIR = f"./valid_results/"

if not os.path.exists(SR_VAL_DIR):
	os.makedirs(SR_VAL_DIR)

# Function to save the images
def tensor_to_img(tensor, filepath):
	tensor = tensor.cpu()
	pil = trf.ToPILImage()(tensor.squeeze_(0))
	pil.save(filepath)
	print(f"Saved to {filepath}")
	

# Function to generate single Super Resolution Image
def generate_sr(lr_img_path):
	with torch.no_grad():
		pil_img = Image.open(lr_img_path)
		img_tensor = trf.ToTensor()(pil_img)
		img_tensor = torch.unsqueeze(img_tensor, 0) # add batch dimension
		img_tensor = img_tensor.to(device)
		sr_img = GEN(img_tensor)
		print(f"Upscaled from size [{img_tensor.shape[2]}, {img_tensor.shape[3]}] to [{sr_img.shape[2]}, {sr_img.shape[3]}]")

	file_name = lr_img_path.split('/')[-1]
	sr_img_path = os.path.join(SR_DIR, f"sr_{file_name}")
	tensor_to_img(sr_img, sr_img_path)
	

# Function to generate Super res images for the validation set
def gen_sr_valset():
	files = glob(f"DIV2K/DIV2K_valid_LR_bicubic/X{UPSCALE}/*.png")
	for lr_img_path in files:
		with torch.no_grad():
			pil_img = Image.open(lr_img_path)
			img_tensor = trf.ToTensor()(pil_img)
			img_tensor = torch.unsqueeze(img_tensor, 0) # add batch dimension
			img_tensor = img_tensor.to(device)
			sr_img = GEN(img_tensor)

		file_name = lr_img_path.split('/')[-1].replace('\\','_')
		sr_img_path = os.path.join(SR_VAL_DIR, f"sr_{file_name}")
		tensor_to_img(sr_img, sr_img_path)

if __name__ == '__main__':
	# Load checkpoints
	GEN = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE)
	if PATH_G.exists():
		checkpoint_G = torch.load(PATH_G)
		GEN.load_state_dict(checkpoint_G['state_dict'])
		GEN.to(device)
	else:
		print("Checkpoints not found, using Xavier initialization.")
		GEN.apply(xavier_init_weights).to(device)
	GEN.eval()

	gen_sr_valset()