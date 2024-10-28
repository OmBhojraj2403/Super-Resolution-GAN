import os
import glob
from config import *
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as trf
from torchvision.transforms import ToTensor

class DIV2K(Dataset):
	def __init__(self, data_dir, transform=ToTensor()):
		# Get all paths of images inside `data_dir` into a list
		pattern = os.path.join(data_dir, "**/*.png")
		self.file_paths = sorted(glob.glob(pattern, recursive=True))
		self.transform = transform

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, index):
		file_name = self.file_paths[index].split('/')[-1]
		img = Image.open(self.file_paths[index])
		img = self.transform(img)
		return img, file_name
	
TRANSFORM_HR = trf.Compose([
	trf.CenterCrop(HR_CROPPED_SIZE),
	trf.ToTensor()
])
TRANSFORM_LR = trf.Compose([
	trf.CenterCrop(LR_CROPPED_SIZE),
	trf.ToTensor()
])

def load_training_data():
	data_train_hr = DIV2K(data_dir=os.path.join("./", TRAIN_HR_DIR), transform=TRANSFORM_HR)
	data_train_lr = DIV2K(data_dir=os.path.join("./", TRAIN_LR_DIR), transform=TRANSFORM_LR)
	return data_train_hr, data_train_lr