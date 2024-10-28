import torch
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader

import os
from config import *
from pathlib import Path
from dataloader import *
from generator import Generator
from discriminator import Discriminator
from perceptual_loss import PerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {str(device).upper()}")

try:
	os.mkdir("./models")
except FileExistsError:
	pass

# PATH_G = Path(f'./models/Gen_{BATCH_SIZE}.pt')
PATH_G = Path(f'./models/Gen_100.pt')
PATH_D = Path(f'./models/Disc_{BATCH_SIZE}.pt')

def train(resume_training=True):
	'''
	:param `resume_training`: whether to continue training from previous checkpoint or not.
	If checkpoints cannot be found, train from beginning, regardless of `resume_training`.
	'''
	### Load data
	data_train_hr, data_train_lr = load_training_data()
	hr_train_loader = DataLoader(dataset=data_train_hr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
	lr_train_loader = DataLoader(dataset=data_train_lr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
	assert len(hr_train_loader) == len(lr_train_loader)

	### Load models
	GEN = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE).to(device)
	DISC = Discriminator().to(device)
	optimizer_G = optim.Adam(GEN.parameters(), lr=LR, betas=BETAS)
	optimizer_D = optim.Adam(DISC.parameters(), lr=LR, betas=BETAS)
	exp_lr_scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=40, gamma=0.1)
	exp_lr_scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=40, gamma=0.1)

	if resume_training and PATH_G.exists() and PATH_D.exists():
		GEN, DISC, optimizer_G, optimizer_D, prev_epochs = load_checkpoints(GEN, DISC, optimizer_G, optimizer_D)
		print("Continue training from previous checkpoints ...")
		warmup = False
	else:
		GEN.apply(xavier_init_weights)
		DISC.apply(xavier_init_weights)
		prev_epochs = 0
		summary(GEN)
		summary(DISC)
		print("Training from start ...")
		warmup = True

	### Train
	GEN.train()
	DISC.train()

	criterion_G = PerceptualLoss(vgg_coef=VGG_LOSS_COEF, adversarial_coef=ADVERSARIAL_LOSS_COEF).to(device)
	warmup_loss = torch.nn.L1Loss()
	criterion_D = torch.nn.BCELoss()

	## Warm up GEN
	if warmup:
		for w in range(WARMUP_EPOCHS):
			print(f"\nWarmup: {w+1}")
			for (batch, hr_batch), lr_batch in zip(enumerate(hr_train_loader), lr_train_loader):
				hr_img, lr_img = hr_batch[0].to(device), lr_batch[0].to(device)
				optimizer_G.zero_grad()

				sr_img = GEN(lr_img)
				err_G = warmup_loss(sr_img, hr_img)
				err_G.backward()

				optimizer_G.step()
				if batch % 10 == 0:
					print(f"\tBatch: {batch + 1}/{len(data_train_hr) // BATCH_SIZE}")
					print(f"\tMAE GEN: {err_G.item():.4f}")

	training_adversarial_loss = []
	training_pixel_loss = []
	training_vgg_loss = []

	for e in range(EPOCHS):
		print(f"\nEpoch: {e+prev_epochs+1}")
		running_pixel_loss = 0.0
		running_adversarial_loss = 0.0
		running_vgg_loss = 0.0

		for (batch, hr_batch), lr_batch in zip(enumerate(hr_train_loader), lr_train_loader):
			# Transfer data to GPU if available
			hr_img, lr_img = hr_batch[0].to(device), lr_batch[0].to(device)

			#### TRAIN DISC: maximize `log(DISC(x)) + log(1-DISC(GEN(z)))`
			optimizer_D.zero_grad()

			# Classify all-real HR images
			real_labels = torch.full(size=(len(hr_img),), fill_value=REAL_VALUE, dtype=torch.float, device=device)
			output_real = DISC(hr_img).view(-1)
			err_D_real = criterion_D(output_real, real_labels)
			err_D_real.backward()

			# Classify all-fake HR images (or SR images)
			fake_labels = torch.full(size=(len(hr_img),), fill_value=FAKE_VALUE, dtype=torch.float, device=device)
			sr_img = GEN(lr_img)
			output_fake = DISC(sr_img.detach()).view(-1)
			err_D_fake = criterion_D(output_fake, fake_labels)
			err_D_fake.backward()

			optimizer_D.step()
			D_Gz1 = output_fake.mean().item()

			#### TRAIN GEN: minimize `log(DISC(GEN(z))`
			optimizer_G.zero_grad()

			output_fake = DISC(sr_img).view(-1)
			pixel_loss, adversarial_loss, vgg_loss = criterion_G(sr_img, hr_img, output_fake)
			err_G = pixel_loss + adversarial_loss + vgg_loss
			err_G.backward()

			optimizer_G.step()

			running_adversarial_loss += adversarial_loss
			running_pixel_loss += pixel_loss
			running_vgg_loss = vgg_loss

			# Print stats
			if batch%10==0:
				print(f"\tBatch: {batch + 1}/{len(data_train_hr) // BATCH_SIZE}")
				D_x = output_real.mean().item()
				D_Gz2 = output_fake.mean().item()
				print(f"\terr_D_real: {err_D_real.item():.4f}; err_D_fake: {err_D_fake.item():.4f}; "
					  f" err_G: {err_G.item():.4f}; D_x: {D_x:.4f}; D_Gz1: {D_Gz1:.4f}; D_Gz2: {D_Gz2:.4f}")
				print(f"\t adversarial_loss: {adversarial_loss:.4f}, vgg_loss: {vgg_loss:.4f}, "
					  f"pixel_loss: {pixel_loss:.4f}")
			## Free up GPU memory
			del hr_img, lr_img, err_D_fake, err_D_real, err_G, real_labels, fake_labels, \
				output_real, output_fake, sr_img, pixel_loss, adversarial_loss, vgg_loss
			torch.cuda.empty_cache()
		
		exp_lr_scheduler_G.step()
		exp_lr_scheduler_D.step()

		epoch_adversarial_loss = running_adversarial_loss/len(hr_train_loader)
		epoch_pixel_loss = running_pixel_loss/len(hr_train_loader)
		epoch_vgg_loss = running_vgg_loss/len(hr_train_loader)
		
		training_adversarial_loss.append(epoch_adversarial_loss.detach().cpu())
		training_pixel_loss.append(epoch_pixel_loss.detach().cpu())
		training_vgg_loss.append(epoch_vgg_loss.detach().cpu())		

		### Save checkpoints
		save_checkpoints(GEN, DISC, optimizer_G, optimizer_D, epoch=prev_epochs+e+1)
	return training_adversarial_loss, training_pixel_loss, training_vgg_loss

def save_checkpoints(GEN, DISC, optimizer_G, optimizer_D, epoch):
	checkpoint_G = {
		'model': GEN,
		'state_dict': GEN.state_dict(),
		'optimizer': optimizer_G.state_dict(),
		'epoch': epoch
	}
	checkpoint_D = {
		'model': DISC,
		'state_dict': DISC.state_dict(),
		'optimizer': optimizer_D.state_dict(),
	}
	torch.save(checkpoint_G, PATH_G)
	torch.save(checkpoint_D, PATH_D)

def load_checkpoints(GEN, DISC, optimizerG, optimizerD):
	print("Loading checkpoints ...")
	checkpoint_G = torch.load(PATH_G)
	checkpoint_D = torch.load(PATH_D)
	GEN.load_state_dict(checkpoint_G['state_dict'])
	optimizerG.load_state_dict(checkpoint_G['optimizer'])
	DISC.load_state_dict(checkpoint_D['state_dict'])
	optimizerD.load_state_dict(checkpoint_D['optimizer'])
	prev_epochs = checkpoint_G['epoch']

	print("Loaded checkpoints successfully!")
	return GEN, DISC, optimizerG, optimizerD, prev_epochs

def xavier_init_weights(model):
	if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
		# torch.nn.init.xavier_uniform_(model.weight)
		torch.nn.init.xavier_normal_(model.weight)

if __name__ == "__main__":
	training_adversarial_loss, training_pixel_loss, training_vgg_loss = train(resume_training=True)