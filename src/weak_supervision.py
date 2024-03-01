# Perform weak supervision to create an idealized anomaly detector


import time
from time import sleep
from tqdm import tqdm
import struct
import math
## External Library
import numpy as np
import matplotlib.pyplot as plt

## Pytorch Imports
import torch
torch.cuda.empty_cache()
import gc
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

sig = pd.read_csv("csv_files/LQ_nonResScalarLQ-M1000_2J.csv")
bkg1 = pd.read_csv("csv_files/SM_dyToTauTau_0J1J2J_wPU.csv")
bkg2 = pd.read_csv("csv_files/SM_ttbarTo2Tau2Nu_2J.csv")

# Format of csv file:
# tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2, isSig

sig.columns = ["pt_tau1", "eta_tau1", "phi_tau1", "pt_tau2", "eta_tau2", "phi_tau2", "m_tau1", "m_tau2", "m_tau1tau2", "label"]
bkg1.columns = ["pt_tau1", "eta_tau1", "phi_tau1", "pt_tau2", "eta_tau2", "phi_tau2", "m_tau1", "m_tau2", "m_tau1tau2", "label"]
bkg2.columns = ["pt_tau1", "eta_tau1", "phi_tau1", "pt_tau2", "eta_tau2", "phi_tau2", "m_tau1", "m_tau2", "m_tau1tau2", "label"]

print(sig.shape, bkg1.shape, bkg2.shape)
print("Min, max m_tt in sig: ", sig['m_tau1tau2'].min(), sig['m_tau1tau2'].max() )
print("Min, max m_tt in bkg1: ", bkg1['m_tau1tau2'].min(), bkg1['m_tau1tau2'].max() )
print("Min, max m_tt in bkg2: ", bkg2['m_tau1tau2'].min(), bkg2['m_tau1tau2'].max() )

sig_sigregion = sig.loc(sig['m_tau1tau2'] >= 800)
bkg1_sigregion = bkg1.loc(bkg1['m_tau1tau2']>= 800)
bkg2_sigregion = bkg2.loc(bkg2['m_tau1tau2']>= 800)
bkg1_bkgregion = bkg1.loc(bkg1['m_tau1tau2']< 800)
bkg2_bkgregion = bkg2.loc(bkg2['m_tau1tau2']< 800)

print("No. of samples with m_tt > 800 GeV in sig, bkg1 and bkg2")
print(sig_sigregion.shape, bkg1_sigregion.shape, bkg2_sigregion.shape)
print("No. of samples with m_tt < 800 GeV in bkg1 and bkg2")
print(bkg1_bkgregion.shape, bkg2_bkgregion.shape)

sig_injection = 0.006

 



train, validate, test = np.split(transformed, [int(.6*len(transformed)), int(.8*len(transformed))])

train_set = torch.tensor(train, dtype=torch.float32)
val_set = torch.tensor(validate, dtype=torch.float32)
test_set = torch.tensor(test, dtype=torch.float32)

train_loader = torch.utils.data.DataLoader(dataset = train_set,
	batch_size = batch_size,
	shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_set,
	batch_size = batch_size,
	shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set,
	batch_size = 1,
	shuffle = True)

# AUTOENCODER CLASS

class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()

		# ENCODER
		self.encoder = torch.nn.Sequential(
			#torch.nn.Linear(228*3 800),
			#torch.nn.ReLU(),
			#torch.nn.Linear(800, 700),
			#torch.nn.ReLU(),
			torch.nn.Linear(228*3, 600),
			torch.nn.ReLU(),
                        #torch.nn.Linear(600, 500),
                        #torch.nn.ReLU(),
                        torch.nn.Linear(600, 400),
                        torch.nn.ReLU(),
			#torch.nn.Linear(400, 300),
			#torch.nn.ReLU(),
			torch.nn.Linear(400, 200),
			torch.nn.ReLU(),
			#torch.nn.Linear(200, 100),
			#torch.nn.ReLU(),
			torch.nn.Linear(200, 100),
			torch.nn.ReLU(),
			torch.nn.Linear(100, 25),
			#torch.nn.ReLU(),
			#torch.nn.Linear(25, 10)
			)

		# DECODER
		self.decoder = torch.nn.Sequential(
			#torch.nn.Linear(10, 25),
			#torch.nn.ReLU(),
			torch.nn.Linear(25, 100),
			torch.nn.ReLU(),
			#torch.nn.Linear(50, 100),
			#torch.nn.ReLU(),
			torch.nn.Linear(100, 200),
			torch.nn.ReLU(),
			#torch.nn.Linear(200, 300),
			#torch.nn.ReLU(),
			torch.nn.Linear(200, 400),
			torch.nn.ReLU(),
			#torch.nn.Linear(400, 500),
                        #torch.nn.ReLU(),
                        torch.nn.Linear(400, 600),
                        torch.nn.ReLU(),
                       	#torch.nn.Linear(600, 700),
                        #torch.nn.ReLU(),
                        torch.nn.Linear(600, 228*3),
                        #torch.nn.ReLU(),
                        #torch.nn.Linear(800, 228*4),
                         
			)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

model = AE()
if gpu_boole: model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
        lr = 2e-3,
        weight_decay = 1e-8)

#loss_function = torch.nn.MSELoss()
#loss_function = chamfer_distance()

# LOAD AN EXISTING MODEL 
if load_model:
	checkpoint = torch.load("checkpoints/ae_epoch3_%s.pth"%(ending))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loaded_epoch = checkpoint['epoch']
	print("loaded epoch = ",loaded_epoch)
	#loss_function = checkpoint['loss']
	#print("loaded loss = ",loss_function)
	train_val_losses = []
	
	with open("losses/train_val_losses_%s.txt"%ending,"r") as f:
		for line in f:
			train_val_losses.append(line.split(' '))
	train_val_losses = np.array(train_val_losses).astype("float32")
#	print(train_val_losses)
	losses = train_val_losses[:,0].tolist()
	val_losses = train_val_losses[:,1].tolist()
	'''
	losses,val_losses = [],[]
	'''
	
else:
	loaded_epoch = 0
	losses,val_losses = [],[]
	


print("train shape ",train.shape)
print("val shape ",validate.shape)
print("test shape ",test.shape)


outputs = []
test_losses = []

# TRAINING & VALIDATION LOOP

if not test_model:
	
	for epoch in range(loaded_epoch,epochs):

		loss_per_epoch, val_loss_per_epoch = 0,0
		i = 0
		with tqdm(train_loader, unit="batch") as tepoch:
			model.train()
			for event in tepoch:
				tepoch.set_description(f"Epoch {epoch}")
				if gpu_boole:
					event = event.cuda()

			  	# Output of Autoencoder
				reconstructed = model.forward(event)

			  	# Calculating the loss function
				event = torch.reshape(event, (batch_size,228,3))
				reconstructed = torch.reshape(reconstructed, (batch_size,228,3))
				#loss = loss_function(reconstructed, event)
				loss,_ = chamfer_distance(reconstructed, event)
			 
				#if epoch > 0 and epoch != loaded_epoch:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			 	 # Adding up all losses in a batch
				#i+=1
				#print("loss %i = "%i,loss)
				#print("loss.cpu().data.numpy().item() = ",loss.cpu().data.numpy().item())
				loss_per_epoch += loss.cpu().data.numpy().item()
				sleep(0.1)
		
		this_loss = loss_per_epoch/math.ceil(train.shape[0]/batch_size)
		torch.save({
			'epoch':epoch,
			'model_state_dict': model.state_dict(),
	            	'optimizer_state_dict': optimizer.state_dict(),
	            	#'loss': loss_function
			},
			"checkpoints/ae_epoch%i_%s.pth"%(epoch%5,ending))
		losses.append(this_loss)
		print("Train Loss: %f"%(this_loss))
		
		# VALIDATION

		for event in val_loader:
			model.eval()
			if gpu_boole:
				event = event.cuda()
			
			
			reconstructed = model.forward(event)
			event = torch.reshape(event, (batch_size,228,3))
			reconstructed = torch.reshape(reconstructed, (batch_size,228,3))
			#val_loss = loss_function(reconstructed, event)
			
			val_loss,_ = chamfer_distance(reconstructed, event)
			val_loss_per_epoch += val_loss.cpu().data.numpy().item()

		val_losses.append(val_loss_per_epoch/math.ceil(validate.shape[0]/batch_size))
		print("Val Loss: %f"%(val_loss_per_epoch/math.ceil(validate.shape[0]/batch_size)))
		
		# EARLY STOPPING
		flag = 0
		if early_stop > 0 and epoch > loaded_epoch + early_stop:
			for e in range(early_stop):
				if val_losses[-e] > val_losses[-e-1]: flag += 1
			if flag == early_stop:
				print("STOPPING TRAINING EARLY, VAL LOSS HAS BEEN INCREASING FOR THE LAST %i EPOCHS"%early_stop)
				break

		with open("losses/train_val_losses_%s.txt"%ending,"w") as f:
			for loss, val_loss in zip(losses, val_losses):
				f.write(str(loss)+" "+str(val_loss)+"\n")

	print("========== TRAINING COMPLETE ===========")

# TESTING
if test_model:

	test_loss_per_epoch = 0.
	input_list, output_list = np.zeros((1,228*3)), np.zeros((1,228*3))
	for idx,event in enumerate(test_loader):
		if idx==0: print(event.numpy())
		if gpu_boole:
			event = event.cuda()

	  
		reconstructed = model.forward(event)
		event = torch.reshape(event, (1,228,3))
		reconstructed = torch.reshape(reconstructed, (1,228,3))
		#test_loss = loss_function(reconstructed, event)
		test_loss,_ = chamfer_distance(reconstructed,event)
		test_loss_per_epoch += test_loss.cpu().data.numpy().item()
		if idx < 2000:
			event = torch.reshape(event, (1,228*3))
			reconstructed = torch.reshape(reconstructed, (1,228*3))			
			#print("Loss for this input: ",test_loss.cpu().data.numpy().item())
			input_list = np.vstack((input_list,(event.cpu().detach().numpy())))
			output_list = np.vstack((output_list,(reconstructed.cpu().detach().numpy())))
			if idx==0: print(event.cpu().detach().numpy())
	test_losses.append(test_loss_per_epoch/int(test.shape[0]))
	print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))