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

ending = "030124"
load_model = False
test_model = False
early_stop = 5
batch_size = 16
epochs = 20


gpu_boole = torch.cuda.is_available()
print("Is GPU available? ",gpu_boole)
if load_model: print("Loading model... ")


sig = pd.read_csv("csv_files/LQ_nonResScalarLQ-M1000_2J.csv", lineterminator='\n')
bkg1 = pd.read_csv("csv_files/SM_dyToTauTau_0J1J2J_MinMass120.csv")
bkg2 = pd.read_csv("csv_files/SM_ttbarTo2Tau2Nu_2J.csv")

# Format of csv file:
# tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2, isSig
print(sig.shape, bkg1.shape, bkg2.shape)
sig.columns = ["pt_tau1", "eta_tau1", "phi_tau1", "pt_tau2", "eta_tau2", "phi_tau2", "m_tau1", "m_tau2", "m_tau1tau2", "label"]
bkg1.columns = ["pt_tau1", "eta_tau1", "phi_tau1", "pt_tau2", "eta_tau2", "phi_tau2", "m_tau1", "m_tau2", "m_tau1tau2", "label"]
bkg2.columns = ["pt_tau1", "eta_tau1", "phi_tau1", "pt_tau2", "eta_tau2", "phi_tau2", "m_tau1", "m_tau2", "m_tau1tau2", "label"]

print(sig.shape, bkg1.shape, bkg2.shape)
print(sig)
print("Min, max m_tt in sig: ", sig['m_tau1tau2'].min(), sig['m_tau1tau2'].max() )
print("Min, max m_tt in bkg1: ", bkg1['m_tau1tau2'].min(), bkg1['m_tau1tau2'].max() )
print("Min, max m_tt in bkg2: ", bkg2['m_tau1tau2'].min(), bkg2['m_tau1tau2'].max() )

# Choose the m_tautau window in which to define 'data' and bkg regions
# The max ditau inv mass is not the same in all samples
m_tt_min = 700 
m_tt_max = np.min([sig['m_tau1tau2'].max(), bkg1['m_tau1tau2'].max(), bkg2['m_tau1tau2'].max()])
print(m_tt_min, m_tt_max)

# define "data" and "bkg" regions
sig_sigregion = sig[sig['m_tau1tau2'] >= m_tt_min] #and sig[sig['m_tau1tau2'] <= m_tt_max]
bkg1_sigregion = bkg1[bkg1['m_tau1tau2']>=  m_tt_min] #and bkg1[bkg1['m_tau1tau2'] <= m_tt_max]
bkg2_sigregion = bkg2[bkg2['m_tau1tau2']>= m_tt_min] #and bkg2[bkg2['m_tau1tau2'] <= m_tt_max]
bkg1_bkgregion = bkg1[bkg1['m_tau1tau2']< m_tt_min]
bkg2_bkgregion = bkg2[bkg2['m_tau1tau2']< m_tt_min]

print("Percent of samples with m_tt > %f GeV in sig, bkg1 and bkg2"%m_tt_min)
print(sig_sigregion.shape[0]/sig.shape[0], bkg1_sigregion.shape[0]/bkg1.shape[0], bkg2_sigregion.shape[0]/bkg2.shape[0])
print("Percent of samples with m_tt < %f GeV in bkg1 and bkg2"%m_tt_min)
print(bkg1_bkgregion.shape[0]/bkg1.shape[0], bkg2_bkgregion.shape[0]/bkg2.shape[0])

# We want to ensure that the sig/bkg ratio in the "data" is realistic and small
# choose at random signal samples to inject into the data 
sig_injection = 0.006
n_sig = int((sig_injection/(1-sig_injection)) * (bkg1_sigregion.shape[0]+bkg2_sigregion.shape[0]))
sig_idxs = np.random.choice(range(0,sig_sigregion.shape[0]),size=n_sig) 
print(sig_idxs)
sig_to_inject = sig_sigregion.loc[sig_sigregion.index[sig_idxs]]
print(sig_to_inject.shape)
# define data and bkg vectors
# label data as 1 and bkg as 0
bkg1_sigregion.loc[:,'label'] = 1
bkg2_sigregion.loc[:,'label'] = 1
print(bkg1_sigregion)
sig_to_inject = sig_to_inject.drop(['m_tau1tau2'],axis=1).to_numpy()
bkg1_sigregion = bkg1_sigregion.drop(['m_tau1tau2'],axis=1).to_numpy()
bkg2_sigregion = bkg2_sigregion.drop(['m_tau1tau2'],axis=1).to_numpy()

# train val test split: 0.7, 0.1, 0.2
train_sig, val_sig, test_sig = np.split(sig_to_inject, [int(.8*len(sig_to_inject)), int(.9*len(sig_to_inject))])
train_bkg1, val_bkg1, test_bkg1 = np.split(bkg1_sigregion, [int(.8*len(bkg1_sigregion)), int(.9*len(bkg1_sigregion))])
train_bkg2, val_bkg2, test_bkg2 = np.split(bkg2_sigregion, [int(.8*len(bkg2_sigregion)), int(.9*len(bkg2_sigregion))]) 

train = np.vstack((train_sig,train_bkg1,train_bkg2))
val = np.vstack((val_sig,val_bkg1,val_bkg2))
test = np.vstack((test_sig,test_bkg1,test_bkg2))

print("train, val, test shapes: ",train.shape, val.shape, test.shape)


train_set = torch.tensor(train, dtype=torch.float32)
val_set = torch.tensor(val, dtype=torch.float32)
test_set = torch.tensor(test, dtype=torch.float32)

batch_size = 16
epochs = 20

train_loader = torch.utils.data.DataLoader(dataset = train_set,
	batch_size = batch_size,
	shuffle = True)
val_loader = torch.utils.data.DataLoader(dataset = val_set,
	batch_size = batch_size,
	shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_set,
	batch_size = batch_size,
	shuffle = True)

# NN

class NN(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(8,32),
			torch.nn.ReLU(),
                        torch.nn.Linear(32,64),
                        torch.nn.ReLU(),
			torch.nn.Linear(64, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32,1),
			torch.nn.Sigmoid()
			)

		
                         
	def forward(self, x):
		label = self.classifier(x)
		return label

model = NN()
if gpu_boole: model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
        lr = 2e-3,
        weight_decay = 1e-8)

loss_function = torch.nn.BCELoss()
#loss_function = chamfer_distance()

# LOAD AN EXISTING MODEL 
if load_model:
	checkpoint = torch.load("checkpoints/weak_supervision_epoch3_%s.pth"%(ending))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loaded_epoch = checkpoint['epoch']
	print("loaded epoch = ",loaded_epoch)
	loss_function = checkpoint['loss']
	print("loaded loss = ",loss_function)
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
	

outputs = []
test_losses = []

# TRAINING & VALIDATION LOOP

if not test_model:
	
	for epoch in range(loaded_epoch,epochs):

		loss_per_epoch, val_loss_per_epoch = 0,0
		i = 0
		with tqdm(train_loader, unit="batch") as tepoch:
			model.train()
			for vector in tepoch:
				tepoch.set_description(f"Epoch {epoch}")
				features, label = vector[:,:8],vector[:,8]
				if gpu_boole:
					features,label = features.cuda(),label.cuda()

			  	# Output of NN
				prediction = model.forward(features)

			  	# Calculating the loss function
							
				loss = loss_function(prediction, label.view(-1,1))
			 
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
	            	'loss': loss_function
			},
			"checkpoints/weak_supervision_epoch%i.pth"%(epoch%5))
		losses.append(this_loss)
		print("Train Loss: %f"%(this_loss))
		
		# VALIDATION

		for vector in val_loader:
			model.eval()
			features, label = vector[:,:8],vector[:,8]
			if gpu_boole:
				features,label = features.cuda(),label.cuda()
			
			
			prediction = model.forward(features)
			
			val_loss = loss_function(prediction, label.view(-1,1))
			val_loss_per_epoch += val_loss.cpu().data.numpy().item()

		val_losses.append(val_loss_per_epoch/math.ceil(val.shape[0]/batch_size))
		print("Val Loss: %f"%(val_loss_per_epoch/math.ceil(val.shape[0]/batch_size)))
		
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
	for vector in enumerate(test_loader):
		features, label = vector[:,:8],vector[:,8]
		if gpu_boole:
			features,label = features.cuda(),label.cuda()
		prediction = model.forward(features)
		test_loss = loss_function(prediction, label.view(1,1))
		test_loss_per_epoch += test_loss.cpu().data.numpy().item()
	
	test_losses.append(test_loss_per_epoch/int(test.shape[0]))
	print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))
