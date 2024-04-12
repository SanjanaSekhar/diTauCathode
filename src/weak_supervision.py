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
import numpy as np 

ending = "041224"
label = "PhivsDY"
load_model = False
test_model = True
early_stop = 5
batch_size = 16
epochs = 20
sig_injection = 0.2
bkg_sig_frac = 5

gpu_boole = torch.cuda.is_available()
print("Is GPU available? ",gpu_boole)
if load_model: print("Loading model... ")

# NN

class NN(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(23,32),
			torch.nn.ReLU(),
                        torch.nn.Linear(32,64),
                        torch.nn.ReLU(),
			torch.nn.Linear(64,32),
			torch.nn.ReLU(),
			torch.nn.Linear(32,1),
			torch.nn.Sigmoid()
			)

		
                         
	def forward(self, x):
		label = self.classifier(x)
		return label

def make_loaders(train,test,val,batch_size):
	train_set = torch.tensor(train, dtype=torch.float32)
	val_set = torch.tensor(val, dtype=torch.float32)
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
	return train_loader, val_loader, test_loader

def train_ws(train_loader,val_loader,losses,val_losses,loaded_epoch,label):

	print("================= Training %s ================="%label)
	outputs = []
	
	for epoch in range(loaded_epoch,epochs):

		loss_per_epoch, val_loss_per_epoch = 0,0
		i = 0
		with tqdm(train_loader, unit="batch") as tepoch:
			model.train()
			for vector in tepoch:
				tepoch.set_description(f"Epoch {epoch}")
				features, label = vector[:,:23],vector[:,23]
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
			"checkpoints/weak_supervision_epoch%i_%s.pth"%(epoch%5,label))
		losses.append(this_loss)
		print("Train Loss: %f"%(this_loss))
		
		# VALIDATION

		for vector in val_loader:
			model.eval()
			features, label = vector[:,:23],vector[:,23]
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

def test_ws(test_loader):
	test_loss_per_epoch = 0.
	test_losses = []
	for vector in test_loader:
		features, label = vector[:,:23],vector[:,23]
		if gpu_boole:
			features,label = features.cuda(),label.cuda()
		prediction = model.forward(features)
		test_loss = loss_function(prediction, label.view(-1,1))
		test_loss_per_epoch += test_loss.cpu().data.numpy().item()
	
	test_losses.append(test_loss_per_epoch/int(test.shape[0]))
	print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))


sig = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/2HDM-vbfPhiToTauTau-M750_2J_MinMass120_NoMisTag.csv", lineterminator='\n')
bkg1 = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/SM_dyToTauTau_0J1J2J_MinMass120_NoMisTag.csv",lineterminator='\n')
bkg2 = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/SM_ttbarTo2Tau2Nu_2J_MinMass120_NoMisTag.csv",lineterminator='\n')

# Format of csv file:
# tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
# tau2_m, m_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
# jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef, isSig
print(sig.shape, bkg1.shape, bkg2.shape)
sig.columns = ["tau1_pt", "tau1_eta", "tau1_phi", "tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m",
				"m_tau1tau2", "met_met", "met_eta", "met_phi", "n_jets", "n_bjets",
				"jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef", "label"]
bkg1.columns = sig.columns
bkg2.columns = sig.columns


print("Min, max m_tt in sig: ", sig['m_tau1tau2'].min(), sig['m_tau1tau2'].max() )
print("Min, max m_tt in bkg1: ", bkg1['m_tau1tau2'].min(), bkg1['m_tau1tau2'].max() )
print("Min, max m_tt in bkg2: ", bkg2['m_tau1tau2'].min(), bkg2['m_tau1tau2'].max() )

# Choose the m_tautau window in which to define 'data' and bkg regions
# The max ditau inv mass is not the same in all samples
m_tt_min = 350 
m_tt_max = 1000

# define "data" and "bkg" regions
sig_sigregion = sig[sig['m_tau1tau2'] >= m_tt_min] #and sig[sig['m_tau1tau2'] <= m_tt_max]
sig_sigregion = sig_sigregion[sig_sigregion['m_tau1tau2'] < m_tt_max]

bkg1_sigregion = bkg1[bkg1['m_tau1tau2'] >=  m_tt_min] #and bkg1[bkg1['m_tau1tau2'] <= m_tt_max]
bkg1_sigregion = bkg1_sigregion[bkg1_sigregion['m_tau1tau2'] < m_tt_max]

bkg2_sigregion = bkg2[bkg2['m_tau1tau2']>= m_tt_min] #and bkg2[bkg2['m_tau1tau2'] <= m_tt_max]
bkg2_sigregion = bkg2_sigregion[bkg2_sigregion['m_tau1tau2'] < m_tt_max]

bkg1_bkgregion = pd.concat([bkg1[bkg1['m_tau1tau2']< m_tt_min], bkg1[bkg1['m_tau1tau2'] >= m_tt_max]])
bkg2_bkgregion = pd.concat([bkg2[bkg2['m_tau1tau2']< m_tt_min], bkg2[bkg2['m_tau1tau2'] >= m_tt_max]])

print("No. of samples in SR in sig, bkg1 and bkg2")
print(sig_sigregion.shape[0], bkg1_sigregion.shape[0], bkg2_sigregion.shape[0])
print("No. of samples in SB in bkg1 and bkg2")
print(bkg1_bkgregion.shape[0], bkg2_bkgregion.shape[0])

# We want to ensure that the sig/bkg ratio in the "data" is realistic and small
# choose at random signal samples to inject into the data 
sig_injection = 0.20
n_sig_bkg1 = int((sig_injection/(1-sig_injection)) * (bkg1_sigregion.shape[0]))
n_sig_bkg2 = int((sig_injection/(1-sig_injection)) * (bkg2_sigregion.shape[0]))

print("No of signal samples to inject into bkg1 = ",n_sig_bkg1)
print("No of signal samples to inject into bkg2 = ",n_sig_bkg2)

sig_bkg1_idxs = np.random.choice(range(0,sig_sigregion.shape[0]),size=n_sig_bkg1)
sig_bkg2_idxs = np.random.choice(range(0,sig_sigregion.shape[0]),size=n_sig_bkg2) 
#print(sig_bkg1_idxs, sig_bkg2_idxs)
sig_to_inject_bkg1 = sig_sigregion.loc[sig_sigregion.index[sig_bkg1_idxs]]
sig_to_inject_bkg2 = sig_sigregion.loc[sig_sigregion.index[sig_bkg2_idxs]]
print(sig_to_inject_bkg1.shape,sig_to_inject_bkg2.shape)

# define data and bkg vectors
# label data as 1 and bkg as 0
bkg1_sigregion_ws = bkg1_sigregion.copy()
bkg2_sigregion_ws = bkg2_sigregion.copy()
bkg1_sigregion_ws.loc[:,'label'] = 1
bkg2_sigregion_ws.loc[:,'label'] = 1
#print(bkg1_sigregion)

sig_to_inject_bkg1_ws = sig_to_inject_bkg1.copy()
sig_to_inject_bkg2_ws = sig_to_inject_bkg2.copy()

sig_to_inject_bkg1_ws = sig_to_inject_bkg1_ws.drop(['m_tau1tau2'],axis=1).to_numpy()
sig_to_inject_bkg2_ws = sig_to_inject_bkg2_ws.drop(['m_tau1tau2'],axis=1).to_numpy()
bkg1_sigregion_ws = bkg1_sigregion_ws.drop(['m_tau1tau2'],axis=1).to_numpy()
bkg2_sigregion_ws = bkg2_sigregion_ws.drop(['m_tau1tau2'],axis=1).to_numpy()

# train val test split: 0.7, 0.1, 0.2
train_sig_bkg1, val_sig_bkg1, test_sig_bkg1 = np.split(sig_to_inject_bkg1_ws, [int(.8*len(sig_to_inject_bkg1_ws)), int(.9*len(sig_to_inject_bkg1_ws))])
train_sig_bkg2, val_sig_bkg2, test_sig_bkg2 = np.split(sig_to_inject_bkg2_ws, [int(.8*len(sig_to_inject_bkg2_ws)), int(.9*len(sig_to_inject_bkg2_ws))])
train_bkg1, val_bkg1, test_bkg1 = np.split(bkg1_sigregion_ws, [int(.8*len(bkg1_sigregion_ws)), int(.9*len(bkg1_sigregion_ws))])
train_bkg2, val_bkg2, test_bkg2 = np.split(bkg2_sigregion_ws, [int(.8*len(bkg2_sigregion_ws)), int(.9*len(bkg2_sigregion_ws))]) 

print("train_sig_bkg1.shape, train_bkg1.shape = ",train_sig_bkg1.shape, train_bkg1.shape)

train = np.vstack((train_sig_bkg1,train_bkg1))
val = np.vstack((val_sig_bkg1,val_bkg1))
test = np.vstack((test_sig_bkg1,test_bkg1))

train2 = np.vstack((train_sig_bkg2,train_bkg2))
val2 = np.vstack((val_sig_bkg2,val_bkg2))
test2 = np.vstack((test_sig_bkg2,test_bkg2))

print("DY : train, val, test shapes: ",train.shape, val.shape, test.shape)
print("ttbar: train, val, test shapes: ",train2.shape, val2.shape, test2.shape)

bkg1_idxs = np.random.choice(range(0,bkg1_bkgregion.shape[0]),size=(train.shape[0]+test.shape[0]+val.shape[0])*bkg_sig_frac)
bkg2_idxs = np.random.choice(range(0,bkg2_bkgregion.shape[0]),size=(train2.shape[0]+test2.shape[0]+val2.shape[0])*bkg_sig_frac)

bkg1_bkgregion_ws = bkg1_bkgregion.loc[bkg1_bkgregion.index[bkg1_idxs]]
bkg2_bkgregion_ws = bkg2_bkgregion.loc[bkg2_bkgregion.index[bkg2_idxs]]

bkg1_bkgregion_ws = bkg1_bkgregion_ws.drop(['m_tau1tau2'],axis=1).to_numpy()
bkg2_bkgregion_ws = bkg2_bkgregion_ws.drop(['m_tau1tau2'],axis=1).to_numpy()

train_bkg1, val_bkg1, test_bkg1 = np.split(bkg1_bkgregion_ws, [int(.8*len(bkg1_bkgregion_ws)), int(.9*len(bkg1_bkgregion_ws))])
train_bkg2, val_bkg2, test_bkg2 = np.split(bkg2_bkgregion_ws, [int(.8*len(bkg2_bkgregion_ws)), int(.9*len(bkg2_bkgregion_ws))])

train = np.vstack((train,train_bkg1))
val = np.vstack((val,val_bkg1))
test = np.vstack((test,test_bkg1))

train2 = np.vstack((train2,train_bkg2))
val2 = np.vstack((val2,val_bkg2))
test2 = np.vstack((test2,test_bkg2))

print("Final samples before training starts")
print("DY: train, val, test shapes: ",train.shape, val.shape, test.shape)
print("ttbar: train, val, test shapes: ",train2.shape, val2.shape, test2.shape)


model = NN()
if gpu_boole: model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),
        lr = 1e-3,
        weight_decay = 1e-8)

loss_function = torch.nn.BCELoss()
#loss_function = chamfer_distance()


# LOAD AN EXISTING MODEL 
if load_model:
	checkpoint = torch.load("checkpoints/weak_supervision_epoch2_%s.pth"%label)
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
	

train_loader, val_loader, test_loader = make_loaders(train,test,val,batch_size)
train_ws(train_loader,val_loader,losses,val_losses,loaded_epoch,label)
if test_model: test_ws(test_loader)

label = "Phivsttbar"
train_loader, val_loader, test_loader = make_loaders(train2,test2,val2,batch_size)
train_ws(train_loader,val_loader,losses,val_losses,loaded_epoch,label)
if test_model: test_ws(test_loader)



	
