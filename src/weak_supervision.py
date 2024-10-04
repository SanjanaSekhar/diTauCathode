# Perform weak supervision to create an idealized anomaly detector

import time
from time import sleep
from tqdm import tqdm
import struct
import math
## External Library
import numpy as np
import matplotlib.pyplot as plt
import os,sys
## Pytorch Imports
import torch
torch.cuda.empty_cache()
import gc
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve
from argparse import ArgumentParser

from make_train_val_test import *
from train_test_bdt import *
from train_test_nn import *
#from plotter import plot_pre_postprocessed

parser = ArgumentParser(description='Train sig vs bkg for identifying CATHODE vars')
parser.add_argument("--name",  default="Phi250vsDY", help="file name extension for residuals and pulls")
parser.add_argument("--sig", default="2HDM-vbfPhiToTauTau-M250_2J_MinMass120_NoMisTag", help = "name of the .csv file for the signal")
parser.add_argument("--sigma",  default=2, type=float, help="value of S/sqrt(B) for signal injection")
parser.add_argument("--early_stop",  default=5, type = int, help="early stopping patience (no. of epochs)")
parser.add_argument("--batch_size",  default=128, type = int, help="batch size for training")
parser.add_argument("--n_epochs",  default=10, type = int, help="no. of epochs to train for")
parser.add_argument("--ending",  default="042624", help="date")
parser.add_argument("--BDT",  default=False, help="Use HistGradientBoostingClassifier instead of NN")
parser.add_argument("--load_model",  default=False, help="load saved model")
parser.add_argument("--epoch_to_load",  default=4, type = int, help="load checkpoint corresponding to this epoch")
parser.add_argument("--train_model",  default=False, help="train and save model")
parser.add_argument("--test_model",  default=False, help="test model")
parser.add_argument("--full_supervision",  default=False, help="Run fully supervised")
parser.add_argument("--sig_injection",  default=0.01, type=float , help="percent of signal to inject into data")
parser.add_argument("--train_frac",  default=0.7, type=float , help="fraction of samples to train on")
parser.add_argument("--val_frac",  default=0.1, type=float , help="fraction of samples to validate on")
parser.add_argument("--bkg_frac",  default=5, type=float, help="n_bkg/n_sig")
parser.add_argument("--m_tt_min",  default=120., type=float, help="lower boundary for sig region in ditau inv mass")
parser.add_argument("--m_tt_max",  default=500., type=float, help="upper boundary for sig region in ditau inv mass")
parser.add_argument("--feature_imp",  default=False, help="Plot feature_importance_")
parser.add_argument("--plot_pre_post",  default=False, help="Plot sig and bkg pre and postproc")
parser.add_argument("--test_ws",  default=False, help="test WS with gaussians")
parser.add_argument("--choose_n_features",  default=10, type = int, help="extract n best features")
parser.add_argument("--case",  default=-1, type = int, help="which subset of features do you want to try? choose between 1 to 4")
options = parser.parse_args()


ending = options.ending
name = options.name+"_sigma%.1f"%options.sigma
load_model = options.load_model
train_model = options.train_model
test_model = options.test_model
early_stop = options.early_stop
batch_size = options.batch_size
epochs = options.n_epochs
sig_injection = options.sig_injection
bkg_sig_frac = options.bkg_frac
m_tt_min = options.m_tt_min
m_tt_max = options.m_tt_max
epoch_to_load = options.epoch_to_load
train_frac, val_frac = options.train_frac, options.val_frac
if options.full_supervision: name += "_fs" 


options.m_tt_min = 100.
options.m_tt_max = 500.

if "Phi250" in name: options.sig = "2HDM-vbfPhiToTauTau-M250_2J_MinMass120_NoMisTag"
if "VAL" in name: options.sig = "VAL_dyVfVfToXiCXiCToTauSTauS_XiM1000_VfM250_MinMass120_NoMisTag"
if "TS250" in name: options.sig = "eVLQ_TPrimeTPrimeToTTPhiPhiToTauTauAll_TpM1000_PhiM250_NoMisTag"
if "HNL" in name: options.sig = "HeavyN_vbsNToTauTau_NM250_2J_LO"

case = options.case 
if case == 1: 
        f_list = ["tau1_m","tau2_m","deltaR_tau1tau2","met_met",
                        "m_tau1tau2","label"]
elif case == 2:
        f_list = ["m_jet1jet2", "deltaR_jet1jet2","deltaR_tau1tau2","met_met",
                        "m_tau1tau2","label"]
elif case == 3:
        f_list = ["m_jet1jet2", "deltaR_jet1jet2","tau1_m","tau2_m",
                        "m_tau1tau2","label"]
elif case == 4:
        f_list = ["m_jet1jet2", "deltaR_jet1jet2","tau1_m","deltaR_tau1tau2",
                        "m_tau1tau2","label"]
elif case == -1: 
        f_list = ["m_jet1jet2", "deltaR_jet1jet2","deltaR_tau1tau2","pt_tau1tau2","n_jets","n_bjets","met_met","met_eta","met_phi","jet1_pt",
                         "m_tau1tau2","label"]

print(options)

print(" Will use the following features: ", f_list)

n_features = len(f_list)-2

if not options.test_ws:
        sig = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/%s.csv"%options.sig, lineterminator='\n')
        bkg1 = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/SM_dyToTauTau_0J1J2J_MinMass120_1M.csv",lineterminator='\n')
        bkg2 = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/SM_ttbarTo2Tau2Nu_0J1J2J_MinMass120_MadSpin_1M.csv",lineterminator='\n')

else:
        x1_sig, x2_sig = np.random.multivariate_normal([7,7], np.diag((0.5,0.5)), 50000).T
        y_sig = np.ones((50000))
        x1_bkg, x2_bkg = np.random.multivariate_normal([4,4], np.diag((4,4)), 100000).T
        y_bkg = np.zeros((100000))
        sig = np.vstack((x1_sig, x2_sig, y_sig)).T
        bkg1 = np.vstack((x1_bkg, x2_bkg, y_bkg)).T

# DY
train_bkg1, val_bkg1, test_bkg1, train_bkg1_ws, val_bkg1_ws, test_bkg1_ws, n_bkg1 = make_bkg_ws_fs(options.test_ws, bkg1, options.m_tt_min, options.m_tt_max, -1, train_frac, val_frac, f_list)
# ttbar
train_bkg2, val_bkg2, test_bkg2, train_bkg2_ws, val_bkg2_ws, test_bkg2_ws, n_bkg2 = make_bkg_ws_fs(options.test_ws, bkg2, options.m_tt_min, options.m_tt_max, 25659, train_frac, val_frac, f_list)
n_sig_bkg1 = int(options.sigma * np.sqrt(n_bkg1 + n_bkg2))
print(n_bkg1, n_bkg2, options.sigma, n_sig_bkg1)
# signal 
train_sig, val_sig, test_sig, train_sig_ws, val_sig_ws, test_sig_ws, feature_list = make_sig_ws_fs(options.test_ws, sig, options.m_tt_min, options.m_tt_max, n_sig_bkg1, train_frac, val_frac, f_list)

# sets with all true labels for full supervision and ROC curve
train = np.vstack((train_sig,train_bkg1,train_bkg2))
val = np.vstack((val_sig,val_bkg1,val_bkg2))
test = np.vstack((test_sig,test_bkg1,test_bkg2))
# sets with label = 0 for pure bkg and 1 for data
train_ws = np.vstack((train_sig_ws,train_bkg1_ws,train_bkg2_ws))
val_ws = np.vstack((val_sig_ws,val_bkg1_ws,val_bkg2_ws))
test_ws = np.vstack((test_sig_ws,test_bkg1_ws,test_bkg2_ws))

np.random.shuffle(train)
np.random.shuffle(val)
np.random.shuffle(test)
np.random.shuffle(train_ws)
np.random.shuffle(val_ws) 
np.random.shuffle(test_ws)

print("Final samples before training starts")
print("%s: train, val, test shapes: "%name, train_ws.shape, val_ws.shape, test_ws.shape)

train, val, test = preprocess(train, val, test)
train_ws, val_ws, test_ws = preprocess(train_ws, val_ws, test_ws)
n_features = len(feature_list[:-1])

if options.BDT:

        print("Using a HistGradientBoostingClassifier...")
        if options.full_supervision: train_test_BDT(train, val, test, options.name, options.sigma, full_supervision=True, n_folds=50)
        else: train_test_BDT(train_ws, val_ws, test, options.name, options.sigma, full_supervision=False, n_folds=50)
                
else:
        gpu_boole = torch.cuda.is_available()
        print("Is GPU available? ",gpu_boole)
        if load_model: print("Loading model... ")
        model = NN()
        if gpu_boole: model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(),
                        lr = 1e-3,
                        weight_decay = 1e-8)

        loss_function = torch.nn.BCELoss()

        if load_model:  loaded_epoch, losses, val_losses = load_trained_model(name, epoch_to_load)
        else:
                loaded_epoch = 0
                losses,val_losses = [],[]
        
        if train_model:
                name = options.name+ "_sigma%.1f"%options.sigma+"_train%.2f_val%.2f"%(train_frac, val_frac) 
                train = np.vstack((train,val))
                train_ws = np.vstack((train_ws,val_ws))
                kf = KFold(n_splits=10)
                for i, (train_idx, val_idx) in enumerate(kf.split(train)):
                        print(">> Training NN with %ith fold as validation"%i)
                        train_kf, val_kf = train[train_idx], train[val_idx]
                        train_ws_kf, val_ws_kf = train_ws[train_idx], train_ws[val_idx]
                        
                        train_loader_ws, val_loader_ws, test_loader_ws = make_loaders(train_ws_kf,test_ws,val_ws_kf,batch_size)
                        train_loader, val_loader, test_loader = make_loaders(train_kf,test,val_kf,batch_size)
                        name = options.name+ "_fold%i"%i + "_sigma%.3f"%options.sigma
                        if not options.full_supervision: train_NN(train_loader_ws, val_loader_ws, losses, val_losses, loaded_epoch, name)      
                        else: train_NN(train_loader, val_loader, losses, val_losses, loaded_epoch, name+"_fs") 
        
        if test_model:
                name = name = options.name+ "_sigma%.1f"%options.sigma
                if not train_model:
                        
                        train_loader_ws, val_loader_ws, test_loader_ws = make_loaders(train_ws,test_ws,val_ws,batch_size)
                        train_loader, val_loader, test_loader = make_loaders(train,test,val,batch_size)
                
                if not options.full_supervision: test_NN(test_loader_ws, test, name, options.sigma, kfold = True)
                else: test_NN(test_loader, test, name, options.sigma, kfold = True)

'''
===========================
        EXTRAS
===========================
'''

#if options.plot_pre_post: plot_pre_postprocessed(train, val, test, train_ws, val_ws, test_ws)
        

if options.feature_imp:
        
        if options.full_supervision: feature_select(train, name, feature_list, k = options.choose_n_features)
        else: feature_select(train_ws, name, feature_list, k = options.choose_n_features)

