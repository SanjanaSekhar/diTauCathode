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

sys.setrecursionlimit(1500)
# NN

class NN(torch.nn.Module):
        def __init__(self):
                super().__init__()

                self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(4,16),
                        torch.nn.ReLU(),
                        torch.nn.Linear(16,32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32,16),
                        torch.nn.ReLU(),
                        torch.nn.Linear(16,1),
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
                shuffle = False)
        return train_loader, val_loader, test_loader

def training(train_loader,val_loader,losses,val_losses,loaded_epoch,name):

        print("================= Training %s ================="%name)
        outputs = []
        
        for epoch in range(loaded_epoch,epochs):

                loss_per_epoch, val_loss_per_epoch = 0,0
                i = 0
                with tqdm(train_loader, unit="batch") as tepoch:
                        model.train()
                        for vector in tepoch:
                                tepoch.set_description(f"Epoch {epoch}")
                                n_features = vector.size()[1]-1
                                features, label = vector[:,:n_features],vector[:,n_features]
                                #print(features.size(),label.size())
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
                        "checkpoints/weak_supervision_epoch%i_%s.pth"%(epoch%5,name))
                losses.append(this_loss)
                print("Train Loss: %f"%(this_loss))
                
                # VALIDATION

                for vector in val_loader:
                        model.eval()
                        n_features = vector.size()[1]-1
                        features, label = vector[:,:n_features],vector[:,n_features]
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
                        for e in range(1,early_stop+1):
                                if val_losses[-e] > val_losses[-early_stop]: flag += 1
                        if flag == early_stop:
                                print("STOPPING TRAINING EARLY, VAL LOSS HAS BEEN INCREASING FOR THE LAST %i EPOCHS"%early_stop)
                                break

                with open("losses/train_val_losses_%s.txt"%ending,"w") as f:
                        for loss, val_loss in zip(losses, val_losses):
                                f.write(str(loss)+" "+str(val_loss)+"\n")

        print("========== TRAINING COMPLETE ===========")

def testing(test_loader_ws, test_true, name, kfold=False):

        if kfold:
                # train_frac = ["0.10","0.30","0.50","0.70"]  
                # val_frac = train_frac[::-1]
                # print(train_frac, val_frac)
                pred_list_all = []
                for i in range(10):
                        pth = "%s_fold%i_sig%0.3f"%(name.split("_")[0],i,sig_injection)
                        if options.full_supervision: pth +=  "_fs"
                        loaded_epoch, losses, val_losses = load_trained_model(pth, epoch_to_load)
                        pred_list = []
                        print("================= Testing %s ================="%pth)
                        test_loss_per_epoch = 0.
                        test_losses = []
                        for vector in test_loader_ws:
                                n_features = vector.size()[1]-1
                                features, label = vector[:,:n_features],vector[:,n_features]
                                if gpu_boole:
                                        features,label = features.cuda(),label.cuda()
                                prediction = model.forward(features)
                                test_loss = loss_function(prediction, label.view(-1,1))
                                test_loss_per_epoch += test_loss.cpu().data.numpy().item()
                                pred_list.append(prediction.cpu().data.numpy().item())
                        
                        test_losses.append(test_loss_per_epoch/int(test.shape[0]))
                        print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))
                        pred_list_all.append(pred_list) 
                
                pred_list_all = np.array(pred_list_all)
                pred_list = np.mean(pred_list_all, axis=0)
                print("After averaging results of kfold, predicted list shape = ", pred_list.shape)

                true_list = test_true[:,-1]
                print(true_list==1)
                # print(np.vstack((true_list,pred_list)))
                if options.full_supervision: np.savetxt("losses/fpr_tpr_%s_fs_sig%0.3f.txt"%(name.split("_")[0], sig_injection), np.vstack((true_list,pred_list)))
                else: np.savetxt("losses/fpr_tpr_%s_sig%0.3f.txt"%(name.split("_")[0], sig_injection), np.vstack((true_list,pred_list)))

        else:
                loaded_epoch, losses, val_losses = load_trained_model(name, epoch_to_load) 
                pred_list = []
                print("================= Testing %s ================="%name)
                test_loss_per_epoch = 0.
                test_losses = []
                for vector in test_loader_ws:
                        n_features = vector.size()[1]-1
                        features, label = vector[:,:n_features],vector[:,n_features]
                        #print(features[:10], label[:10])
                        if gpu_boole:
                                features,label = features.cuda(),label.cuda()
                        prediction = model.forward(features)
                        test_loss = loss_function(prediction, label.view(-1,1))
                        test_loss_per_epoch += test_loss.cpu().data.numpy().item()
                        pred_list.append(prediction.cpu().data.numpy().item())
                
                test_losses.append(test_loss_per_epoch/int(test.shape[0]))
                print("Test Loss: %f"%(test_loss_per_epoch/int(test.shape[0])))

                true_list = test_true[:,-1]
                print(true_list==1)
                # print(np.vstack((true_list,pred_list)))
                np.savetxt("losses/fpr_tpr_%s.txt"%name,np.vstack((true_list,pred_list)))
        




def make_train_test_val_ws(test_ws, sig, bkg1, m_tt_min = 350., m_tt_max = 1000., sig_injection = 0.2, bkg_sig_frac = 5, train_frac = 0.8, val_frac = 0.1, name = "PhivsDY", f_list = ["m_jet1jet2", "deltaR_jet1jet2"]):

        # CREATE an IDEAL Anomaly Detector
        # Train pure bkg vs sig+bkg in the SR only

        if test_ws:
                sig_sigregion = pd.DataFrame(sig)
                bkg1_sigregion = pd.DataFrame(bkg1)
        else:
                # Format of csv file:
                # tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
                # tau2_m, m_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
                # jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef,
                # jet2_pt, jet2_eta, jet2_phi, jet2_cef, jet2_nef, bjet2_pt, bjet2_eta, bjet2_phi, bjet2_cef, bjet2_nef, isSig

                print(sig.shape, bkg1.shape)
                sig.columns = [ "m_jet1jet2", "deltaR_jet1jet2", "m_bjet1bjet2", "deltaR_bjet1bjet2", "deltaR_tau1tau2",
                                "tau1_pt", "tau1_eta", "tau1_phi", "tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m",
                                "m_tau1tau2","pt_tau1tau2", "eta_tau1tau2", "phi_tau1tau2", "met_met", "met_eta", "met_phi", "n_jets", "n_bjets",
                                "jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef",
                                "jet2_pt", "jet2_eta", "jet2_phi", "jet2_cef", "jet2_nef", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef", "label"]
                bkg1.columns = sig.columns
                # deltaR of taus
                #sig["deltaR_taus"] = ((sig["tau1_eta"]-sig["tau2_eta"]).pow(2) + (sig["tau1_phi"]-sig["tau2_phi"]).pow(2)).pow(0.5)
                #bkg1["deltaR_taus"] = ((bkg1["tau1_eta"]-bkg1["tau2_eta"]).pow(2) + (bkg1["tau1_phi"]-bkg1["tau2_phi"]).pow(2)).pow(0.5)
                
                #sig.drop(labels=["tau1_pt", "tau2_pt", "tau1_m","tau2_m", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef"], axis=1, inplace=True)
                #bkg1.drop(labels=["tau1_pt", "tau2_pt", "tau1_m","tau2_m", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef"], axis=1, inplace=True)
                sig = sig[f_list]
                bkg1 = bkg1[f_list]
                
                print(sig.shape, bkg1.shape)
                print("Min, max m_tt in sig: ", sig['m_tau1tau2'].min(), sig['m_tau1tau2'].max() )
                print("Min, max m_tt in bkg1: ", bkg1['m_tau1tau2'].min(), bkg1['m_tau1tau2'].max() )

                # Choose the m_tautau window in which to define 'data' and bkg regions
                # The max ditau inv mass is not the same in all samples


                # define "data" and "bkg" regions
                sig_sigregion = sig[sig['m_tau1tau2'] >= m_tt_min] #and sig[sig['m_tau1tau2'] <= m_tt_max]
                sig_sigregion = sig_sigregion[sig_sigregion['m_tau1tau2'] < m_tt_max]

                bkg1_sigregion = bkg1[bkg1['m_tau1tau2'] >=  m_tt_min] #and bkg1[bkg1['m_tau1tau2'] <= m_tt_max]
                bkg1_sigregion = bkg1_sigregion[bkg1_sigregion['m_tau1tau2'] < m_tt_max]
                print("No. of samples in SR in sig, bkg")
                print(sig_sigregion.shape[0], bkg1_sigregion.shape[0])

                # shuffle the background indices
                bkg1_sigregion = bkg1_sigregion.sample(frac=1).reset_index(drop=True)

        # split background in 2, one for data one for pure bkg
        bkg1_bkgregion = bkg1_sigregion[0:int(bkg1_sigregion.shape[0]/2)]
        bkg1_sigregion = bkg1_sigregion[int(bkg1_sigregion.shape[0]/2):]
        #bkg1_bkgregion = pd.concat([bkg1[bkg1['m_tau1tau2']< m_tt_min], bkg1[bkg1['m_tau1tau2'] >= m_tt_max]])
        print("No. of bkg samples in data and pure bkg")
        print(bkg1_sigregion.shape[0], bkg1_bkgregion.shape[0])
        
        #print("bkg1_sigregion",bkg1_sigregion,"bkg1_bkgregion",bkg1_bkgregion)
        # We want to ensure that the sig/bkg ratio in the "data" is realistic and small
        # choose at random signal samples to inject into the data 

        #n_sig_bkg1 = int((sig_injection/(1-sig_injection)) * (bkg1_sigregion.shape[0]))
        n_sig_bkg1 = int(sig_injection * bkg1_sigregion.shape[0])
        print("No of signal samples to inject into bkg1 = ",n_sig_bkg1)

        sig_bkg1_idxs = np.random.choice(range(0,sig_sigregion.shape[0]),size=n_sig_bkg1)
        #print(sig_bkg1_idxs, sig_bkg2_idxs)
        sig_to_inject_bkg1 = sig_sigregion.loc[sig_sigregion.index[sig_bkg1_idxs]]
        #print(sig_to_inject_bkg1.shape)
    
        n_bkg1 = int((1-sig_injection) * bkg1_sigregion.shape[0])
        bkg1_idxs = np.random.choice(range(0,bkg1_sigregion.shape[0]),size=n_bkg1)
        bkg_to_inject_bkg1 = bkg1_sigregion.loc[bkg1_sigregion.index[bkg1_idxs]]
        bkg1_sigregion = bkg_to_inject_bkg1.copy()
        
        print("No of bkg samples in data = ",n_bkg1)
        print("Shape of signal and bkg in data: ",sig_to_inject_bkg1.shape, bkg1_sigregion.shape)


        # define data and bkg vectors
        # label data as 1 and pure bkg as 0
        # bkg1_sigregion has label = 0 in the data region, bkg1_sigregion_ws has label = 1
        bkg1_sigregion_ws = bkg1_sigregion.copy()
        if not test_ws: bkg1_sigregion_ws.loc[:,'label'] = 1
        else: bkg1_sigregion_ws[bkg1_sigregion_ws.columns[2]] = 1

        
        # sig_to_inject_bkg1 and sig_to_inject_bkg1_ws both have label = 1
        sig_to_inject_bkg1_ws = sig_to_inject_bkg1.copy()
        if not test_ws: sig_to_inject_bkg1 = sig_to_inject_bkg1.drop(['m_tau1tau2'],axis=1)
        feature_list = sig_to_inject_bkg1.columns
        sig_to_inject_bkg1 = sig_to_inject_bkg1.to_numpy()
        if not test_ws:
                sig_to_inject_bkg1_ws = sig_to_inject_bkg1_ws.drop(['m_tau1tau2'],axis=1)
                bkg1_sigregion_ws = bkg1_sigregion_ws.drop(['m_tau1tau2'],axis=1)
                bkg1_sigregion = bkg1_sigregion.drop(['m_tau1tau2'],axis=1)
        
        sig_to_inject_bkg1_ws = sig_to_inject_bkg1_ws.to_numpy()
        bkg1_sigregion_ws = bkg1_sigregion_ws.to_numpy()
        bkg1_sigregion = bkg1_sigregion.to_numpy()
        

        np.savetxt("sig_to_inject_bkg1.txt",sig_to_inject_bkg1)
        np.savetxt("sig_to_inject_bkg1_ws.txt",sig_to_inject_bkg1_ws)
        np.savetxt("bkg1_sigregion_ws.txt", bkg1_sigregion_ws)
        np.savetxt("bkg1_sigregion.txt",bkg1_sigregion)

        # train val test split: train_frac, train_frac + val_frac, 1-(train_frac + val_frac)
        train_sig, val_sig, test_sig = np.split(sig_to_inject_bkg1, [int(train_frac*len(sig_to_inject_bkg1)), int((train_frac + val_frac)*len(sig_to_inject_bkg1))])
        train_bkg1, val_bkg1, test_bkg1 = np.split(bkg1_sigregion, [int(train_frac*len(bkg1_sigregion)), int((train_frac + val_frac)*len(bkg1_sigregion))])
        train_sig_ws, val_sig_ws, test_sig_ws = np.split(sig_to_inject_bkg1_ws, [int(train_frac*len(sig_to_inject_bkg1_ws)), int((train_frac + val_frac)*len(sig_to_inject_bkg1_ws))])
        train_bkg1_ws, val_bkg1_ws, test_bkg1_ws = np.split(bkg1_sigregion_ws, [int(train_frac*len(bkg1_sigregion_ws)), int((train_frac + val_frac)*len(bkg1_sigregion_ws))])
        #print(train_bkg1_ws) 
        print("train_sig.shape, train_bkg1.shape, train_bkg1_ws.shape = ",train_sig.shape, train_bkg1.shape, train_bkg1_ws.shape)
        
        
        # sets with all true labels for full supervision and ROC curve
        train = np.vstack((train_sig,train_bkg1))
        val = np.vstack((val_sig,val_bkg1))
        test = np.vstack((test_sig,test_bkg1))
        # sets with label = 0 for SB (bkg) and 1 for SR (data)
        train_ws = np.vstack((train_sig_ws,train_bkg1_ws))
        val_ws = np.vstack((val_sig_ws,val_bkg1_ws))
        test_ws = np.vstack((test_sig_ws,test_bkg1_ws))


        print("%s : Weak supervision -  train, val, test shapes: "%name,train.shape, val.shape, test.shape)

        #bkg1_idxs = np.random.choice(range(0,bkg1_bkgregion.shape[0]),size=(train.shape[0]+test.shape[0]+val.shape[0])*bkg_sig_frac)
        # both bkg1_bkgregion and bkg1_bkgregion_ws have label = 0
        #bkg1_bkgregion_ws = bkg1_bkgregion.loc[bkg1_bkgregion.index[bkg1_idxs]]
        bkg1_bkgregion_ws = bkg1_bkgregion.copy()

        bkg1_bkgregion_ws = bkg1_bkgregion_ws.drop(['m_tau1tau2'],axis=1)

        bkg1_bkgregion_ws = bkg1_bkgregion_ws.to_numpy()
        np.savetxt("bkg1_bkgregion_ws.txt",bkg1_bkgregion_ws)

        train_bkg1, val_bkg1, test_bkg1 = np.split(bkg1_bkgregion_ws, [int(train_frac*len(bkg1_bkgregion_ws)), int((train_frac + val_frac)*len(bkg1_bkgregion_ws))])
        # sets with all true labels for full supervision and ROC curve
        train = np.vstack((train,train_bkg1))
        val = np.vstack((val,val_bkg1))
        test = np.vstack((test,test_bkg1))
        # sets with label = 0 for pure bkg and 1 for data
        train_ws = np.vstack((train_ws,train_bkg1))
        val_ws = np.vstack((val_ws,val_bkg1))
        test_ws = np.vstack((test_ws,test_bkg1))
        
        np.random.shuffle(train)
        np.random.shuffle(val)
        np.random.shuffle(test)
        np.random.shuffle(train_ws)
        np.random.shuffle(val_ws) 
        np.random.shuffle(test_ws)
 
        np.savetxt("train_ws.txt", train_ws)
        
        print("Final samples before training starts")
        print("%s: train, val, test shapes: "%name, train_ws.shape, val_ws.shape, test_ws.shape)
        print(train[:,-1], train_ws[:,-1])
        return train, val, test, train_ws, val_ws, test_ws, feature_list.to_list()

from sklearn.preprocessing import StandardScaler

def preprocess(train, val, test):
        print(train.shape)
        n_features = train.shape[1] - 1
        scaler = StandardScaler()
        scaler.fit(np.vstack((train,val))[:,0:n_features])
        train[:,0:n_features] = scaler.transform(train[:,0:n_features])
        test[:,0:n_features] = scaler.transform(test[:,0:n_features])
        val[:,0:n_features] = scaler.transform(val[:,0:n_features])
        print("After preprocessing, mean, std for the features:", scaler.mean_,scaler.var_)
        return train, val, test

# LOAD AN EXISTING MODEL 
def load_trained_model(name, epoch):
        checkpoint = torch.load("checkpoints/weak_supervision_epoch%i_%s.pth"%(epoch,name))
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
#       print(train_val_losses)
        losses = train_val_losses[:,0].tolist()
        val_losses = train_val_losses[:,1].tolist()
        '''
        losses,val_losses = [],[]
        '''
        return loaded_epoch, losses, val_losses
        
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2

def feature_select(vector, name, feature_list, k = 7):
        #vector = vector.drop(['m_tau1tau2'],axis=1)
        n_features = vector.shape[1]-1
        x = vector[:,0:n_features]
        y = vector[:,n_features]
        print(x.shape, y.shape)
        bestfeatures = SelectKBest(score_func=f_classif, k = k)
        bestfeatures.fit(x,y)
        print(bestfeatures.pvalues_)
        scores = [ 0 if pvalue == 0 else -np.log10(pvalue) for pvalue in bestfeatures.pvalues_] 
        #scores = -np.log10(bestfeatures.pvalues_)
        print(scores)
        scores /= max(scores)
    
        feature_l = feature_list[:-1]
        print(feature_l, len(feature_l))
        plt.figure(figsize=(10,10))
        plt.bar(feature_l, scores, width=0.2)
        plt.title("Feature univariate score")
        plt.xlabel("Features")
        plt.xticks(rotation=90)
        plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
        plt.savefig("feature_importance_%s.png"%name)
        plt.close()
        '''
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(x.columns)
        #concat two dataframes for better visualization 
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Specs','Score']  #naming the dataframe columns
        print(featureScores.nlargest(k,'Score'))
        ''' 

parser = ArgumentParser(description='Train sig vs bkg for identifying CATHODE vars')
parser.add_argument("--name",  default="Phi250vsDY", help="file name extension for residuals and pulls")
parser.add_argument("--sig", default="2HDM-vbfPhiToTauTau-M250_2J_MinMass120_NoMisTag", help = "name of the .csv file for the signal")
parser.add_argument("--bkg",  default="SM_dyToTauTau_0J1J2J_MinMass120_NoMisTag", help="name of the .csv file for the bkg")
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
parser.add_argument("--case",  default=2, type = int, help="which subset of features do you want to try? choose between 1 to 4")
options = parser.parse_args()


ending = options.ending
name = options.name+"_sig%.3f"%options.sig_injection
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

if options.full_supervision: name += "_fs" 

gpu_boole = torch.cuda.is_available()
print("Is GPU available? ",gpu_boole)
if load_model: print("Loading model... ")


if "Phi250" in name:
        options.sig = "2HDM-vbfPhiToTauTau-M250_2J_MinMass120_NoMisTag"
        options.m_tt_min = 100.
        options.m_tt_max = 500.
        if "ttbar" in name:
                options.bkg = "SM_ttbarTo2Tau2Nu_0J1J2J_MinMass120_NoMisTag_MadSpin_1M"
        if "DY" in name:
                options.bkg = "SM_dyToTauTau_0J1J2J_MinMass120_1M"
if "Phi750" in name:
        options.sig = "2HDM-vbfPhiToTauTau-M750_2J_MinMass350_NoMisTag"
        
        if "ttbar" in name:
                options.m_tt_min = 350.
                options.m_tt_max = 1200.
                options.bkg = "SM_ttbarTo2Tau2Nu_0J1J2J_MinMass350_NoMisTag_MadSpin_1M"
        if "DY" in name:
                options.m_tt_min = 400.
                options.m_tt_max = 1000.
                options.bkg = "SM_dyToTauTau_0J1J2J_MinMass350_NoMisTag_1M"
if "ttPhi750" in name:
        options.sig = "2HDM-ttPhiToTauTau-M750_2J_MinMass350_NoMisTag"
        
        
        if "ttbar" in name:
                options.m_tt_min = 350.
                options.m_tt_max = 1200.
                options.bkg = "SM_ttbarTo2Tau2Nu_0J1J2J_MinMass350_NoMisTag_MadSpin_1M"
        if "DY" in name:
                options.m_tt_min = 400.
                options.m_tt_max = 1000.
                options.bkg = "SM_dyToTauTau_0J1J2J_MinMass350_NoMisTag_1M"
if "TS250" in name:
        options.sig = "eVLQ_T-M1000_S-M250_NoMisTag"
        options.m_tt_min = 100.
        options.m_tt_max = 500.
        if "ttbar" in name:
                options.bkg = "SM_ttbarTo2Tau2Nu_0J1J2J_MinMass120_NoMisTag_MadSpin_1M"
        if "DY" in name:
                options.bkg = "SM_dyToTauTau_0J1J2J_MinMass120_1M"

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

print(options)

print(" Will use the following features: ", f_list)


if not options.test_ws:
        sig = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/%s.csv"%options.sig, lineterminator='\n')
        bkg1 = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/%s.csv"%options.bkg,lineterminator='\n')
        #bkg2 = pd.read_csv("~/nobackup/CATHODE_ditau/Delphes/diTauCathode/csv_files/SM_ttbarTo2Tau2Nu_2J_MinMass120_NoMisTag.csv",lineterminator='\n')

else:
        x1_sig, x2_sig = np.random.multivariate_normal([7,7], np.diag((0.5,0.5)), 50000).T
        y_sig = np.ones((50000))
        x1_bkg, x2_bkg = np.random.multivariate_normal([4,4], np.diag((4,4)), 100000).T
        y_bkg = np.zeros((100000))
        sig = np.vstack((x1_sig, x2_sig, y_sig)).T
        bkg1 = np.vstack((x1_bkg, x2_bkg, y_bkg)).T



if load_model:  loaded_epoch, losses, val_losses = load_trained_model(name, epoch_to_load)
else:
        loaded_epoch = 0
        losses,val_losses = [],[]

if options.BDT:
        from sklearn.ensemble import HistGradientBoostingClassifier,GradientBoostingClassifier
        from boosted_decision_tree import HGBClassifier
        from pickle import dump, load
        train, val, test, train_ws, val_ws, test_ws, feature_list = make_train_test_val_ws(options.test_ws, sig, bkg1, options.m_tt_min, options.m_tt_max, sig_injection, bkg_sig_frac, options.train_frac, options.val_frac, name, f_list)
        train, val, test = preprocess(train, val, test)
        train_ws, val_ws, test_ws = preprocess(train_ws, val_ws, test_ws)
        n_features = len(f_list)-2
        print("Using a HistGradientBoostingClassifier instead of NN...")
        if options.full_supervision:
                pred_list_all = []
                kf = KFold(n_splits = 10)
                train = np.vstack((train,val))
                name = options.name+"_sig%.3f"%options.sig_injection+"_fs"
                for i,(train_i,val_i) in enumerate(kf.split(train)):
                        train_kf, val_kf = train[train_i], train[val_i]
                        print(train_kf[:10])
                        print(val_kf[:10])
                        if not np.any(val_kf==0): print("THERE ARE NO BKG EVENTS IN THE VAL SET")
                        print(">> Training BDT with %ith fold as validation"%i)
                        bdt = HGBClassifier(max_iters=100, early_stopping=True)
                        bdt.fit(train_kf[:,:n_features],train_kf[:,n_features], val_kf[:,:n_features], val_kf[:,n_features])
                        pred_list = bdt.predict_proba(test[:,:n_features])[:,1]
                        pred_list_all.append(pred_list)
                pred_list_all = np.array(pred_list_all)
                pred_list = np.mean(pred_list_all, axis=0)
                true_list = test[:,-1]
                print("After averaging results of kfold, predicted list shape = ", pred_list.shape)
                np.savetxt("losses/fpr_tpr_bdt_%s_fs_kfold.txt"%(name.split("_")[0]),np.vstack((true_list,pred_list)))
                
        else:
                pred_list_all = []
                kf = KFold(n_splits = 10)
                train_ws = np.vstack((train_ws,val_ws))
                #name = options.name+"_sig%.3f"%options.sig_injection+"_fs"+"_train%.2f_val%.2f"%((0.8-val_frac), val_frac)
                name = options.name+"_sig%.3f"%options.sig_injection
                for i,(train_i,val_i) in enumerate(kf.split(train_ws)):
                        train_kf, val_kf = train_ws[train_i], train_ws[val_i]
                        print(train_kf[:10])
                        print(val_kf[:10])
                        if not np.any(val_kf==0): print("THERE ARE NO BKG EVENTS IN THE VAL SET")
                        print(">> Training with %ith fold as validation"%i)
                        bdt = HGBClassifier(max_iters=100, early_stopping=True)
                        bdt.fit(train_kf[:,:n_features],train_kf[:,n_features], val_kf[:,:n_features], val_kf[:,n_features])
                        pred_list = bdt.predict_proba(test_ws[:,:n_features])[:,1]
                        pred_list_all.append(pred_list)
                        
                pred_list_all = np.array(pred_list_all)
                pred_list = np.mean(pred_list_all, axis=0)
                true_list = test[:,-1]
                print("After averaging results of kfold, predicted list shape = ", pred_list.shape)
                np.savetxt("losses/fpr_tpr_bdt_%s_kfold.txt"%(name.split("_")[0]),np.vstack((true_list,pred_list)))

else:
        model = NN()
        if gpu_boole: model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(),
                        lr = 1e-3,
                        weight_decay = 1e-8)

        loss_function = torch.nn.BCELoss()
        
        if train_model:
                # train_frac_list = [0.1, 0.3, 0.5, 0.7]  
                # val_frac_list = train_frac_list[::-1]
                train_frac, val_frac = options.train_frac, options.val_frac
                name = options.name+ "_sig%.3f"%options.sig_injection+"_train%.2f_val%.2f"%(train_frac, val_frac) 
                train, val, test, train_ws, val_ws, test_ws, feature_list = make_train_test_val_ws(options.test_ws, sig, bkg1, options.m_tt_min, options.m_tt_max, sig_injection, bkg_sig_frac, train_frac, val_frac, name, f_list)
                train, val, test = preprocess(train, val, test)
                train_ws, val_ws, test_ws = preprocess(train_ws, val_ws, test_ws)
                train = np.vstack((train,val))
                train_ws = np.vstack((train_ws,val_ws))
                kf = KFold(n_splits=10)
                for i, (train_idx, val_idx) in enumerate(kf.split(train)):
                        print(">> Training NN with %ith fold as validation"%i)
                        train_kf, val_kf = train[train_idx], train[val_idx]
                        train_ws_kf, val_ws_kf = train_ws[train_idx], train_ws[val_idx]
                        
                        train_loader_ws, val_loader_ws, test_loader_ws = make_loaders(train_ws_kf,test_ws,val_ws_kf,batch_size)
                        train_loader, val_loader, test_loader = make_loaders(train_kf,test,val_kf,batch_size)
                        name = options.name+ "_fold%i"%i + "_sig%.3f"%options.sig_injection
                        if not options.full_supervision: training(train_loader_ws, val_loader_ws, losses, val_losses, loaded_epoch, name)      
                        else: training(train_loader, val_loader, losses, val_losses, loaded_epoch, name+"_fs") 
        
        if test_model:
                name = options.name
                if not train_model:
                        train, val, test, train_ws, val_ws, test_ws, feature_list = make_train_test_val_ws(options.test_ws, sig, bkg1, options.m_tt_min, options.m_tt_max, sig_injection, 
                        bkg_sig_frac, options.train_frac, options.val_frac, name, f_list) 
                        train, val, test = preprocess(train, val, test)
                        train_ws, val_ws, test_ws = preprocess(train_ws, val_ws, test_ws)
                        train_loader_ws, val_loader_ws, test_loader_ws = make_loaders(train_ws,test_ws,val_ws,batch_size)
                        train_loader, val_loader, test_loader = make_loaders(train,test,val,batch_size)
                
                if not options.full_supervision: testing(test_loader_ws, test, name, kfold = True)
                else: testing(test_loader, test, name, kfold = True)

'''
===========================
        EXTRAS
===========================
'''

if options.plot_pre_post:
        train, val, test, train_ws, val_ws, test_ws, feature_list = make_train_test_val_ws(options.test_ws, sig, bkg1, options.m_tt_min, options.m_tt_max, sig_injection, bkg_sig_frac, options.train_frac, options.val_frac, name, f_list)

        # plot data vs bkg for pre and post proc
        data_pre = train_ws[train_ws[:,2]==1]
        bkg_pre =  train_ws[train_ws[:,2]==0]
        sig_pre = train[train[:,2]==1]
        bkg_fs_pre = train[train[:,2]==0]

        train, val, test = preprocess(train, val, test)
        train_ws, val_ws, test_ws = preprocess(train_ws, val_ws, test_ws)

        data = train_ws[train_ws[:,2]==1]
        bkg =  train_ws[train_ws[:,2]==0]
        sig = train[train[:,2]==1]
        bkg_fs = train[train[:,2]==0]
        
        print("m_jj in data post proc (IAD):", data[:,0]) 
        print("Plotting %s pre and post processing: m_jj"%name)
        plt.hist(data_pre[:,0],label="Data before preprocessing",histtype='step')
        plt.hist(bkg_pre[:,0],label="Bkg before preprocessing",histtype='step')
        #plt.hist(data[:,0],label="Data after preprocessing",histtype='step')
        #plt.hist(bkg[:,0],label="Bkg after preprocessing",histtype='step')
        plt.xlim(0,4000)
        plt.xlabel("m_jj")
        plt.title("Distributions for IAD for %s"%name)
        plt.legend()
        plt.savefig("%s_m_jj_pre.png"%name)
        plt.close()

        plt.hist(data[:,0],label="Data after preprocessing",histtype='step')
        plt.hist(bkg[:,0],label="Bkg after preprocessing",histtype='step')
        plt.xlabel("m_jj")
        plt.title("Distributions for IAD for %s"%name)
        plt.legend()
        plt.savefig("%s_m_jj_post.png"%name)
        plt.close()

        print("Plotting %s pre and post processing: deltaR_jj"%name)
        plt.hist(data[:,1],label="Data after preprocessing",histtype='step')
        plt.hist(bkg[:,1],label="Bkg after preprocessing",histtype='step')
        plt.hist(data_pre[:,1],label="Data before preprocessing",histtype='step')
        plt.hist(bkg_pre[:,1],label="Bkg before preprocessing",histtype='step')
        plt.xlabel("deltaR_jj")
        plt.title("Distributions for IAD for %s"%name)
        plt.legend()
        plt.savefig("%s_deltaR_jj_pre_post.png"%name)
        plt.close()
        
        print("m_jj in sig post proc (FS):", sig[:,0])
        print("Plotting %s pre and post processing (FS): m_jj"%name)
        plt.hist(sig_pre[:,0],label="Signal before preprocessing",histtype='step')
        plt.hist(bkg_fs_pre[:,0],label="Bkg before preprocessing",histtype='step')
        #plt.hist(sig[:,0],label="Signal after preprocessing",histtype='step')
        #plt.hist(bkg_fs[:,0],label="Bkg after preprocessing",histtype='step')
        plt.xlim(0,4000)
        plt.xlabel("m_jj")
        plt.title("Distributions for FS for %s"%name)
        plt.legend()
        plt.savefig("%s_fs_m_jj_pre.png"%name)
        plt.close()

        plt.hist(sig[:,0],label="Signal after preprocessing",histtype='step')
        plt.hist(bkg_fs[:,0],label="Bkg after preprocessing",histtype='step')
        plt.xlabel("m_jj")
        plt.title("Distributions for FS for %s"%name)
        plt.legend()
        plt.savefig("%s_fs_m_jj_post.png"%name)
        plt.close()
        
        print("Plotting %s pre and post processing (FS): deltaR_jj"%name)
        plt.hist(sig[:,1],label="Signal after preprocessing",histtype='step')
        plt.hist(bkg_fs[:,1],label="Bkg after preprocessing",histtype='step')
        plt.hist(sig_pre[:,1],label="Signal before preprocessing",histtype='step')
        plt.hist(bkg_fs_pre[:,1],label="Bkg before preprocessing",histtype='step')
        plt.xlabel("deltaR_jj")
        plt.title("Distributions for FS for %s"%name)
        plt.legend()
        plt.savefig("%s_fs_deltaR_jj_pre_post.png"%name)
        plt.close()

n_features = len(feature_list[:-1])
print("n_features = ",n_features)
if options.feature_imp:
        if options.full_supervision: feature_select(train, name, feature_list, k = options.choose_n_features)
        else: feature_select(train_ws, name, feature_list, k = options.choose_n_features)

