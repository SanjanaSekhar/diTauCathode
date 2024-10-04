
import time
from time import sleep
from tqdm import tqdm
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import torch
torch.cuda.empty_cache()
import gc
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable


from sklearn.metrics import roc_curve
from argparse import ArgumentParser

sys.setrecursionlimit(10000)
# NN

class NN(torch.nn.Module):
        def __init__(self):
                super().__init__()

                self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(6,32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32,32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32,16),
                        torch.nn.ReLU(),
                        torch.nn.Linear(16,1),
                        torch.nn.Sigmoid()
                        )          
                         
        def forward(self, x):
                label = self.classifier(x)
                return label

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

def train_NN(train_loader,val_loader,losses,val_losses,loaded_epoch,name):

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

def test_NN(test_loader_ws, test_true, name, sigma, kfold=False):

        if kfold:
                pred_list_all = []
                for i in range(10):
                        pth = "%s_fold%i_sigma%0.1f"%(name.split("_")[0],i,sigma)
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
                if options.full_supervision: np.savetxt("losses/fpr_tpr_nn_%s_fs.txt"%(name), np.vstack((true_list,pred_list)))
                else: np.savetxt("losses/fpr_tpr_nn_%s.txt"%(name), np.vstack((true_list,pred_list)))

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
        
