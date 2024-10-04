import math
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, chi2


def make_sig_ws_fs(test_ws, sig, m_tt_min = 100., m_tt_max = 500., n_sig_bkg1 = 500, train_frac = 0.8, val_frac = 0.1, name = "PhivsDY", f_list = ["m_jet1jet2", "deltaR_jet1jet2"]):

        # CREATE an IDEAL Anomaly Detector
        # Train pure bkg vs sig+bkg in the SR only

        if test_ws:
                sig_sigregion = pd.DataFrame(sig)
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
                
                # deltaR of taus
                #sig["deltaR_taus"] = ((sig["tau1_eta"]-sig["tau2_eta"]).pow(2) + (sig["tau1_phi"]-sig["tau2_phi"]).pow(2)).pow(0.5)
                #bkg1["deltaR_taus"] = ((bkg1["tau1_eta"]-bkg1["tau2_eta"]).pow(2) + (bkg1["tau1_phi"]-bkg1["tau2_phi"]).pow(2)).pow(0.5)
                
                #sig.drop(labels=["tau1_pt", "tau2_pt", "tau1_m","tau2_m", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef"], axis=1, inplace=True)
                #bkg1.drop(labels=["tau1_pt", "tau2_pt", "tau1_m","tau2_m", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef"], axis=1, inplace=True)
                if f_list:
                    sig = sig[f_list]
                
                print("Signal shape: ", sig.shape)
                print("Min, max m_tt in sig: ", sig['m_tau1tau2'].min(), sig['m_tau1tau2'].max() )

                # Choose the m_tautau window in which to define 'data' and bkg regions
                # The max ditau inv mass is not the same in all samples


                # define "data" and "bkg" regions
                sig_sigregion = sig[sig['m_tau1tau2'] >= m_tt_min] #and sig[sig['m_tau1tau2'] <= m_tt_max]
                sig_sigregion = sig_sigregion[sig_sigregion['m_tau1tau2'] < m_tt_max]

        
                print("No. of samples in SR in sig")
                print(sig_sigregion.shape[0])

                # shuffle the background indices
                bkg1_sigregion = bkg1_sigregion.sample(frac=1).reset_index(drop=True)


        sig_bkg1_idxs = np.random.choice(range(0,sig_sigregion.shape[0]),size=n_sig_bkg1)
        #print(sig_bkg1_idxs, sig_bkg2_idxs)
        sig_to_inject_bkg1 = sig_sigregion.loc[sig_sigregion.index[sig_bkg1_idxs]]
        
        print("Shape of signal in data: ",sig_to_inject_bkg1.shape)


        # sig_to_inject_bkg1 and sig_to_inject_bkg1_ws both have label = 1
        sig_to_inject_bkg1_ws = sig_to_inject_bkg1.copy()
        if not test_ws: sig_to_inject_bkg1 = sig_to_inject_bkg1.drop(['m_tau1tau2'],axis=1)
        feature_list = sig_to_inject_bkg1.columns
        sig_to_inject_bkg1 = sig_to_inject_bkg1.to_numpy()
        if not test_ws:
                sig_to_inject_bkg1_ws = sig_to_inject_bkg1_ws.drop(['m_tau1tau2'],axis=1)
                
        
        sig_to_inject_bkg1_ws = sig_to_inject_bkg1_ws.to_numpy()
        

        # train val test split: train_frac, train_frac + val_frac, 1-(train_frac + val_frac)
        train_sig, val_sig, test_sig = np.split(sig_to_inject_bkg1, [int(train_frac*len(sig_to_inject_bkg1)), int((train_frac + val_frac)*len(sig_to_inject_bkg1))])
        train_sig_ws, val_sig_ws, test_sig_ws = np.split(sig_to_inject_bkg1_ws, [int(train_frac*len(sig_to_inject_bkg1_ws)), int((train_frac + val_frac)*len(sig_to_inject_bkg1_ws))])
        #print(train_bkg1_ws) 
        print("train_sig.shape = ",train_sig.shape)
        
        return train_sig, val_sig, test_sig, train_sig_ws, val_sig_ws, test_sig_ws, feature_list.to_list()

def make_bkg_ws_fs(test_ws, bkg1, m_tt_min = 100., m_tt_max = 500., n_bkg1 = -1, train_frac = 0.8, val_frac = 0.1, name = "PhivsDY", f_list = ["m_jet1jet2", "deltaR_jet1jet2"]):

        # CREATE an IDEAL Anomaly Detector
        # Train pure bkg vs sig+bkg in the SR only

        if test_ws:
                bkg1_sigregion = pd.DataFrame(bkg1)
        else:
                # Format of csv file:
                # tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
                # tau2_m, m_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
                # jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef,
                # jet2_pt, jet2_eta, jet2_phi, jet2_cef, jet2_nef, bjet2_pt, bjet2_eta, bjet2_phi, bjet2_cef, bjet2_nef, isSig

                print(sig.shape, bkg1.shape)
                bkg1.columns = [ "m_jet1jet2", "deltaR_jet1jet2", "m_bjet1bjet2", "deltaR_bjet1bjet2", "deltaR_tau1tau2",
                                "tau1_pt", "tau1_eta", "tau1_phi", "tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m",
                                "m_tau1tau2","pt_tau1tau2", "eta_tau1tau2", "phi_tau1tau2", "met_met", "met_eta", "met_phi", "n_jets", "n_bjets",
                                "jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef",
                                "jet2_pt", "jet2_eta", "jet2_phi", "jet2_cef", "jet2_nef", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef", "label"]
                
                # deltaR of taus
                #sig["deltaR_taus"] = ((sig["tau1_eta"]-sig["tau2_eta"]).pow(2) + (sig["tau1_phi"]-sig["tau2_phi"]).pow(2)).pow(0.5)
                #bkg1["deltaR_taus"] = ((bkg1["tau1_eta"]-bkg1["tau2_eta"]).pow(2) + (bkg1["tau1_phi"]-bkg1["tau2_phi"]).pow(2)).pow(0.5)
                
                #sig.drop(labels=["tau1_pt", "tau2_pt", "tau1_m","tau2_m", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef"], axis=1, inplace=True)
                #bkg1.drop(labels=["tau1_pt", "tau2_pt", "tau1_m","tau2_m", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef"], axis=1, inplace=True)
                if f_list:
                    bkg1 = bkg1[f_list]
                
                print("Bkg shape: ", bkg1.shape)
                print("Min, max m_tt in bkg1: ", bkg1['m_tau1tau2'].min(), bkg1['m_tau1tau2'].max())

                # Choose the m_tautau window in which to define 'data' and bkg regions
                # The max ditau inv mass is not the same in all samples


                # define "data" and "bkg" regions
               
                bkg1_sigregion = bkg1[bkg1['m_tau1tau2'] >=  m_tt_min] #and bkg1[bkg1['m_tau1tau2'] <= m_tt_max]
                bkg1_sigregion = bkg1_sigregion[bkg1_sigregion['m_tau1tau2'] < m_tt_max]
                print("No. of samples in SR in bkg")
                print(bkg1_sigregion.shape[0])

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
        if n_bkg1 > 0:
            bkg1_idxs = np.random.choice(range(0,bkg1_sigregion.shape[0]),size=n_bkg1)
            bkg_to_inject_bkg1 = bkg1_sigregion.loc[bkg1_sigregion.index[bkg1_idxs]]
            bkg1_sigregion = bkg_to_inject_bkg1.copy()
            
            print("No of bkg samples in data = ",n_bkg1)
        print("Shape of bkg in data: ", bkg1_sigregion.shape)
        n_events = bkg1_sigregion.shape[0]


        # define data and bkg vectors
        # label data as 1 and pure bkg as 0
        # bkg1_sigregion has label = 0 in the data region, bkg1_sigregion_ws has label = 1
        bkg1_sigregion_ws = bkg1_sigregion.copy()
        if not test_ws:
                print("SETTING LABEL OF BKG TO 1") 
                bkg1_sigregion_ws.loc[:,'label'] = 1
        else: bkg1_sigregion_ws[bkg1_sigregion_ws.columns[2]] = 1

    
        if not test_ws:
                bkg1_sigregion_ws = bkg1_sigregion_ws.drop(['m_tau1tau2'],axis=1)
                bkg1_sigregion = bkg1_sigregion.drop(['m_tau1tau2'],axis=1)
        
        bkg1_sigregion_ws = bkg1_sigregion_ws.to_numpy()
        bkg1_sigregion = bkg1_sigregion.to_numpy()
        
        # train val test split: train_frac, train_frac + val_frac, 1-(train_frac + val_frac)
        train_bkg1_data, val_bkg1_data, test_bkg1_data = np.split(bkg1_sigregion, [int(train_frac*len(bkg1_sigregion)), int((train_frac + val_frac)*len(bkg1_sigregion))])
        train_bkg1_data_ws, val_bkg1_data_ws, test_bkg1_data_ws = np.split(bkg1_sigregion_ws, [int(train_frac*len(bkg1_sigregion_ws)), int((train_frac + val_frac)*len(bkg1_sigregion_ws))])
        #print(train_bkg1_ws) 
        print("train_bkg1.shape, train_bkg1_ws.shape = ",train_bkg1_data.shape, train_bkg1_data_ws.shape)
        
        # both bkg1_bkgregion and bkg1_bkgregion_ws have label = 0
        bkg1_bkgregion_ws = bkg1_bkgregion.copy()
        bkg1_bkgregion_ws = bkg1_bkgregion_ws.drop(['m_tau1tau2'],axis=1)
        bkg1_bkgregion_ws = bkg1_bkgregion_ws.to_numpy()

        train_bkg1, val_bkg1, test_bkg1 = np.split(bkg1_bkgregion_ws, [int(train_frac*len(bkg1_bkgregion_ws)), int((train_frac + val_frac)*len(bkg1_bkgregion_ws))])
        # sets with all true labels for full supervision and ROC curve
        train_bkg = np.vstack((train_bkg1_data,train_bkg1))
        val_bkg = np.vstack((val_bkg1_data,val_bkg1))
        test_bkg = np.vstack((test_bkg1_data,test_bkg1))
        # sets with label = 0 for pure bkg and 1 for data
        train_bkg_ws = np.vstack((train_bkg1_data_ws,train_bkg1))
        val_bkg_ws = np.vstack((val_bkg1_data_ws,val_bkg1))
        test_bkg_ws = np.vstack((test_bkg1_data_ws,test_bkg1))
        
        return train_bkg, val_bkg, test_bkg, train_bkg_ws, val_bkg_ws, test_bkg_ws, n_events

def preprocess(train, val, test):
        print(train.shape)
        n_features = train.shape[1] - 1
        #scaler = StandardScaler()
        #scaler = MinMaxScaler()
        scaler = PowerTransformer()
        scaler.fit(np.vstack((train,val))[:,0:n_features])
        train[:,0:n_features] = scaler.transform(train[:,0:n_features])
        test[:,0:n_features] = scaler.transform(test[:,0:n_features])
        val[:,0:n_features] = scaler.transform(val[:,0:n_features])
        #print("After preprocessing, mean, std for the features:", scaler.mean_,scaler.var_)
        return train, val, test



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