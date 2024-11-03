# Plotter

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from sklearn.metrics import roc_curve, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

def plot_features(sig_labels,bkg_labels):


	# Format of csv file:
	# tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
	# tau2_m, m_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
	# jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef, isSig
    
    columns = ["m_jet1jet2", "deltaR_jet1jet2", "m_bjet1bjet2", "deltaR_bjet1bjet2", "deltaR_tau1tau2",
                    "tau1_pt", "tau1_eta", "tau1_phi", "tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m",
                    "m_tau1tau2", "pt_tau1tau2", "eta_tau1tau2", "phi_tau1tau2","met_met", "met_eta", "met_phi", "n_jets", "n_bjets",
                    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef",
                    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_cef", "jet2_nef", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef", "event_weight","label"]

    bkg = []
    for b in bkg_labels:
        bkg.append(pd.read_csv("csv_files/%s.csv" % b))
    for i in range(len(bkg)):
        bkg[i].columns = columns
        bkg[i] = bkg[i][["m_jet1jet2", "deltaR_jet1jet2","m_tau1tau2", "pt_tau1tau2","deltaR_tau1tau2","met_met","n_jets", "n_bjets", "event_weight"]]
    bkg[2]["event_weight"] = bkg[2]["event_weight"] * 0.0001 
    for sig__ in sig_labels:
        sig = pd.read_csv("csv_files/%s.csv" % sig__)

        sig.columns = ["m_jet1jet2", "deltaR_jet1jet2", "m_bjet1bjet2", "deltaR_bjet1bjet2", "deltaR_tau1tau2",
                    "tau1_pt", "tau1_eta", "tau1_phi", "tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m",
		            "m_tau1tau2", "pt_tau1tau2", "eta_tau1tau2", "phi_tau1tau2","met_met", "met_eta", "met_phi", "n_jets", "n_bjets",
			        "jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef",
	                "jet2_pt", "jet2_eta", "jet2_phi", "jet2_cef", "jet2_nef", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef", "event_weight","label"]
    
        

        sig = sig[["m_jet1jet2", "deltaR_jet1jet2","m_tau1tau2", "pt_tau1tau2","deltaR_tau1tau2","met_met","n_jets", "n_bjets", "event_weight"]]


        pp = PdfPages('plots/%s_DY_ttbar_QCD_distributions.pdf'%sig__)
        print("Plotting ", sig__)
        
        print(bkg[2]["event_weight"])
        for col in sig.columns:
            print(col)
            plt.figure(figsize=(10,7))
            #for b, l in zip(bkg, bkg_labels):
            if 'm_' in col: plt.hist([bkg[2][col],bkg[1][col],bkg[0][col]], label = bkg_labels, bins = 200,  density=True, histtype = "barstacked", weights = [bkg[2]["event_weight"],bkg[1]["event_weight"],bkg[0]["event_weight"]])
            else: plt.hist([bkg[2][col],bkg[1][col],bkg[0][col]], label = bkg_labels, bins = 50,  density=True, histtype = "barstacked", weights = [bkg[2]["event_weight"],bkg[1]["event_weight"],bkg[0]["event_weight"]])
            if col != "event_weight": 
                if 'm_' in col: plt.hist(sig[col], label = sig__, bins = 200, density=True, histtype = "step")
                else: plt.hist(sig[col], label = sig__, bins = 50, density=True, histtype = "step")
            plt.legend()
            plt.title("Distribution of %s"%col)
            plt.xlabel(col)
            plt.yscale('log')
            if 'm_' in col: plt.xlim(0,500)
            pp.savefig()
            plt.close()
	
        pp.close()

def plot_pre_postprocessed(train, val, test, train_ws, val_ws, test_ws):


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


def plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title):
	
	tpr_list, bkg_rej_list, bkg_eff_list, sic_list = [],[],[],[]
	# thresholds = np.linspace(0.25,1.8,40)
	for l,n in zip(ws_lists,ws_names):
		tpr, tnr, fpr = [],[],[]
		fpr, tpr, thresholds = roc_curve(l[0],l[1])
		# print("thresholds shape: ",thresholds.shape, "for ",n)
		# for thresh in thresholds:
		# 	pred = np.array(l[1]>thresh).astype(int)
		# 	tn, fp, fn, tp = confusion_matrix(l[0],pred).ravel()
		# 	tpr.append(tp/(tp+fn))
		# 	tnr.append(tn/(tn+fp))
		# 	fpr.append(fp/(tn+fp))
		
		# #print(tnr, fpr, tpr)
		# fpr = np.array(fpr)
		# tnr = np.array(tnr)
		# tpr = np.array(tpr)
		bkg_rej = 1 / (fpr+0.001)
		bkg_eff_list.append(fpr)
		sic = tpr / np.sqrt(fpr+0.001)
		tpr_list.append(tpr)
		bkg_rej_list.append(bkg_rej)
		sic_list.append(sic)

	for l,n in zip(fs_lists,fs_names):
		tpr, tnr, fpr = [],[],[]
		fpr, tpr, thresholds = roc_curve(l[0],l[1])
		# print("thresholds shape: ",thresholds.shape, "for ",n)
		# for thresh in thresholds:
		# 	pred = np.array(l[1]>thresh).astype(int)
		# 	tn, fp, fn, tp = confusion_matrix(l[0],pred).ravel()
		# 	tpr.append(tp/(tp+fn))
		# 	tnr.append(tn/(tn+fp))
		# 	fpr.append(fp/(tn+fp))
		
		# #print(tnr, fpr, tpr)
		# fpr = np.array(fpr)
		# tnr = np.array(tnr)
		# tpr = np.array(tpr)
		bkg_rej = 1 / (fpr+0.001)
		bkg_eff_list.append(fpr)
		sic = tpr / np.sqrt(fpr+0.001)
		tpr_list.append(tpr)
		bkg_rej_list.append(bkg_rej)
		sic_list.append(sic)

	names = ws_names + fs_names

	random_tpr = np.linspace(0, 1, len(tpr))
	random_bkg_rej = 1 / (random_tpr+0.001)
	random_sic = random_tpr / np.sqrt(random_tpr+0.001)

	# ROC curve
	plt.figure(figsize=(8,8))
	for i in range(len(names)):
		print("Plotting ",names[i])
		plt.plot(tpr_list[i], bkg_rej_list[i], label=names[i])
	plt.plot(random_tpr, random_bkg_rej, label="random")
	plt.xlabel("Signal Efficiency (True Positive Rate)")
	plt.ylabel("Background Rejection")
	plt.yscale("log")
	plt.legend()
	plt.title(plt_title)
	plt.savefig("plots/ROC_%s.png"%plt_title)
	plt.close()

	# SIC curve
	plt.figure(figsize=(8,8))
	for i in range(len(names)):
		print("Plotting ",names[i])
		plt.plot(bkg_eff_list[i], sic_list[i], label=names[i])
	plt.plot(random_tpr, random_sic, label="random")
	plt.xlabel("Background Efficiency (False Positive Rate)")
	plt.ylabel("Significance Improvement")
	plt.legend()
	plt.title(plt_title)
	plt.savefig("plots/SIC_%s.png"%plt_title)
	plt.close()

sig_list = ["2HDM-vbfPhiToTauTau-M250_2J_MinMass120_NoMisTag",
            "eVLQ_TPrimeTPrimeToTTPhiPhiToTauTauAll_TpM1000_PhiM250_NoMisTag",
            "HeavyN_vbsNToTauTau_NM250_2J_LO", 
            "VAL_dyVfVfToXiCXiCToTauSTauS_XiM1000_VfM250_MinMass120_NoMisTag"]

bkg_list = ["SM_dyToTauTau_0J1J2J_MinMass120_3M", "SM_ttbarTo2Tau2Nu_0J1J2J_MinMass120_MadSpin_2M", "SM_QCD_JJ_0J1J2J_MinMass120_LO_6M"]
'''
bkg = []

for b in bkg_list:
	bkg.append(pd.read_csv("csv_files/%s.csv"%b))


for sig in sig_list:
	sig__ = pd.read_csv("csv_files/%s.csv"%sig)
'''
plot_features(sig_list, bkg_list)

#injections = ["0.100","0.050","0.010","0.005"]
#injections = ["0.100"]#,"0.200","0.300","0.400","0.500","0.600","0.700","0.800","0.900"]
'''
sigmas = [2,2.4,3,4,5]
sigs = ["Phi250",'VAL','HNL',"TS250"]
# losses/fpr_tpr_bdt_VAL_sigma3.0_N50.txt
# losses/fpr_tpr_bdt_VAL_sigma2.0_fs_fs_N50.txt
for sig in sigs:
	ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
	for sigma in sigmas:
		ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_%s_sigma%.1f_N50.txt"%(sig,sigma)))
		ws_names.append(r"BDT IAD with $S/\sqrt{B}$=%.1f"%sigma)
		fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_%s_sigma%.1f_fs_fs_N50.txt"%(sig, sigma)))
		fs_names.append(r"BDT Full Sup with $S/\sqrt{B}$=%.1f"%sigma)
	
	plt_title = "%s_BDT_sigma_scan"%(sig) 
	plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)

for mass in masses:
	for bkg in bkgs:
		ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
		
		# fpr_tpr_ttPhi750vsDY_fs_sig0.010.txt	
		# fpr_tpr_bdt_ttPhi750vsDY_fs_kfold.txt
		ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s.txt"%(mass,bkg)))
		ws_names.append("NN IAD with 10-fold cross val (1%% signal)")
		fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_fs.txt"%(mass,bkg)))
		fs_names.append("NN Full Sup with 10-fold cross val (1%% signal)")
		# ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_0.3val_N50.txt"%(mass,bkg)))
		# ws_names.append("BDT IAD with N=50 ensembles (1%% signal)")
		# fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_fs_0.3val_N50.txt"%(mass,bkg)))
		# fs_names.append("BDT Full Sup with N=50 ensembles (1%% signal)")
		# ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_sig0.010_N50.txt"%(mass,bkg)))
		# ws_names.append("BDT IAD with N=50 ensembles with PowerTransformer (1%% signal)")
		# fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_sig0.010_fs_fs_N50.txt"%(mass,bkg)))
		# fs_names.append("BDT Full Sup with N=50 ensembles with PowerTransformer (1%% signal)")
		ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_yeojohn_skl_sig0.010_N50.txt"%(mass,bkg)))
		ws_names.append("BDT IAD with N=50 ensembles with PowerTransformer (1%% signal)")
		fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_yeojohn_skl_sig0.010_fs_fs_N50.txt"%(mass,bkg)))
		fs_names.append("BDT Full Sup with N=50 ensembles with PowerTransformer (1%% signal)")
		# ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_skl_10f_sig0.010_N50.txt"%(mass,bkg)))
		# ws_names.append("BDT IAD with N=50 ensembles using sklearn with 10 feats(1%% signal)")
		# fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_skl_10f_sig0.010_fs_fs_N50.txt"%(mass,bkg)))
		# fs_names.append("BDT Full Sup with N=50 ensembles using sklearn with 10 feats(1%% signal)")
		# ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_rf_sig0.010_N50.txt"%(mass,bkg)))
		# ws_names.append("BDT IAD with N=10 ensembles RandomForests with 10 feats(1%% signal)")
		# fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_rf_sig0.010_fs_fs_N50.txt"%(mass,bkg)))
		# fs_names.append("BDT Full Sup with N=10 ensembles RandomForests with 10 feats(1%% signal)")
		# ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_sig0.900_N50.txt"%(mass,bkg)))
		# ws_names.append("BDT IAD with N=50 ensembles (90%% signal)")
		# fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_sig0.900_fs_fs_N50.txt"%(mass,bkg)))
		# fs_names.append("BDT Full Sup with N=50 ensembles (90%% signal)")
		plt_title = "Phi%ivs%s_BDT_hgb_skl_sig0.010"%(mass,bkg) 
		plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)
'''
