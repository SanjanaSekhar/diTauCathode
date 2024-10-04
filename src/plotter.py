# Plotter

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from sklearn.metrics import roc_curve
from matplotlib.backends.backend_pdf import PdfPages

def plot_features(sig,bkg1,bkg2):


	# Format of csv file:
	# tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
	# tau2_m, m_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
	# jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef, isSig
    print(sig.shape, bkg1.shape, bkg2.shape)
	
    sig.columns = ["m_jet1jet2", "deltaR_jet1jet2", "m_bjet1bjet2", "deltaR_bjet1bjet2", "deltaR_tau1tau2",
                    "tau1_pt", "tau1_eta", "tau1_phi", "tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m",
		            "m_tau1tau2", "pt_tau1tau2", "eta_tau1tau2", "phi_tau1tau2","met_met", "met_eta", "met_phi", "n_jets", "n_bjets",
			        "jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef",
	                "jet2_pt", "jet2_eta", "jet2_phi", "jet2_cef", "jet2_nef", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef", "label"]
    bkg1.columns = sig.columns
    bkg2.columns = sig.columns
    pp = PdfPages('plots/TS250_ttbar_DY_features.pdf')

    for col in sig.columns:

	    plt.figure(figsize=(10,7))
	    plt.hist(bkg1[col], label = "DY + 0/1/2 jets", bins = 30, histtype = "step")
	    plt.hist(bkg2[col], label = "ttbar + 2 jets", bins = 30, histtype = "step")
	    plt.hist(sig[col], label = "T'(1000) + S (250) + 2 jets", bins = 30, histtype = "step")
	    plt.legend()
	    plt.title("Distribution of %s"%col)
	    plt.xlabel(col)
	    plt.yscale('log')
	    #plt.ylim(np.amin([sig[col].min(),bkg1[col].min(),bkg2[col].min()]), np.amax([sig[col].max(),bkg1[col].max(),bkg2[col].max()])*10)
	    plt.ylabel("No. of di-tau events (hadronic)")
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
	
	tpr_list, bkg_rej_list, sic_list = [],[],[]

	for l,n in zip(ws_lists,ws_names):
		fpr, tpr, _ = roc_curve(l[0], l[1])
		#print(n, fpr, tpr)
		bkg_rej = 1 / (fpr+0.001)
		sic = tpr / np.sqrt(fpr+0.001)
		tpr_list.append(tpr)
		bkg_rej_list.append(bkg_rej)
		sic_list.append(sic)

	for l,n in zip(fs_lists,fs_names):
		fpr, tpr, _ = roc_curve(l[0], l[1])
		#print(n, fpr, tpr)
		bkg_rej = 1 / (fpr+0.001)
		sic = tpr / np.sqrt(fpr+0.001)
		tpr_list.append(tpr)
		bkg_rej_list.append(bkg_rej)
		sic_list.append(sic)

	names = ws_names + fs_names

	random_tpr = np.linspace(0, 1, len(fpr))
	random_bkg_rej = 1 / (random_tpr+0.001)
	random_sic = random_tpr / np.sqrt(random_tpr+0.001)

	# ROC curve
	plt.figure(figsize=(8,8))
	for i in range(len(names)):
		print("Plotting ",names[i])
		plt.plot(tpr_list[i], bkg_rej_list[i], label=names[i])
	plt.plot(random_tpr, random_bkg_rej, label="random")
	plt.xlabel("True Positive Rate")
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
		plt.plot(tpr_list[i], sic_list[i], label=names[i])
	plt.plot(random_tpr, random_sic, label="random")
	plt.xlabel("True Positive Rate")
	plt.ylabel("Significance Improvement")
	plt.legend(loc="upper right")
	plt.title(plt_title)
	plt.savefig("plots/SIC_%s.png"%plt_title)
	plt.close()


#sig = pd.read_csv("csv_files/2HDM-vbfPhiToTauTau-M750_2J_MinMass350_NoMisTag.csv", lineterminator='\n')
#sig = pd.read_csv("csv_files/2HDM-TSToTauTau-M750_2J_MinMass350_NoMisTag.csv")
# sig = pd.read_csv("csv_files/eVLQ_T-M1000_S-M250_NoMisTag.csv")
# bkg1 = pd.read_csv("csv_files/SM_dyToTauTau_0J1J2J_MinMass120_1M.csv")
# bkg2 = pd.read_csv("csv_files/SM_ttbarTo2Tau2Nu_0J1J2J_MinMass120_NoMisTag_MadSpin_1M.csv")

#plot_features(sig, bkg1, bkg2)

#injections = ["0.100","0.050","0.010","0.005"]
#injections = ["0.100"]#,"0.200","0.300","0.400","0.500","0.600","0.700","0.800","0.900"]
masses = [250]
bkgs = ["DY"]


# name: losses/fpr_tpr_TS750vsttbar-case3_fs_sig0.010.txt

# for mass in masses:
# 	for bkg in bkgs:
# 		ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
		
# 		ws_lists.append(np.loadtxt("losses/fpr_tpr_TS%ivs%s-case1_sig0.010.txt"%(mass,bkg)))
# 		ws_names.append(r"IAD: $m_{\tau 1}, m_{\tau 2}, \Delta R_{\tau\tau}$, MET")
# 		fs_lists.append(np.loadtxt("losses/fpr_tpr_TS%ivs%s-case1_fs_sig0.010.txt"%(mass,bkg)))
# 		fs_names.append(r"FS: $m_{\tau 1}, m_{\tau 2}, \Delta R_{\tau\tau}$, MET")

# 		ws_lists.append(np.loadtxt("losses/fpr_tpr_TS%ivs%s-case2_sig0.010.txt"%(mass,bkg)))
# 		ws_names.append(r"IAD: $m_{jj}, \Delta R_{jj}, \Delta R_{\tau\tau}$, MET")
# 		fs_lists.append(np.loadtxt("losses/fpr_tpr_TS%ivs%s-case2_fs_sig0.010.txt"%(mass,bkg)))
# 		fs_names.append(r"FS: $m_{jj}, \Delta R_{jj}, \Delta R_{\tau\tau}$, MET")

# 		ws_lists.append(np.loadtxt("losses/fpr_tpr_TS%ivs%s-case3_sig0.010.txt"%(mass,bkg)))
# 		ws_names.append(r"IAD: $m_{jj}, \Delta R_{jj}, m_{\tau 1}, m_{\tau 2}$")
# 		fs_lists.append(np.loadtxt("losses/fpr_tpr_TS%ivs%s-case3_fs_sig0.010.txt"%(mass,bkg)))
# 		fs_names.append(r"FS: $m_{jj}, \Delta R_{jj}, m_{\tau 1}, m_{\tau 2}$")

# 		ws_lists.append(np.loadtxt("losses/fpr_tpr_TS%ivs%s-case4_sig0.010.txt"%(mass,bkg)))
# 		ws_names.append(r"IAD: $m_{jj}, \Delta R_{jj}, m_{\tau 1}, \Delta R_{\tau\tau}$")
# 		fs_lists.append(np.loadtxt("losses/fpr_tpr_TS%ivs%s-case4_fs_sig0.010.txt"%(mass,bkg)))
# 		fs_names.append(r"FS: $m_{jj}, \Delta R_{jj}, m_{\tau 1}, \Delta R_{\tau\tau}$")

# 		plt_title = "TS%ivs%s_cases_sig0.01"%(mass,bkg) 
# 		plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)

#  create mode 100644 losses/fpr_tpr_bdt_Phi250vsDY_sig0.010_N50.txt
#  create mode 100644 losses/fpr_tpr_bdt_Phi250vsDY_sig0.010_fs_fs_N50.txt
#  create mode 100644 losses/fpr_tpr_bdt_Phi250vsDY_sig0.050_N50.txt
#  create mode 100644 losses/fpr_tpr_bdt_Phi250vsDY_sig0.050_fs_fs_N50.txt

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
