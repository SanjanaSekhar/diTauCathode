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
masses = [750]
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


for mass in masses:
	for bkg in bkgs:
		ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
		
		# fpr_tpr_ttPhi750vsDY_fs_sig0.010.txt	
		# fpr_tpr_bdt_ttPhi750vsDY_fs_kfold.txt
		ws_lists.append(np.loadtxt("losses/fpr_tpr_ttPhi%ivs%s_sig0.010.txt"%(mass,bkg)))
		ws_names.append("NN IAD with 10-fold cross val (1%% signal)")
		fs_lists.append(np.loadtxt("losses/fpr_tpr_ttPhi%ivs%s_fs_sig0.010.txt"%(mass,bkg)))
		fs_names.append("NN Full Sup with 10-fold cross val (1%% signal)")
		ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_ttPhi%ivs%s_kfold.txt"%(mass,bkg)))
		ws_names.append("BDT IAD with 10-fold cross val (1%% signal)")
		fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_ttPhi%ivs%s_fs_kfold.txt"%(mass,bkg)))
		fs_names.append("BDT Full Sup with 10-fold cross val (1%% signal)")
		plt_title = "Phi%ivs%s_sig0.01_10fold_NNvsBDT"%(mass,bkg) 
		plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)

'''
for mass in sig_masses:
	for bkg in bkgs:
		ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
		for inj in injections:
			for tr in train_frac:
				ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_sig%s_trainfrac%s.txt"%(mass,bkg,inj,tr)))
				ws_names.append("NN IAD: %.2f%% signal, %.2f%% train fraction"%(float(inj)*100,float(tr)*100))
				#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_sig%s_fs_trainfrac%s.txt"%(mass,bkg,inj,tr)))
				#fs_names.append("NN Full Sup: %.2f%% signal, %.2f%% train fraction"%(float(inj)*100,float(tr)*100))
				# ws_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_sig%s.txt"%(mass,bkg,inj)))
				# ws_names.append("BDT IAD: %.2f%% signal"%(float(inj)*100))
				# fs_lists.append(np.loadtxt("losses/fpr_tpr_bdt_Phi%ivs%s_sig%s_fs.txt"%(mass,bkg,inj)))
				# fs_names.append("BDT Full Sup: %.2f%% signal"%(float(inj)*100))
				plt_title = "Phi%ivs%s_mjj_deltaRjj_train-frac-cmps_WeakSup"%(mass,bkg) 
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_mjj_deltaRjj_sig%s.txt"%(mass,bkg,inj)))
			ws_names.append("NN IAD: %.2f%% signal, 80%% train fraction"%(float(inj)*100))
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_mjj_deltaRjj_sig%s.txt"%(mass,bkg,inj)))
			#fs_names.append("NN Full Sup: %.2f%% signal, 80%% train fraction"%(float(inj)*100))
			# ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_sig0.000.txt"%(mass,bkg)))
			# ws_names.append("Bkg vs Bkg: 0%% signal")
			#print(ws_names)
			plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)


for mass in sig_masses:
	for bkg in bkgs:
		ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
		for inj in injections:
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_type2_sig%s.txt"%(mass,bkg,inj))) # 8 features
			ws_names.append("NN IAD: 8 features")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_type2_sig%s_fs.txt"%(mass,bkg,inj))) # 8 features
			fs_names.append("NN Full Sup: 8 features")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_4feats_sig%s.txt"%(mass,bkg,inj))) # 4 features
			ws_names.append("NN IAD: 4 features")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_4feats_sig%s_fs.txt"%(mass,bkg,inj))) # 4 features
			fs_names.append("NN Full Sup: 4 features")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_all_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: ALL(41) features")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_all_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: ALL(41) features")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_mjj_deltaRjj_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: m_jj deltaR_jj")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_mjj_deltaRjj_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: m_jj deltaR_jj")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_j1cef_j1nef_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: jet1_cef jet1_nef")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_j1cef_j1nef_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: jet1_cef jet1_nef")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_deltaRtt_met_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: deltaR_tautau MET")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_deltaRtt_met_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: deltaR_tautau MET")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_pttt_deltaRjj_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: pT_tautau deltaR_jj")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_pttt_deltaRjj_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: pT_tautau deltaR_jj")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_m1m2t_deltaRtt_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: m_tau1 m_tau2 deltaR_tautau")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_m1m2t_deltaRtt_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: m_tau1 m_tau2 deltaR_tautau")
			plt_title = "Phi%ivs%s_Feature-Comparison"%(mass,bkg) 
		# ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_sig0.000.txt"%(mass,bkg)))
		# ws_names.append("Bkg vs Bkg: 0%% signal")
		#print(ws_names)
		plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)

for mass in sig_masses:
	for bkg in bkgs:
		ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
		for inj in injections:
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_type2_sig%s.txt"%(mass,bkg,inj))) # 8 features
			ws_names.append("NN IAD: 8 features")
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_sig%s_fs.txt"%(mass,bkg,inj))) # 8 features
			#fs_names.append("NN Full Sup: 8 features")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_4feats_sig%s.txt"%(mass,bkg,inj))) # 4 features
			ws_names.append("NN IAD: 4 features")
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_4feats_sig%s_fs.txt"%(mass,bkg,inj))) # 4 features
			#fs_names.append("NN Full Sup: 4 features")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_all_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: ALL(41) features")
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_all_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			#fs_names.append("NN Full Sup: ALL(41) features")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_mjj_deltaRjj_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: m_jj deltaR_jj")
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_mjj_deltaRjj_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			#fs_names.append("NN Full Sup: m_jj deltaR_jj")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_j1cef_j1nef_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: jet1_cef jet1_nef")
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_j1cef_j1nef_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			#fs_names.append("NN Full Sup: jet1_cef jet1_nef")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_deltaRtt_met_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: deltaR_tautau MET")
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_deltaRtt_met_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			#fs_names.append("NN Full Sup: deltaR_tautau MET")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_pttt_deltaRjj_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: pT_tautau deltaR_jj")
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_pttt_deltaRjj_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			#fs_names.append("NN Full Sup: pT_tautau deltaR_jj")
			ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_m1m2t_deltaRtt_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			ws_names.append("NN IAD: m_tau1 m_tau2 deltaR_tautau")
			#fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_m1m2t_deltaRtt_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			#fs_names.append("NN Full Sup: m_tau1 m_tau2 deltaR_tautau")
			plt_title = "Phi%ivs%s_Feature-Comparison_Weak-Supervision"%(mass,bkg) 
		# ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_sig0.000.txt"%(mass,bkg)))
		# ws_names.append("Bkg vs Bkg: 0%% signal")
		#print(ws_names)
		plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)

for mass in sig_masses:
	for bkg in bkgs:
		ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
		for inj in injections:
			#ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_sig%s.txt"%(mass,bkg,inj))) # 8 features
			#ws_names.append("NN IAD: 8 features")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_type2_sig%s_fs.txt"%(mass,bkg,inj))) # 8 features
			fs_names.append("NN Full Sup: 8 features")
			#ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_4feats_sig%s.txt"%(mass,bkg,inj))) # 4 features
			#ws_names.append("NN IAD: 4 features")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_4feats_sig%s_fs.txt"%(mass,bkg,inj))) # 4 features
			fs_names.append("NN Full Sup: 4 features")
			#ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_all_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			#ws_names.append("NN IAD: ALL(41) features")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_all_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: ALL(41) features")
			#ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_mjj_deltaRjj_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			#ws_names.append("NN IAD: m_jj deltaR_jj")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_mjj_deltaRjj_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: m_jj deltaR_jj")
			#ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_j1cef_j1nef_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			#ws_names.append("NN IAD: jet1_cef jet1_nef")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_j1cef_j1nef_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: jet1_cef jet1_nef")
			#ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_deltaRtt_met_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			#ws_names.append("NN IAD: deltaR_tautau MET")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_deltaRtt_met_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: deltaR_tautau MET")
			#ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_pttt_deltaRjj_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			#ws_names.append("NN IAD: pT_tautau deltaR_jj")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_pttt_deltaRjj_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: pT_tautau deltaR_jj")
			#ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_m1m2t_deltaRtt_sig%s.txt"%(mass,bkg,inj))) # all 41 features
			#ws_names.append("NN IAD: m_tau1 m_tau2 deltaR_tautau")
			fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_m1m2t_deltaRtt_sig%s_fs.txt"%(mass,bkg,inj))) # all 41 features
			fs_names.append("NN Full Sup: m_tau1 m_tau2 deltaR_tautau")
			plt_title = "Phi%ivs%s_Feature-Comparison_Full-Supervision"%(mass,bkg) 
		# #ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivs%s_sig0.000.txt"%(mass,bkg)))
		# #ws_names.append("Bkg vs Bkg: 0%% signal")
		#print(ws_names)
		plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)

ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
ws_lists.append(np.loadtxt("losses/fpr_tpr_test_sig0.100_train0.70_val0.10.txt"))
ws_names.append("NN IAD: test")
fs_lists.append(np.loadtxt("losses/fpr_tpr_test_sig0.100_fs_train0.70_val0.10.txt"))
fs_names.append("NN Full Sup: test")
plt_title = "test_sig0.100_fs_train0.70_val0.10"
plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)

'''
