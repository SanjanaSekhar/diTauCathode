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
	sig.columns = ["tau1_pt", "tau1_eta", "tau1_phi", "tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m",
					"m_tau1tau2", "pt_tau1tau2", "eta_tau1tau2", "phi_tau1tau2","met_met", "met_eta", "met_phi", "n_jets", "n_bjets",
					"jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef",
					"jet2_pt", "jet2_eta", "jet2_phi", "jet2_cef", "jet2_nef", "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef", "label"]
	bkg1.columns = sig.columns
	bkg2.columns = sig.columns
	pp = PdfPages('plots/Phi750_ttbar_DY_features.pdf')

	for col in sig.columns:

		plt.figure(figsize=(10,7))
		plt.hist(bkg1[col], label = "DY + 0/1/2 jets", bins = 30, histtype = "step")
		plt.hist(bkg2[col], label = "ttbar + 2 jets", bins = 30, histtype = "step")
		plt.hist(sig[col], label = "VBF Phi + 2 jets, mass = 750 GeV", bins = 30, histtype = "step")
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

	for l in ws_lists:
		fpr, tpr, _ = roc_curve(l[0], l[1])
		bkg_rej = 1 / (fpr+0.001)
		sic = tpr / np.sqrt(fpr+0.001)
		tpr_list.append(tpr)
		bkg_rej_list.append(bkg_rej)
		sic_list.append(sic)

	for l in fs_lists:
		fpr, tpr, _ = roc_curve(l[0], l[1])
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
	plt.legend()
	plt.title(plt_title)
	plt.savefig("plots/SIC_%s.png"%plt_title)
	plt.close()


sig = pd.read_csv("csv_files/2HDM-vbfPhiToTauTau-M750_2J_MinMass120_NoMisTag.csv", lineterminator='\n')
bkg1 = pd.read_csv("csv_files/SM_dyToTauTau_0J1J2J_MinMass120_NoMisTag.csv")
bkg2 = pd.read_csv("csv_files/SM_ttbarTo2Tau2Nu_2J_MinMass120_NoMisTag.csv")

#plot_features(sig, bkg1, bkg2)

injections = ["0.050","0.010","0.005"]
sig_masses = [250,750]


for mass in sig_masses:
	ws_lists, ws_names, fs_lists, fs_names = [],[],[],[]
	for inj in injections:
		ws_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivsDY_sig%s.txt"%(mass,inj)))
		ws_names.append("IAD: %.2f%% signal"%(float(inj)*100))
		fs_lists.append(np.loadtxt("losses/fpr_tpr_Phi%ivsDY_sig%s_fs.txt"%(mass,inj)))
		fs_names.append("Full Sup: %.2f%% signal"%(float(inj)*100))
		plt_title = "Phi%ivsDY"%mass 
	
	#print(ws_names)
	plot_ROC_SIC(ws_lists, ws_names, fs_lists, fs_names, plt_title)
