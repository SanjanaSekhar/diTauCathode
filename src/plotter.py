# Plotter

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from sklearn.metrics import roc_curve

def plot_features(sig,bkg1,bkg2):


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
		plt.savefig("plots/%s.png"%col)
		plt.close()

def plot_ROC_SIC(true_list1, pred_list1,name1,true_list2, pred_list2,name2):
	fpr, tpr, _ = roc_curve(true_list1, pred_list1)
	bkg_rej = 1 / fpr
	sic = tpr / np.sqrt(fpr)

	fpr2, tpr2, _ = roc_curve(true_list2, pred_list2)
	bkg_rej2 = 1 / fpr2
	sic2 = tpr2 / np.sqrt(fpr2)

	random_tpr = np.linspace(0, 1, len(fpr))
	random_bkg_rej = 1 / random_tpr
	random_sic = random_tpr / np.sqrt(random_tpr)

	# ROC curve
	plt.plot(tpr, bkg_rej, label=name1)
	plt.plot(tpr2, bkg_rej2, label=name2)
	plt.plot(random_tpr, random_bkg_rej, label="random")
	plt.xlabel("True Positive Rate")
	plt.ylabel("Background Rejection")
	plt.yscale("log")
	plt.legend(loc="upper right")

	plt.savefig("plots/ROC_%s.png"%name1)
	plt.close()

	# SIC curve
	plt.plot(tpr, sic, label=name1)
	plt.plot(tpr2, sic2, label=name2)
	plt.plot(random_tpr, random_sic, label="random")
	plt.xlabel("True Positive Rate")
	plt.ylabel("Significance Improvement")
	plt.legend(loc="upper right")

	plt.savefig("plots/SIC_%s.png"%name1)
	plt.close()


sig = pd.read_csv("csv_files/2HDM-vbfPhiToTauTau-M750_2J_MinMass120_NoMisTag.csv", lineterminator='\n')
bkg1 = pd.read_csv("csv_files/SM_dyToTauTau_0J1J2J_MinMass120_NoMisTag.csv")
bkg2 = pd.read_csv("csv_files/SM_ttbarTo2Tau2Nu_2J_MinMass120_NoMisTag.csv")

# plot_features(sig, bkg1, bkg2)

lists = np.loadtxt("losses/fpr_tpr_Phivsttbar.txt")
lists2 = np.loadtxt("losses/fpr_tpr_Phivsttbar_fs.txt")
plot_ROC_SIC(lists[0],lists[1], "Phivsttbar-weak_sup",lists2[0],lists2[1], "Phivsttbar-full_sup")

lists = np.loadtxt("losses/fpr_tpr_PhivsDY.txt")
lists2 = np.loadtxt("losses/fpr_tpr_PhivsDY_fs.txt")
plot_ROC_SIC(lists[0],lists[1], "PhivsDY-weak_sup",lists2[0],lists2[1],"PhivsDY-full_sup")