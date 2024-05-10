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

def plot_ROC_SIC(true_list1, pred_list1,name1,true_list2, pred_list2,name2, true_list3, pred_list3,name3,true_list4, pred_list4,name4):
	fpr, tpr, _ = roc_curve(true_list1, pred_list1)
	bkg_rej = 1 / (fpr+0.001)
	sic = tpr / np.sqrt(fpr+0.001)

	fpr2, tpr2, _ = roc_curve(true_list2, pred_list2)
	bkg_rej2 = 1 / (fpr2+0.001)
	sic2 = tpr2 / np.sqrt(fpr2+0.001)

	fpr3, tpr3, _ = roc_curve(true_list3, pred_list3)
	bkg_rej3 = 1 / (fpr3+0.001)
	sic3 = tpr3 / np.sqrt(fpr3+0.001)

	fpr4, tpr4, _ = roc_curve(true_list4, pred_list4)
	bkg_rej4 = 1 / (fpr4+0.001)
	sic4 = tpr4 / np.sqrt(fpr4+0.001)

	random_tpr = np.linspace(0, 1, len(fpr))
	random_bkg_rej = 1 / (random_tpr+0.001)
	random_sic = random_tpr / np.sqrt(random_tpr+0.001)

	# ROC curve
	plt.figure(figsize=(6,4))
	plt.plot(tpr, bkg_rej, label=name1)
	plt.plot(tpr2, bkg_rej2, label=name2)
	plt.plot(tpr3, bkg_rej3, label=name3)
	plt.plot(tpr4, bkg_rej4, label=name4)
	plt.plot(random_tpr, random_bkg_rej, label="random")
	plt.xlabel("True Positive Rate")
	plt.ylabel("Background Rejection")
	plt.yscale("log")
	plt.legend(loc="upper right")
	plt.title(name1.replace("-IAD",""))
	plt.savefig("plots/ROC_n9_%s.png"%name1)
	plt.close()

	# SIC curve
	plt.figure(figsize=(6,4))
	plt.plot(tpr, sic, label=name1)
	plt.plot(tpr2, sic2, label=name2)
	plt.plot(tpr3, sic3, label=name1)
	plt.plot(tpr4, sic4, label=name2)
	plt.plot(random_tpr, random_sic, label="random")
	plt.xlabel("True Positive Rate")
	plt.ylabel("Significance Improvement")
	plt.legend(loc="upper right")
	plt.title(name1.replace("-IAD",""))
	plt.savefig("plots/SIC_n9_%s.png"%name1)
	plt.close()


sig = pd.read_csv("csv_files/2HDM-vbfPhiToTauTau-M750_2J_MinMass120_NoMisTag.csv", lineterminator='\n')
bkg1 = pd.read_csv("csv_files/SM_dyToTauTau_0J1J2J_MinMass120_NoMisTag.csv")
bkg2 = pd.read_csv("csv_files/SM_ttbarTo2Tau2Nu_2J_MinMass120_NoMisTag.csv")

plot_features(sig, bkg1, bkg2)


lists = np.loadtxt("losses/fpr_tpr_Phi250vsttbar.txt")
lists2 = np.loadtxt("losses/fpr_tpr_Phi250vsttbar_fs.txt")
lists3 = np.loadtxt("losses/fpr_tpr_Phi250vsttbar_n9.txt")
lists4 = np.loadtxt("losses/fpr_tpr_Phi250vsttbar_n9_fs.txt")
plot_ROC_SIC(lists[0],lists[1], "Phi250vsttbar-IAD",lists2[0],lists2[1], "Phi250vsttbar-full_sup",lists3[0],lists3[1], "Phi250vsttbar-IAD (9 features)",lists4[0],lists4[1], "Phi250vsttbar-full_sup (9 features)")

lists = np.loadtxt("losses/fpr_tpr_Phi750vsttbar.txt")
lists2 = np.loadtxt("losses/fpr_tpr_Phi750vsttbar_fs.txt")
lists3 = np.loadtxt("losses/fpr_tpr_Phi750vsttbar_n9.txt")
lists4 = np.loadtxt("losses/fpr_tpr_Phi750vsttbar_n9_fs.txt")
plot_ROC_SIC(lists[0],lists[1], "Phi750vsttbar-IAD",lists2[0],lists2[1], "Phi750vsttbar-full_sup",lists3[0],lists3[1], "Phi750vsttbar-IAD (9 features)",lists4[0],lists4[1], "Phi750vsttbar-full_sup (9 features)")

lists = np.loadtxt("losses/fpr_tpr_Phi250vsDY.txt")
lists2 = np.loadtxt("losses/fpr_tpr_Phi250vsDY_fs.txt")
lists3 = np.loadtxt("losses/fpr_tpr_Phi250vsDY_n9.txt")
lists4 = np.loadtxt("losses/fpr_tpr_Phi250vsDY_n9_fs.txt")
plot_ROC_SIC(lists[0],lists[1], "Phi250vsDY-IAD",lists2[0],lists2[1], "Phi250vsDY-full_sup",lists3[0],lists3[1], "Phi250vsDY-IAD (9 features)",lists4[0],lists4[1], "Phi250vsDY-full_sup (9 features)")

lists = np.loadtxt("losses/fpr_tpr_Phi750vsDY.txt")
lists2 = np.loadtxt("losses/fpr_tpr_Phi750vsDY_fs.txt")
lists3 = np.loadtxt("losses/fpr_tpr_Phi750vsDY_n9.txt")
lists4 = np.loadtxt("losses/fpr_tpr_Phi750vsDY_n9_fs.txt")
plot_ROC_SIC(lists[0],lists[1], "Phi750vsDY-IAD",lists2[0],lists2[1], "Phi750vsDY-full_sup",lists3[0],lists3[1], "Phi750vsDY-IAD (9 features)",lists4[0],lists4[1], "Phi750vsDY-full_sup (9 features)")
