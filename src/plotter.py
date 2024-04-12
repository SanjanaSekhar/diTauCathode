# Plotter

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 


sig = pd.read_csv("csv_files/2HDM-vbfPhiToTauTau-M750_2J_MinMass120_NoMisTag.csv", lineterminator='\n')
bkg1 = pd.read_csv("csv_files/SM_dyToTauTau_0J1J2J_MinMass120_NoMisTag.csv")
bkg2 = pd.read_csv("csv_files/SM_ttbarTo2Tau2Nu_2J_MinMass120_NoMisTag.csv")

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
