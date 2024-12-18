# Plotter

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd 
from sklearn.metrics import roc_curve, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

import ROOT
from ROOT import *
gStyle.SetOptStat(0)
gROOT.SetBatch(1)

def binwidth_normalize(hist):
	for j in range(hist.GetNbinsX()):
		binc = hist.GetBinContent(j)
		width = hist.GetBinWidth(j)
		hist.SetBinContent(j,binc/width)
	return hist

def compare_QCD(bkgs,bkg_labels,plot_label):


	# Format of csv file:
	# tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
	# tau2_m, m_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
	# jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef, isSig

	m_sig = ROOT.TH1F("m_sig", "m_sig", 60, 0.0, 800.0)
	m_tt_sig = ROOT.TH1F("m_tt_sig", "m_tt_sig", 60, 100.0, 900.0)
	delta_sig = ROOT.TH1F("delta_sig","delta_sig",20, 0, 5)
	n_sig = ROOT.TH1F("n_sig","n_sig", 7,0,7.)
	m_bkg, m_tt_bkg, delta_bkg, n_bkg = [],[],[],[]
	
	for i in range(len(bkgs)):
		m_tt_bkg.append(m_tt_sig.Clone("m_tt_bkg%i" %i))
		m_bkg.append(m_sig.Clone("m_bkg%i" %i))
		delta_bkg.append(delta_sig.Clone("delta_bkg%i" %i))
		n_bkg.append(n_sig.Clone("n_bkg%i" %i))
		m_tt_bkg[i].Sumw2()
		m_bkg[i].Sumw2()
		delta_bkg[i].Sumw2()
		n_bkg[i].Sumw2()
	
	columns = ["m_jet1jet2", "pt_jet1jet2", "deltaR_jet1jet2", "m_bjet1bjet2", "deltaR_bjet1bjet2", "deltaR_tau1tau2","deltaeta_tau1tau2","tau1_pt", "tau1_eta", "tau1_phi", 
				"tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m","m_tau1tau2", "pt_tau1tau2", "eta_tau1tau2", "phi_tau1tau2",
				"met_met", "met_eta", "met_phi", "n_jets", "n_bjets","jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", 
				"bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef","jet2_pt", "jet2_eta", "jet2_phi", "jet2_cef", "jet2_nef", 
				"bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef", "event_weight","label"]
	
	bkg = []
	for b in bkgs:
		bkg.append(pd.read_csv("csv_files/%s.csv" % b))


	for i in range(len(bkg)):
		bkg[i].columns = columns
		#bkg[i]["deltaeta_tau1tau2"] = abs(bkg[i]['tau1_eta'] - bkg[i]['tau2_eta'])
		bkg[i] = bkg[i][["deltaR_jet1jet2", "deltaeta_tau1tau2","deltaR_tau1tau2","n_jets", "n_bjets","m_tau1tau2", "m_jet1jet2", "pt_tau1tau2", "pt_jet1jet2", "met_met",  "event_weight"]]

	bkg[1]["n_jets"] = bkg[1]["n_jets"] - 1
	bkg[1]["event_weight"] = bkg[1]["event_weight"] * 50
	#print("mjj in QCD: ",bkg[2]["m_jet1jet2"].min(),bkg[2]["m_jet1jet2"].max()) 
	#print("deltaR_tau1tau2 in QCD: ",bkg[2]["deltaR_tau1tau2"].min(),bkg[2]["deltaR_tau1tau2"].max())

	colors = [kTeal-5, kPink+6]

	for i in range(len(bkg)):

		m_bkg[i].SetLineColor(colors[i])
		m_tt_bkg[i].SetLineColor(colors[i])
		delta_bkg[i].SetLineColor(colors[i])
		n_bkg[i].SetLineColor(colors[i])

		m_bkg[i].SetLineWidth(3)
		m_tt_bkg[i].SetLineWidth(3)
		delta_bkg[i].SetLineWidth(3)
		n_bkg[i].SetLineWidth(3)

	for i in range(len(bkg)):
		m_bkg[i].Reset()
		m_tt_bkg[i].Reset()
		delta_bkg[i].Reset()
		n_bkg[i].Reset()
			
	
	c = ROOT.TCanvas('qcd', 'qcd', 900, 700)
	for idx,col in enumerate(bkg[0].columns[:-1]):

		for i in range(len(bkg)):
			for entry,wt in zip(bkg[i][col], bkg[i]["event_weight"]):
				#wt = 1
				if 'm_t' in col: m_tt_bkg[i].Fill(entry, wt)
				elif 'm' in col or 'pt' in col: m_bkg[i].Fill(entry, wt)
				elif 'delta' in col: delta_bkg[i].Fill(entry, wt)
				elif 'n' in col: n_bkg[i].Fill(entry, wt)
			
			if 'm_t' in col: 
				m_tt_bkg[i] = binwidth_normalize(m_tt_bkg[i])
			elif 'm' in col or 'pt' in col: 
				m_bkg[i] = binwidth_normalize(m_bkg[i])
			elif 'delta' in col: 
				delta_bkg[i] = binwidth_normalize(delta_bkg[i])
			elif 'n' in col: 
				n_bkg[i] = binwidth_normalize(n_bkg[i])
		
		if 'm_t' in col: 
			m_tt_bkg[0].SetTitle("Distribution of "+col)
			m_tt_bkg[0].Draw("hist")
			m_tt_bkg[1].Draw("hist same")
			leg_mtt = ROOT.TLegend(0.65,0.8,0.9,0.9)
		elif 'm' in col or 'pt' in col: 
			m_bkg[0].SetTitle("Distribution of "+col)
			m_bkg[0].Draw("hist")
			m_bkg[1].Draw("hist same")
			leg_m = ROOT.TLegend(0.65,0.8,0.9,0.9)
		elif 'delta' in col: 
			delta_bkg[0].SetTitle("Distribution of "+col)
			delta_bkg[0].Draw("hist")
			delta_bkg[1].Draw("hist same")
			leg_delta = ROOT.TLegend(0.65,0.8,0.9,0.9)
		elif 'n' in col:
			n_bkg[0].SetTitle("Distribution of "+col)
			n_bkg[0].Draw("hist")
			n_bkg[1].Draw("hist same")
			leg_n = ROOT.TLegend(0.65,0.8,0.9,0.9)

		
		for i in range(len(bkg)):
			if 'm_t' in col: leg_mtt.AddEntry(m_tt_bkg[i], bkg_labels[i])
			elif 'm' in col or 'pt' in col: leg_m.AddEntry(m_bkg[i], bkg_labels[i])
			elif 'delta' in col: leg_delta.AddEntry(delta_bkg[i], bkg_labels[i])
			elif 'n' in col: leg_n.AddEntry(n_bkg[i], bkg_labels[i])
		
		if 'm_t' in col: leg_mtt.Draw()
		elif 'm' in col or 'pt' in col: leg_m.Draw()
		elif 'delta' in col: leg_delta.Draw()
		elif 'n' in col: leg_n.Draw()

		c.Update()
		if idx==0: c.Print("plots/compare%s.pdf("%plot_label)
		elif idx==len(bkg[0].columns)-2: c.Print("plots/compare%s.pdf)"%plot_label)
		else: c.Print("plots/compare%s.pdf"%plot_label)

def plot_features(sigs,sig_labels,bkgs,bkg_labels,sig_scale,plot_label):


	# Format of csv file:
	# tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
	# tau2_m, m_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
	# jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef, isSig

	m_sig = ROOT.TH1F("m_sig", "m_sig", 60, 0.0, 800.0)
	m_tt_sig = ROOT.TH1F("m_tt_sig", "m_tt_sig", 60, 100, 1000.0)
	delta_sig = ROOT.TH1F("delta_sig","delta_sig",30, 0, 5)
	n_sig = ROOT.TH1F("n_sig","n_sig",7,0,7)
	m_bkg, m_tt_bkg, delta_bkg, n_bkg = [],[],[],[]
	m_sig.Sumw2()
	delta_sig.Sumw2()
	n_sig.Sumw2()
	m_tt_sig.Sumw2()
	
	for i in range(len(bkgs)):
		m_tt_bkg.append(m_tt_sig.Clone("m_tt_bkg%i" %i))
		m_bkg.append(m_sig.Clone("m_bkg%i" %i))
		delta_bkg.append(delta_sig.Clone("delta_bkg%i" %i))
		n_bkg.append(n_sig.Clone("n_bkg%i" %i))
		#m_tt_bkg[i].Sumw2()
		#m_bkg[i].Sumw2()
		#delta_bkg[i].Sumw2()
		#n_bkg[i].Sumw2()
	
	columns = ["m_jet1jet2", "pt_jet1jet2", "deltaR_jet1jet2", "m_bjet1bjet2", "deltaR_bjet1bjet2", "deltaR_tau1tau2","deltaeta_tau1tau2","tau1_pt", "tau1_eta", "tau1_phi", 
				"tau2_pt", "tau2_eta", "tau2_phi", "tau1_m","tau2_m","m_tau1tau2", "pt_tau1tau2", "eta_tau1tau2", "phi_tau1tau2",
				"met_met", "met_eta", "met_phi", "n_jets", "n_bjets","jet1_pt", "jet1_eta", "jet1_phi", "jet1_cef", "jet1_nef", 
				"bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_cef", "bjet1_nef","jet2_pt", "jet2_eta", "jet2_phi", "jet2_cef", "jet2_nef", 
				"bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_cef", "bjet2_nef", "event_weight","label"]
	
	bkg = []
	for b in bkgs:
		bkg.append(pd.read_csv("csv_files/%s.csv" % b))


	for i in range(len(bkg)):
		bkg[i].columns = columns
		#bkg[i]["deltaeta_tau1tau2"] = abs(bkg[i]['tau1_eta'] - bkg[i]['tau2_eta'])
		bkg[i] = bkg[i][["deltaR_jet1jet2", "deltaeta_tau1tau2","deltaR_tau1tau2","n_jets", "n_bjets","m_tau1tau2", "m_jet1jet2", "pt_tau1tau2", "pt_jet1jet2","met_met",  "event_weight"]]

	#bkg[0]["n_jets"] = bkg[0]["n_jets"] -2
	#print("mjj in QCD: ",bkg[2]["m_jet1jet2"].min(),bkg[2]["m_jet1jet2"].max()) 
	#print("deltaR_tau1tau2 in QCD: ",bkg[2]["deltaR_tau1tau2"].min(),bkg[2]["deltaR_tau1tau2"].max())

	colors = [kTeal-5, kBlue-7, kPink+6]

	for i in range(len(bkg)):

		m_bkg[i].SetLineColor(colors[i])
		m_tt_bkg[i].SetLineColor(colors[i])
		delta_bkg[i].SetLineColor(colors[i])
		n_bkg[i].SetLineColor(colors[i])

		m_bkg[i].SetFillColor(colors[i])
		m_tt_bkg[i].SetFillColor(colors[i])
		delta_bkg[i].SetFillColor(colors[i])
		n_bkg[i].SetFillColor(colors[i])

	m_sig.SetLineColor(kRed)
	m_tt_sig.SetLineColor(kRed)
	delta_sig.SetLineColor(kRed)
	n_sig.SetLineColor(kRed)

	m_sig.SetLineWidth(2)
	m_tt_sig.SetLineWidth(2)
	delta_sig.SetLineWidth(2)
	n_sig.SetLineWidth(2)

	scale = sig_scale # scale signal

	for sig__,sig_label in zip(sigs,sig_labels):
		sig = pd.read_csv("csv_files/%s.csv" % sig__)

		sig.columns = columns
		#sig["deltaeta_tau1tau2"] = abs(sig['tau1_eta'] - sig['tau2_eta'])
		sig = sig[["deltaR_jet1jet2", "deltaeta_tau1tau2","deltaR_tau1tau2","n_jets", "n_bjets","m_tau1tau2", "m_jet1jet2", "pt_tau1tau2","pt_jet1jet2", "met_met",  "event_weight"]]
		print(sig_label)

		for i in range(len(bkg)):
			m_bkg[i].Reset()
			m_tt_bkg[i].Reset()
			delta_bkg[i].Reset()
			n_bkg[i].Reset()
			
			#print("reset bkg hists")

		m_sig.Reset()
		m_tt_sig.Reset()
		delta_sig.Reset()
		n_sig.Reset()
		
		c = ROOT.TCanvas(sig__, sig__, 900, 700)
		for idx,col in enumerate(sig.columns[:-1]):
			if 'm_t' in col: stack_mtt = ROOT.THStack(col, col)
			elif 'm' in col or 'pt' in col: stack = ROOT.THStack(col, col)
			elif 'delta' in col: stack_delta = ROOT.THStack(col, col)
			elif 'n' in col: stack_n = ROOT.THStack(col, col)
			#print("Plotting ", col)
			for i in range(len(bkg)):
				for entry,wt in zip(bkg[i][col], bkg[i]["event_weight"]):
					#wt = 1
					if 'm_t' in col:
						#if i==0: print(entry,wt) 
						m_tt_bkg[i].Fill(entry, wt)
					elif 'm' in col or 'pt' in col: m_bkg[i].Fill(entry, wt)
					elif 'delta' in col: delta_bkg[i].Fill(entry, wt)
					elif 'n' in col: n_bkg[i].Fill(entry, wt)
				
				if 'm_t' in col: 
					#m_tt_bkg[i].Print("range")
					m_tt_bkg[i] = binwidth_normalize(m_tt_bkg[i])
					#m_tt_bkg[i].Print("range")
					stack_mtt.Add(m_tt_bkg[i])
				elif 'm' in col or 'pt' in col: 
					m_bkg[i] = binwidth_normalize(m_bkg[i])
					stack.Add(m_bkg[i])
				elif 'delta' in col: 
					delta_bkg[i] = binwidth_normalize(delta_bkg[i])
					stack_delta.Add(delta_bkg[i])
				elif 'n' in col: 
					n_bkg[i] = binwidth_normalize(n_bkg[i])
					stack_n.Add(n_bkg[i])
			#print("Filled bkg histograms")
			for entry in sig[col]:
				if 'm_t' in col: m_tt_sig.Fill(entry, scale)
				elif 'm' in col or 'pt' in col: m_sig.Fill(entry, scale)
				elif 'delta' in col: delta_sig.Fill(entry, scale)
				elif 'n' in col: n_sig.Fill(entry, scale)
			
			m_tt_sig = binwidth_normalize(m_tt_sig)
			m_sig = binwidth_normalize(m_sig)
			delta_sig =  binwidth_normalize(delta_sig)
			n_sig = binwidth_normalize(n_sig)

			#c = ROOT.TCanvas(col, col, 900, 700)
			if 'm_t' in col: 
				stack_mtt.SetTitle("Distribution of "+col)
				stack_mtt.Draw("hist")
				m_tt_sig.Draw("hist same")
				leg_mtt = ROOT.TLegend(0.65,0.65,0.9,0.9)
			elif 'm' in col or 'pt' in col: 
				stack.SetTitle("Distribution of "+col)
				stack.Draw("hist")
				m_sig.Draw("hist same")
				leg_m = ROOT.TLegend(0.65,0.65,0.9,0.9)
			elif 'delta' in col: 
				stack_delta.SetTitle("Distribution of "+col)
				stack_delta.Draw("hist")
				delta_sig.Draw("hist same")
				leg_delta = ROOT.TLegend(0.65,0.65,0.9,0.9)
			elif 'n' in col:
				stack_n.SetTitle("Distribution of "+col)
				stack_n.Draw("hist")
				n_sig.Draw("hist same")
				leg_n = ROOT.TLegend(0.65,0.65,0.9,0.9)

			
			for i in range(len(bkg)):
				if 'm_t' in col: leg_mtt.AddEntry(m_tt_bkg[i], bkg_labels[i])
				elif 'm' in col or 'pt' in col: leg_m.AddEntry(m_bkg[i], bkg_labels[i])
				elif 'delta' in col: leg_delta.AddEntry(delta_bkg[i], bkg_labels[i])
				elif 'n' in col: leg_n.AddEntry(n_bkg[i], bkg_labels[i])
			
			if 'm_t' in col: 
				leg_mtt.AddEntry(m_tt_sig, str(scale)+" * "+sig_label)
				leg_mtt.Draw()
			elif 'm' in col or 'pt' in col: 
				leg_m.AddEntry(m_sig, str(scale)+" * "+sig_label)
				leg_m.Draw()
			elif 'delta' in col: 
				leg_delta.AddEntry(delta_sig, str(scale)+" * "+sig_label)
				leg_delta.Draw()
			elif 'n' in col: 
				leg_n.AddEntry(n_sig, str(scale)+" * "+sig_label)
				leg_n.Draw()

			c.Update()
			if idx==0: c.Print("plots/"+sig__+"%s.pdf("%plot_label)
			elif idx==len(sig.columns)-2: c.Print("plots/"+sig__+"%s.pdf)"%plot_label)
			else: c.Print("plots/"+sig__+"%s.pdf"%plot_label)

			if 'm_t' in col: stack_mtt.Delete()
			elif 'm' in col or 'pt' in col: stack.Delete()
			elif 'delta' in col:  stack_delta.Delete()
			elif 'n' in col: stack_n.Delete()
			



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

sig_list = ["2HDM-vbfPhiToTauTau-M250_2J_MinMass120_NoMisTag"]
			#"eVLQ_TPrimeTPrimeToTTPhiPhiToTauTauAll_TpM1000_PhiM250_NoMisTag",
			#"HeavyN_vbsNToTauTau_NM250_2J_LO", 
			#"VAL_dyVfVfToXiCXiCToTauSTauS_XiM1000_VfM250_MinMass120_NoMisTag"]

bkg_list = ["SM_QCD_JJ_0J1J2J_MinMass120_LO_6M_TauTag","SM_ttbarTo2Tau2Nu_0J1J2J_MinMass120_MadSpin_2M", "SM_dyToTauTau_0J1J2J_MinMass120_3M"]
sig_names = ["250 GeV heavy Higgs (VBF)"]#, "250 GeV scalar from T'", "250 GeV HNL", 
				#"250 GeV VAL"]
bkg_names = ["QCD multijet (tautagged)", "ttbar + 0/1/2 jets", "DY + 0/1/2 jets"]
#bkg_names = ["QCD multijet", "QCD multijet (tautagged)"]
#plot_features(sig_list, sig_names, bkg_list, bkg_names, sig_scale = 100, plot_label = "_tautagged")

bkg_list = ["SM_WTo1Tau1Nu_0J1J2J_LO_MinMass120_1M","SM_QCD_JJ_0J1J2J_MinMass120_LO_6M_TauTag"]
bkg_names = ["50 * W+jets (tautagged)", "QCD multijet (tautagged)"]
compare_QCD(bkg_list[::-1], bkg_names[::-1], plot_label = "_QCD_Wjets")

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
