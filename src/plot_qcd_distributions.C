#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TGraph.h"
#include "TLorentzVector.h"
#include <cstdlib>
#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "../../classes/DelphesClasses.h"
#include "../../external/ExRootAnalysis/ExRootTreeReader.h"
#include "../../external/ExRootAnalysis/ExRootResult.h"
#endif

/*
 example running:
 root -l create_dataset.C
 */

class TFile;

void plot_qcd_distributions() {


	int debug = 0; 
	gSystem->Load("libDelphes");
	int isSig = label;

	char infile[200], outfile[200];
	string csv_path = "/uscms/home/ssekhar/nobackup/CATHODE_ditau/Delphes/";
	string in_path = "root://cmseos.fnal.gov//store/user/tvami/diTauCathode/";
    string file_n = "SM_QCD_JJ_0J1J2J_MinMass120_LO_1M_Part1";
	//string file_name = "LQ_nonResScalarLQ-M1000_2J";
	string file_name = file_n.c_str();
	sprintf(infile,"%s%s.root",in_path.c_str(),file_name.c_str());
	TFile * fin = TFile::Open(infile);
	FILE *fout;
	sprintf(outfile,"%s/diTauCathode/csv_files/%s.csv",csv_path.c_str(),file_name.c_str());
	fout = fopen(outfile, "w");
	
	std::cout << "Sample used is " << file_name.c_str() << std::endl;
	

	TChain chain("Delphes");
	chain.Add(infile);

	ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
	Long64_t numberOfEntries = treeReader->GetEntries();
	int n_frac = 200;
		// Get pointers to branches used in this analysis
	
	TClonesArray *branchJet = treeReader->UseBranch("Jet");
	TClonesArray *branchParticle = treeReader->UseBranch("Particle");
	TClonesArray *branchMET = treeReader->UseBranch("MissingET");
	TClonesArray *branchMu = treeReader->UseBranch("Muon");
	TClonesArray *branchEl = treeReader->UseBranch("Electron");
	TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");

	float tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2, pt_tau1tau2, eta_tau1tau2, phi_tau1tau2, met_met, met_eta, met_phi, tau1_d1, tau1_d2, tau2_d1, tau2_d2;
        int n_jets, n_bjets, n_taus, n_extra_jets;
        float m_jet1jet2, m_bjet1bjet2, pt_jet1jet2;
	float tau1_ncharged, tau1_nneutrals, tau1_ehadeem, tau2_ncharged, tau2_nneutrals, tau2_ehadeem;
	float jet1_m, jet1_pt, jet1_eta, jet1_phi, bjet1_m, bjet1_pt, bjet1_eta, bjet1_phi, jet1_ehadeem, bjet1_ehadeem, jet1_cef, jet1_nef, bjet1_cef, bjet1_nef;
	float jet2_m, jet2_pt, jet2_eta, jet2_phi, bjet2_m, bjet2_pt, bjet2_eta, bjet2_phi, jet2_ehadeem, bjet2_ehadeem, jet2_cef, jet2_nef, bjet2_cef, bjet2_nef;
	float deltaR_jet1jet2, deltaR_bjet1bjet2, deltaR_tau1tau2;

	std::cout << "Running on " << n_frac << " out of " << numberOfEntries << " events" << std::endl;
	int numTauJet1s = 0, numTauJet2s = 0, numGenTau1s = 0, numGenTau2s = 0, numGenTauJet1s = 0, numGenTauJet2s = 0;
	int nevents = 0;

	TH1F m_jj("m_jj", "m_jj", 150, 0.0, 1500.0);
	TH1F m_tautau("m_tautau", "m_tautau", 150, 0.0, 1500.0);
	TH1F pT_j1("pT_j1", "pT_j1", 150, 0.0, 1500.0);
	TH1F pT_j2("pT_j2", "pT_j2", 150, 0.0, 1500.0);
	TH1F pT_jj("pT_jj", "pT_jj", 150, 0.0, 1500.0);
	TH1F n_tauH("n_tauH", "n_tauH", 15, -0.5, 14.5);
	TH1F n_ex_jets("n_extra_jets", "n_extra_jets", 15, -0.5, 14.5);
	TH1F pT_tau1("pT_tau1", "pT_tau1", 150, 0.0, 1500.0);
	TH1F pT_tau2("pT_tau2", "pT_tau2", 150, 0.0, 1500.0);
	TH1F pT_tautau("pT_tautau", "pT_tautau", 150, 0.0, 1500.0);
	TH1F dR_jj("dR_jj", "dR_jj", 20, -1, 1);
	TH1F dR_tautau("dR_tautau", "dR_tautau", 20, -1, 1);
	
	float arr_mjj[1000], arr_mtt[1000], arr_ntaus[1000], arr_jet1pt[1000]; int k = 0; 

//  numberOfEntries = 1000;
	for (Long64_t entry = 0; entry < numberOfEntries; ++entry) {
	//for (Long64_t entry = 0; entry < n_frac; ++entry) {	
		if (entry % 20000 == 0) {
			std:cout << "Processing event " << entry << std::endl;
		
		}
		treeReader->ReadEntry(entry);
		bool filled = false;
		bool filledTau1 = false, filledTau2 = false;
		bool filledJet1 = false, filledJet2 = false;
		bool filledBjet1 = false, filledBjet2 = false;
		bool found_gtau1 = false, found_gtau2 = false;
		int numJets = 0;
		n_jets = 0; n_bjets = 0; n_taus = 0; n_extra_jets = 0;
		jet1_m = 0., jet1_pt = 0.; bjet1_m = 0., bjet1_pt = 0.;
		jet1_eta = 0., jet1_phi = 0., bjet1_eta = 0., bjet1_phi = 0.,jet1_ehadeem = 0, bjet1_ehadeem = 0.;
		jet1_cef = 0., jet1_nef = 0.,bjet1_cef = 0., bjet1_nef = 0.; 
		jet2_m = 0., jet2_pt = 0.; bjet2_m = 0., bjet2_pt = 0.;
        jet2_eta = 0., jet2_phi = 0., bjet2_eta = 0., bjet2_phi = 0.,jet2_ehadeem = 0, bjet2_ehadeem = 0.;
        jet2_cef = 0., jet2_nef = 0.,bjet2_cef = 0., bjet2_nef = 0.;
		deltaR_tau1tau2 = 0.,deltaR_jet1jet2 = 0;
		TLorentzVector tau1_p4, jet1_p4;
		
			
			for (int i = 0; i < branchJet->GetEntries(); i++){
				Jet *jet = (Jet*) branchJet->At(i);
				if (!jet) continue;
				
				if(!filledJet1){
					jet1_pt = jet->PT;
					jet1_eta = jet->Eta;
					jet1_phi = jet->Phi;
					jet1_p4 = jet->P4();
					filledJet1 = true;
				}
				else{
					jet2_pt = jet->PT;
					jet2_eta = jet->Eta;
					jet2_phi = jet->Phi;
					m_jet1jet2 = (jet1_p4 + jet->P4()).M();
					deltaR_jet1jet2 = pow((pow((jet1_eta - jet2_eta),2) +  pow((jet1_phi - jet2_phi),2)),0.5);
					pt_tau1tau2 = (jet1_p4 + jet->P4()).Pt();
					filledJet2 = true;
				}
				if(filledJet2) n_extra_jets++;
				if (jet->BTag == 1) {
					n_bjets++;
					bjet1_pt = jet->PT;
					bjet1_eta = jet->Eta;
					bjet1_phi = jet->Phi;
				}
				if (jet->TauTag == 1) { // not a real tau
					n_taus++;
					if(!filledTau1){
						
						tau1_pt = jet->PT;
						tau1_eta = jet->Eta;
						tau1_phi = jet->Phi;
						tau1_p4 = jet->P4();
						filledTau1 = true;
					}
					else if(!filledTau2){
						tau2_pt = jet->PT;
						tau2_eta = jet->Eta;
						tau2_phi = jet->Phi;
						m_tau1tau2 = (tau1_p4 + jet->P4()).M();
						pt_tau1tau2 = (tau1_p4 + jet->P4()).Pt();
						eta_tau1tau2 = (tau1_p4 + jet->P4()).Eta();	
						phi_tau1tau2 = (tau1_p4 + jet->P4()).Phi();
						deltaR_tau1tau2 = pow((pow((tau1_eta - tau2_eta),2) +  pow((tau1_phi - tau2_phi),2)),0.5);
						filledTau2 = true;
					}
					
				} 			
				
			}
			if(n_taus > 0) {
				arr_mjj[k] = m_jet1jet2;
				arr_mtt[k] = m_tau1tau2;
				arr_ntaus[k] = n_taus;
				arr_jet1pt[k] = jet1_pt;
				k++;
			}
			m_jj->Fill(m_jet1jet2);
			m_tautau->Fill(m_tau1tau2);
			n_ex_jets->Fill(n_extra_jets);
			n_tauH->Fill(n_taus);
			dR_jj->Fill(deltaR_jet1jet2);
			dR_tautau->Fill(deltaR_tau1tau2);
			pT_j1->Fill(jet1_pt);
			pT_j2->Fill(jet2_pt);
			pT_tau1->Fill(tau1_pt);
			pT_tau2->Fill(tau2_pt);
			pT_tautau->Fill(pt_tau1tau2);
			pT_jj->Fill(pt_jet1jet2);

			
		}
		string outputFileName = "Histos_" + file_n + ".root";
		TFile outputFile(outputFileName.c_str(), "RECREATE");
		
		m_jj->Write();
		m_tautau->Write();
		n_ex_jets->Write();
		n_tauH->Write();
		dR_jj->Write();
		dR_tautau->Write();
		pT_j1->Write();
		pT_j2->Write();
		pT_tau1->Write();
		pT_tau2->Write();
		pT_tautau->Write();
		pT_jj->Write();

		outputFile.Close();

		TCanvas *c = new TCanvas("c", "Histograms", 200, 10, 900, 700);
		auto g = new TGraph(k,arr_mtt,arr_ntaus);
		g->setTitle("No. of fake hadronic taus per event vs m_{#tau#tau}; m_{#tau#tau}; No. of fake #tau_H");
		g->Draw("AC*");

		c->Print("ntaus_vs_mtt_QCD.png");
		delete c;

		TCanvas *c2 = new TCanvas("c2", "Histograms", 200, 10, 900, 700);
		auto g2 = new TGraph(k,arr_mjj,arr_ntaus);
		g2->setTitle("No. of fake hadronic taus per event vs m_{jj}; m_{jj}; No. of fake #tau_H");
		g2->Draw("AC*");

		c2->Print("ntaus_vs_mjj_QCD.png");
		delete c2;

		TCanvas *c3 = new TCanvas("c3", "Histograms", 200, 10, 900, 700);
		auto g3 = new TGraph(k,arr_jet1pt,arr_ntaus);
		g3->setTitle("No. of fake hadronic taus per event vs pT_j1; pT_j1; No. of fake #tau_H");
		g3->Draw("AC*");

		c3->Print("ntaus_vs_pTj1_QCD.png");
		delete c3;


}

