#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
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

int create_dataset(string file_n, int label) {


	int debug = 0; 
	gSystem->Load("libDelphes");
	int isSig = label;

	char infile[200], outfile[200];
	string csv_path = "/uscms/home/ssekhar/nobackup/CATHODE_ditau/Delphes/";
	string in_path = "root://cmseos.fnal.gov//store/user/tvami/diTauCathode/";
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
        int n_jets, n_bjets, n_jets_all;
        float m_jet1jet2, m_bjet1bjet2;
	float tau1_ncharged, tau1_nneutrals, tau1_ehadeem, tau2_ncharged, tau2_nneutrals, tau2_ehadeem;
	float jet1_m, jet1_pt, jet1_eta, jet1_phi, bjet1_m, bjet1_pt, bjet1_eta, bjet1_phi, jet1_ehadeem, bjet1_ehadeem, jet1_cef, jet1_nef, bjet1_cef, bjet1_nef;
	float jet2_m, jet2_pt, jet2_eta, jet2_phi, bjet2_m, bjet2_pt, bjet2_eta, bjet2_phi, jet2_ehadeem, bjet2_ehadeem, jet2_cef, jet2_nef, bjet2_cef, bjet2_nef;
	Float_t n_subj[5];
	Double_t jet_pt[5], bjet_pt[5];
	Int_t sorted_jet_idx[5], sorted_bjet_idx[5];	
	std::cout << "Running on " << n_frac << " out of " << numberOfEntries << " events" << std::endl;
	int numTauJet1s = 0, numTauJet2s = 0, numGenTau1s = 0, numGenTau2s = 0, numGenTauJet1s = 0, numGenTauJet2s = 0;
	int nevents = 0;
//  numberOfEntries = 1000;
	for (Long64_t entry = 0; entry < numberOfEntries; ++entry) {
	//for (Long64_t entry = 0; entry < n_frac; ++entry) {	
		if (entry % 20000 == 0) {
			std:cout << "Processing event " << entry << std::endl;
		
		}
		treeReader->ReadEntry(entry);
		bool filled = false;
		bool filledTau = false, filledGenTau = false;
		bool filledJet1 = false, filledJet2 = false;
		bool filledBjet1 = false, filledBjet2 = false;
		bool found_gtau1 = false, found_gtau2 = false;
		int numJets = 0;
		n_jets = 0; n_bjets = 0;
		jet1_m = 0., jet1_pt = 0.; bjet1_m = 0., bjet1_pt = 0.;
		jet1_eta = 0., jet1_phi = 0., bjet1_eta = 0., bjet1_phi = 0.,jet1_ehadeem = 0, bjet1_ehadeem = 0.;
		jet1_cef = 0., jet1_nef = 0.,bjet1_cef = 0., bjet1_nef = 0.; 
		jet2_m = 0., jet2_pt = 0.; bjet2_m = 0., bjet2_pt = 0.;
        jet2_eta = 0., jet2_phi = 0., bjet2_eta = 0., bjet2_phi = 0.,jet2_ehadeem = 0, bjet2_ehadeem = 0.;
        jet2_cef = 0., jet2_nef = 0.,bjet2_cef = 0., bjet2_nef = 0.;
		TLorentzVector jet1_p4, bjet1_p4;
			
			// set aside 2 highest pT jets assuming they are hadronic taus
			// use the remaining jets to check for jets and bjets
			n_jets_all = branchJet->GetEntries();
			if(n_jets_all > 2) // this means there are two "fake taus" and some more jets
			{
				for (int i = 2; i < branchJet->GetEntries(); i++){
				 Jet *jet = (Jet*) branchJet->At(i);
				if (jet->BTag == 1) n_bjets++;
				else {if (jet->TauTag == 0) n_jets++;}
			}
			}
			
			for (int i = 0; i < branchJet->GetEntries(); ++i) {

					Jet *jet = (Jet*) branchJet->At(i);
					MissingET *met = (MissingET*) branchMET->At(0);
				if (!jet) continue;
				if(n_bjets > 0){
					if (jet->BTag == 1){
						if(!filledBjet1){
							bjet1_m = (jet->P4()).M();
							bjet1_pt = jet->PT;
							bjet1_eta = jet->Eta;
							bjet1_phi = jet->Phi;
							bjet1_ehadeem = jet->EhadOverEem;
							bjet1_nef = jet->NeutralEnergyFraction;
							bjet1_cef = jet->ChargedEnergyFraction;
							filledBjet1 = true;
							if(n_bjets==2) bjet1_p4 = jet->P4();
						}
						else{
							
							if(filledBjet1 and n_bjets > 1){
								bjet2_m = (jet->P4()).M();
								bjet2_pt = jet->PT;
								bjet2_eta = jet->Eta;
								bjet2_phi = jet->Phi;
								bjet2_ehadeem = jet->EhadOverEem;
								bjet2_nef = jet->NeutralEnergyFraction;
								bjet2_cef = jet->ChargedEnergyFraction;
								m_bjet1bjet2 = (bjet1_p4 + jet->P4()).M();
								filledBjet2 = true;
							}
						}
					}		
				}
				if(n_jets > 0){
					if (jet->TauTag == 0 and jet->BTag == 0){
						if(!filledJet1){
							jet1_m = (jet->P4()).M();
							jet1_pt = jet->PT;
							jet1_eta = jet->Eta;
							jet1_phi = jet->Phi;
							jet1_ehadeem = jet->EhadOverEem;
							jet1_nef = jet->NeutralEnergyFraction;
							jet1_cef = jet->ChargedEnergyFraction;
							filledJet1 = true;
							if(n_jets==2) jet1_p4 = jet->P4();
						}
						else{
							if(filledJet1 and n_jets > 1){
								jet2_m = (jet->P4()).M();
								jet2_pt = jet->PT;
								jet2_eta = jet->Eta;
								jet2_phi = jet->Phi;
								jet2_ehadeem = jet->EhadOverEem;
								jet2_nef = jet->NeutralEnergyFraction;
								jet2_cef = jet->ChargedEnergyFraction;
								m_jet1jet2 = (jet1_p4 + jet->P4()).M();
								filledJet2 = true;
							}
						
						}
					}		
				}
			}

				//assume: jet1 and jet2 are fake taus 

				Jet *jet = (Jet*) branchJet->At(0);
				MissingET *met = (MissingET*) branchMET->At(0);
				
				tau1_pt = jet->PT;
				tau1_eta = jet->Eta;
				tau1_phi = jet->Phi;
				tau1_m = (jet->P4()).M();
				//for(int i=0; i<5; i++) n_subj[i] = (jet->Tau)[i];
				met_met = met->MET;
				met_eta = met->Eta;
				met_phi = met->Phi;
				tau1_ncharged = jet->NCharged;
				tau1_nneutrals = jet->NNeutrals;
				tau1_ehadeem = jet->EhadOverEem;

				Jet *jet2 = (Jet*) branchJet->At(1);
				
				m_tau1tau2 = (jet->P4() + jet2->P4()).M();
				pt_tau1tau2 = (jet->P4() + jet2->P4()).Pt();
				eta_tau1tau2 = (jet->P4() + jet2->P4()).Eta();	
				phi_tau1tau2 = (jet->P4() + jet2->P4()).Phi();
				tau2_pt = jet2->PT;
				tau2_eta = jet2->Eta;
				tau2_phi = jet2->Phi;
				tau2_m = (jet2->P4()).M();
				tau2_ncharged = jet2->NCharged;
				tau2_nneutrals = jet2->NNeutrals;
				tau2_ehadeem = jet2->EhadOverEem; 
				

				filledTau = true;
							
						
		
			if(filledTau) {
				float deltaR_jet1jet2, deltaR_bjet1bjet2, deltaR_tau1tau2;

				
				//printf("n_jets = %i,jet1_pt = %.2f, jet1_eta = %.2f, jet1_phi = %.2f, n_bjets = %i, bjet1_pt = %.2f, bjet1_eta = %.2f, bjet1_phi = %.2f\n",n_jets,jet1_pt, jet1_eta, jet1_phi,bjet1_pt, bjet1_eta, bjet1_phi, n_bjets);
				if (n_jets == 0) { jet1_pt = 0., jet1_eta = 0., jet1_phi = 0., jet1_ehadeem = 0., m_jet1jet2 = 0.;}
				if (n_bjets == 0) {bjet1_pt = 0., bjet1_eta = 0., bjet1_phi = 0., bjet1_ehadeem = 0., m_bjet1bjet2 = 0.;}
				if (n_jets < 2) {jet2_pt = 0., jet2_eta = 0., jet2_phi = 0., jet2_ehadeem = 0., m_jet1jet2 = jet1_m;}
                if (n_bjets < 2) {bjet2_pt = 0., bjet2_eta = 0., bjet2_phi = 0., bjet2_ehadeem = 0., m_bjet1bjet2 = bjet1_m;}


				deltaR_tau1tau2 = pow((pow((tau1_eta - tau2_eta),2) +  pow((tau1_phi - tau2_phi),2)),0.5);
				deltaR_jet1jet2 = pow((pow((jet1_eta - jet2_eta),2) +  pow((jet1_phi - jet2_phi),2)),0.5);
				deltaR_bjet1bjet2 = pow((pow((bjet1_eta - bjet2_eta),2) +  pow((bjet1_phi - bjet2_phi),2)),0.5);

				

				if(m_tau1tau2 >= 120){
					nevents++;
					fprintf(fout,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i\n", 
					m_jet1jet2, deltaR_jet1jet2, m_bjet1bjet2, deltaR_bjet1bjet2, deltaR_tau1tau2,
					tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
					tau2_m, m_tau1tau2, pt_tau1tau2, eta_tau1tau2, phi_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
					jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef, 
					jet2_pt, jet2_eta, jet2_phi, jet2_cef, jet2_nef, bjet2_pt, bjet2_eta, bjet2_phi, bjet2_cef, bjet2_nef, isSig);
	//printf("No. of tau jets = %i\n",numTauJets);  
				}
			}
				
			}
			
			return 1;
		}

