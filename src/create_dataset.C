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
GenParticle* getMother(TClonesArray *branchParticle, const GenParticle *particle);
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

	float tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2, met_met, met_eta, met_phi, tau1_d1, tau1_d2, tau2_d1, tau2_d2;
        int n_jets, n_bjets;
	float tau1_ncharged, tau1_nneutrals, tau1_ehadeem, tau2_ncharged, tau2_nneutrals, tau2_ehadeem;
	float jet1_pt, jet1_eta, jet1_phi, bjet1_pt, bjet1_eta, bjet1_phi, jet1_ehadeem, bjet1_ehadeem, jet1_cef, jet1_nef, bjet1_cef, bjet1_nef;
	float jet2_pt, jet2_eta, jet2_phi, bjet2_pt, bjet2_eta, bjet2_phi, jet2_ehadeem, bjet2_ehadeem, jet2_cef, jet2_nef, bjet2_cef, bjet2_nef;
	Float_t n_subj[5];
	Double_t jet_pt[5], bjet_pt[5];
	Int_t sorted_jet_idx[5], sorted_bjet_idx[5];	
	std::cout << "Running on " << n_frac << " out of " << numberOfEntries << " events" << std::endl;
	int numTauJet1s = 0, numTauJet2s = 0, numGenTau1s = 0, numGenTau2s = 0, numGenTauJet1s = 0, numGenTauJet2s = 0;
	
//  numberOfEntries = 1000;
	//for (Long64_t entry = 0; entry < numberOfEntries; ++entry) {
	for (Long64_t entry = 0; entry < n_frac; ++entry) {	
		if (entry % 20000 == 0) {
			std:cout << "Processing event " << entry << std::endl;
			printf("No. of events with at least 1 tagged hadronic tau jets = %i\n",numTauJet1s);
			printf("No. of events with at least 2 tagged hadronic tau jets = %i\n",numTauJet2s);
			//printf("No. of events with 1 gen tau- jet = %i\n",numGenTau1s);
			//printf("No. of events with 1 gen tau+ jet = %i\n",numGenTau2s);

		
		}
		treeReader->ReadEntry(entry);
		bool filled = false;
		bool filledTau = false, filledGenTau = false;
		bool found_gtau1 = false, found_gtau2 = false;
		int numJets = 0;
		n_jets = 0; n_bjets = 0;
		jet1_pt = 9999999.; bjet1_pt = 9999999.;
		jet1_eta = 0., jet1_phi = 0., bjet1_eta = 0., bjet1_phi = 0.,jet1_ehadeem = 0, bjet1_ehadeem = 0.;
		jet1_cef = 0., jet1_nef = 0.,bjet1_cef = 0., bjet1_nef = 0.; 
		jet2_pt = 9999999.; bjet2_pt = 9999999.;
                jet2_eta = 0., jet2_phi = 0., bjet2_eta = 0., bjet2_phi = 0.,jet2_ehadeem = 0, bjet2_ehadeem = 0.;
                jet2_cef = 0., jet2_nef = 0.,bjet2_cef = 0., bjet2_nef = 0.;
		
			
			
			for (int i = 0; i < branchJet->GetEntries(); i++){
				 Jet *jet = (Jet*) branchJet->At(i);
				if (jet->BTag == 1) n_bjets++;
				else {if (jet->TauTag == 0) n_jets++;}
			}
			cout << "n_jets = " << n_jets << " n_bjets = " << n_bjets<< endl;
			if(n_jets > 0) {
			Double_t jet_pt[n_jets];

			int j = 0, k = 0;
			for (int i = 0; i < branchJet->GetEntries(); i++){
				 Jet *jet = (Jet*) branchJet->At(i);
                                //if (jet->BTag == 1) {bjet_pt[j] = jet->PT; j++;}
                                if (jet->TauTag == 0) {jet_pt[k] = jet->PT; k++;}
                        }
			Int_t sorted_jet_idx[n_jets];
			if(n_jets > 1) {
				TMath::Sort(n_jets, jet_pt, sorted_jet_idx);
				//if(n_bjets > 1) TMath::Sort(n_bjets, bjet_pt, sorted_bjet_idx);
				for(int i = 0; i < n_jets; i++)
					cout << sorted_jet_idx[i] << " ";
				}
			}
			if(n_bjets > 0){
			Double_t bjet_pt[n_bjets];

                        int j = 0, k = 0;
                        for (int i = 0; i < branchJet->GetEntries(); i++){
                                 Jet *jet = (Jet*) branchJet->At(i);
                                if (jet->BTag == 1) {bjet_pt[j] = jet->PT; j++;}
                                //else {if (jet->TauTag == 0) {jet_pt[k] = jet->PT; k++;}}
                        }
                        Int_t sorted_bjet_idx[n_bjets];
                        //if(n_jets > 1) TMath::Sort(n_jets, jet_pt, sorted_jet_idx);
                        if(n_bjets > 1) {
				TMath::Sort(n_bjets, bjet_pt, sorted_bjet_idx);
				for(int i = 0; i < n_bjets; i++)
					cout << sorted_bjet_idx[i] << " ";
				}
			}

			for (int i = 0; i < branchJet->GetEntries(); ++i) {

					Jet *jet = (Jet*) branchJet->At(i);
					MissingET *met = (MissingET*) branchMET->At(0);
				if (!jet) continue;
				if (jet->BTag == 1){
					if(n_bjets > 0 and jet->PT == bjet_pt[sorted_bjet_idx[0]]) {
					
						bjet1_pt = jet->PT;
						bjet1_eta = jet->Eta;
						bjet1_phi = jet->Phi;
						bjet1_ehadeem = jet->EhadOverEem;
						bjet1_nef = jet->NeutralEnergyFraction;
						bjet1_cef = jet->ChargedEnergyFraction;
					}
					else{ if(n_bjets > 1 and jet->PT == bjet_pt[sorted_bjet_idx[1]]) {

                                                bjet2_pt = jet->PT;
                                                bjet2_eta = jet->Eta;
                                                bjet2_phi = jet->Phi;
                                                bjet2_ehadeem = jet->EhadOverEem;
                                                bjet2_nef = jet->NeutralEnergyFraction;
                                                bjet2_cef = jet->ChargedEnergyFraction;
                                        }}
				}
				else{ if(jet->TauTag == 0) {
					
					if(n_jets > 0 and jet->PT == jet_pt[sorted_jet_idx[0]]) {
						jet1_pt = jet->PT;
						jet1_eta = jet->Eta;
						jet1_phi = jet->Phi;
						jet1_ehadeem = jet->EhadOverEem;
						jet1_nef = jet->NeutralEnergyFraction;	
						jet1_cef = jet->ChargedEnergyFraction;	
					}
					else{ if(n_jets > 1 and jet->PT == jet_pt[sorted_jet_idx[1]]) {
                                                jet2_pt = jet->PT;
                                                jet2_eta = jet->Eta;
                                                jet2_phi = jet->Phi;
                                                jet2_ehadeem = jet->EhadOverEem;
                                                jet2_nef = jet->NeutralEnergyFraction;
                                                jet2_cef = jet->ChargedEnergyFraction;
					}}
				}}
				if(jet->TauTag == 1 and !filledTau) {
					++numTauJet1s;
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
				

					for (int j = 0; j < branchJet->GetEntries(); ++j) {
						auto* jet2 = static_cast<Jet*>(branchJet->At(j));
						if (!jet2) continue;
					
						

					if (jet2->TauTag == 1 and jet!=jet2) {
							m_tau1tau2 = (jet->P4() + jet2->P4()).M();
							tau2_pt = jet2->PT;
							tau2_eta = jet2->Eta;
							tau2_phi = jet2->Phi;
							tau2_m = (jet2->P4()).M();
							tau2_ncharged = jet2->NCharged;
                                        		tau2_nneutrals = jet2->NNeutrals;
                                        		tau2_ehadeem = jet2->EhadOverEem; 
							numTauJet2s++;	

							filledTau = true;
							
						}
					}
				}
			}
		
			if(filledTau) {
				//printf("n_jets = %i,jet1_pt = %.2f, jet1_eta = %.2f, jet1_phi = %.2f, n_bjets = %i, bjet1_pt = %.2f, bjet1_eta = %.2f, bjet1_phi = %.2f\n",n_jets,jet1_pt, jet1_eta, jet1_phi,bjet1_pt, bjet1_eta, bjet1_phi, n_bjets);
				if (n_jets == 0) {jet1_pt = 0., jet1_eta = 0., jet1_phi = 0., jet1_ehadeem = 0.;}
				if (n_bjets == 0) {bjet1_pt = 0., bjet1_eta = 0., bjet1_phi = 0., bjet1_ehadeem = 0.;}
				if (n_jets < 2) {jet2_pt = 0., jet2_eta = 0., jet2_phi = 0., jet2_ehadeem = 0.;}
                                if (n_bjets < 2) {bjet2_pt = 0., bjet2_eta = 0., bjet2_phi = 0., bjet2_ehadeem = 0.;}
				//if (jet1_ehadeem > 900) printf("jet1_pt, jet1_eta, jet1_phi, jet1_ehadeem, jet1_nef, jet1_cef = %f, %f, %f, %f, %f, %f\n",jet1_pt, jet1_eta, jet1_phi, jet1_ehadeem,jet1_nef, jet1_cef);
				//if (bjet1_ehadeem > 900) printf("bjet1_pt, bjet1_eta, bjet1_phi, bjet1_ehadeem, bjet1_nef, bjet1_cef = %f, %f, %f, %f, %f, %f\n",bjet1_pt, bjet1_eta, bjet1_phi, bjet1_ehadeem,bjet1_nef, bjet1_cef);
				//printf("tau1_ncharged, tau1_nneutrals, tau1_ehadeem, tau1_ncharged, tau1_nneutrals, tau2_ehadeem = %f,%f,%f,%f,%f,%f\n",tau1_ncharged, tau1_nneutrals, tau1_ehadeem, tau1_ncharged, tau1_nneutrals, tau2_ehadeem);	
				fprintf(fout,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i\n", 
					tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, 
					tau2_m, m_tau1tau2, met_met, met_eta, met_phi, n_jets, n_bjets, 
					jet1_pt, jet1_eta, jet1_phi, jet1_cef, jet1_nef, bjet1_pt, bjet1_eta, bjet1_phi, bjet1_cef, bjet1_nef, 
					jet2_pt, jet2_eta, jet2_phi, jet2_cef, jet2_nef, bjet2_pt, bjet2_eta, bjet2_phi, bjet2_cef, bjet2_nef, isSig);
	//printf("No. of tau jets = %i\n",numTauJets);  
				}
				/*
				int found_ntaus = 0, found_leptau = 0, found_hadtau =0;
		float neu1_pT=0., neu1_eta, neu1_phi, neu2_pT=0., neu2_eta, neu2_phi;
		TLorentzVector neu1_p4, neu2_p4, el_p4, mu_p4, met_p4;
		float delR = 9999.;
		for (int m=0; m < branchParticle->GetEntries();++m){

			if(found_ntaus=2) break;

			GenParticle *p = (GenParticle*) branchParticle->At(m);
				// only select final state electrons or muons
			if(!p or p->Status!=1) continue;
				
				if(p->PID == 11 or p->PID == -11){ //electrons
					GenParticle *genMom = getMother(branchParticle, p);
					if(genMom->PID == 15 or genMom->PID == -15) {
						
						if(genMom->D1 > genMom->D2){
							cout << "Daughter list invalid" << endl;
							continue;
						}
						
						// search for neutrinos
						for(int u = genMom->D1; u < genMom->D2; u++){

							GenParticle *d1 = (GenParticle*) branchParticle->At(u);
							if(d1==p or !d1) continue;
							if(!neu1_p4){ neu1_p4 = d1.P4(); }
							else{ neu2_p4 = d1.P4();}
						}
						//search for reco electron
						for(u = 0; u < branchEl->GetEntries(); u++){
							Electron *e = (Electron*) branchEl->At(u);
							if(!e) continue;
							delR = e.P4().DeltaR(p.P4());
							if(delR < 0.1){
								cout << "found electron at DeltaR = " << delR << "from gen_electron daughter" << endl;
								el_p4 = e.P4();
							}
						}
						// search for missingET
						for(u = 0; u < branchMET->GetEntries(); u++){
							MissingET *met = (Electron*) branchMET->At(u);
							if(!met) continue;
							delR = met.P4().DeltaR(neu1_p4+neu2_p4);
							if(delR < 0.1){
								cout << "found MET at DeltaR = " << delR << "from gen_neutrino daughters" << endl;
								met_p4 = met.P4();
							}
						}
						if(!el_p4 or !met_p4) cout << "Could not find electron daughter or MET" << endl;
						else{
						cout << "found leptonic tau with electron and met daughters" << endl;
						found_leptau ++;
						tau1_p4 = el_p4 + met_p4;
						tau1_m = tau1_p4.M();
						tau1_pt = tau1_p4.Pt();
						tau1_eta = tau1_p4.Eta();
						tau1_phi = tau1_p4.Phi();
						}
					}
				}

				if(p->PID == 13 or p->PID == -13){ //muons
					GenParticle *genMom = getMother(branchParticle, p);
					if(genMom->PID == 15 or genMom->PID == -15) {
						
						if(genMom->D1 > genMom->D2){
							cout << "Daughter list invalid" << endl;
							continue;
						}
						
						// search for neutrinos
						for(int u = genMom->D1; u < genMom->D2; u++){

							GenParticle *d1 = (GenParticle*) branchParticle->At(u);
							if(d1==p or !d1) continue;
							if(!neu1_p4){ neu1_p4 = d1.P4(); }
							else{ neu2_p4 = d1.P4();}
						}
						//search for reco muon
						for(u = 0; u < branchMu->GetEntries(); u++){
							Muon *mu = (Muon*) branchEl->At(u);
							if(!mu) continue;
							delR = mu.P4().DeltaR(p.P4());
							if(delR < 0.1){
								cout << "found electron at DeltaR = " << delR << "from gen_electron daughter" << endl;
								mu_p4 = mu.P4();
							}
						}
						// search for missingET
						for(u = 0; u < branchMET->GetEntries(); u++){
							MissingET *met = (Electron*) branchMET->At(u);
							if(!met) continue;
							delR = met.P4().DeltaR(neu1_p4+neu2_p4);
							if(delR < 0.1){
								cout << "found MET at DeltaR = " << delR << "from gen_neutrino daughters" << endl;
								met_p4 = met.P4();
							}
						}
						if(!el_p4 or !met_p4) cout << "Could not find electron daughter or MET" << endl;
						else{
						cout << "found leptonic tau with electron and met daughters" << endl;
						found_leptau ++;
						tau1_p4 = el_p4 + met_p4;
						tau1_m = tau1_p4.M();
						tau1_pt = tau1_p4.Pt();
						tau1_eta = tau1_p4.Eta();
						tau1_phi = tau1_p4.Phi();
						}
					}
				}	


				}
				*/
			}
			printf("No. of events with at least 1 tagged tau jets = %i\n",numTauJet1s);
			printf("No. of events with at least 2 tagged tau jets = %i\n",numTauJet2s);
			//printf("No. of events with 1 gen tau- jet = %i\n",numGenTau1s);
			//printf("No. of events with 1 gen tau+ jet = %i\n",numGenTau2s);
			return 1;
		}
// Taken from readDelphes.C

		GenParticle* getMother(TClonesArray *branchParticle, const GenParticle *particle){
			GenParticle *genMother = NULL;
			if (particle == 0 ){
				printf("ERROR! null candidate pointer, this should never happen\n");
				return 0;
			}
			if (particle->M1 > 0 && particle->PID != 0) {
				genMother = (GenParticle*) (branchParticle->At(particle->M1));
				if (genMother->PID ==  particle->PID) {
					return getMother(branchParticle, genMother);
				} else {
					return genMother;
				}
			}
			else {
				return NULL;
			}
		}  
