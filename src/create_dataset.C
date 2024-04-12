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
	int n_frac = 10;
		// Get pointers to branches used in this analysis
	
	TClonesArray *branchJet = treeReader->UseBranch("Jet");
	TClonesArray *branchParticle = treeReader->UseBranch("Particle");
	TClonesArray *branchMET = treeReader->UseBranch("MissingET");
	TClonesArray *branchMu = treeReader->UseBranch("Muon");
	TClonesArray *branchEl = treeReader->UseBranch("Electron");
	TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");

	float tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2, met_met, met_eta, met_phi, tau1_d1, tau1_d2, tau2_d1, tau2_d2;
	Float_t n_subj[5];
	
	std::cout << "Running on " << n_frac << " out of " << numberOfEntries << " events" << std::endl;
	int numTauJet1s = 0, numTauJet2s = 0, numGenTau1s = 0, numGenTau2s = 0, numGenTauJet1s = 0, numGenTauJet2s = 0;
	
//  numberOfEntries = 1000;
	for (Long64_t entry = 0; entry < n_frac; ++entry) {
		
		if (entry % 20000 == 0) {
			std:cout << "Processing event " << entry << std::endl;
			printf("No. of events with at least 1 tagged hadronic tau jets = %i\n",numTauJet1s);
			printf("No. of events with at least 2 tagged hadronic tau jets = %i\n",numTauJet2s);
			printf("No. of events with 1 gen tau- jet = %i\n",numGenTau1s);
			printf("No. of events with 1 gen tau+ jet = %i\n",numGenTau2s);

		}
		treeReader->ReadEntry(entry);
		bool filled = false;
		bool filledTau = false, filledGenTau = false;
		bool found_gtau1 = false, found_gtau2 = false;
		int numJets = 0;
		//printf("No. of gen particles in this event: %i\n",branchParticle->GetEntries());
		// for (int y=0; y < branchGenJet->GetEntries();++y){
		// 	Jet *gjet = (Jet*) branchGenJet->At(y); 
		// 	if (!gjet or gjet->TauTag != 1) continue;        
		// 	if (gjet->TauTag == 1 and !filledGenTau) ++numGenTauJet1s;
		// 	for (int z = 0; z < branchGenJet->GetEntries(); ++z) {
		// 		auto* gjet2 = static_cast<Jet*>(branchGenJet->At(z));
		// 		if (!gjet2 || gjet == gjet2) continue;
		// 		if (gjet2->TauTag == 1) {
		// 			if (!filledGenTau) {  
		// 				++numGenTauJet2s;
		// 				filledGenTau = true;
		// 			}}
		// 		}
		// 	}      
		

				  
		//printf("No. of gen tau- babies = %i, no. of gen tau+ babies = %i\n",numGenTau1s,numGenTau2s);

				for (int i = 0; i < branchJet->GetEntries(); ++i) {

					Jet *jet = (Jet*) branchJet->At(i);

					if (!jet or jet->TauTag != 1) continue;
			 //printf("No. of MET in this event: %i\n",branchMET->GetEntries());
					MissingET *met = (MissingET*) branchMET->At(0);
			//if(isBkg) label = 0; 
			//else label = 1;

					if (jet->TauTag == 1 and !filledTau) {
						++numTauJet1s;
						tau1_pt = jet->PT;
						tau1_eta = jet->Eta;
						tau1_phi = jet->Phi;
						tau1_m = (jet->P4()).M();
						for(int i=0; i<5; i++) n_subj[i] = (jet->Tau)[i];
							met_met = met->MET;
						met_eta = met->Eta;
						met_phi = met->Phi;
					}

					for (int j = 0; j < branchJet->GetEntries(); ++j) {
						auto* jet2 = static_cast<Jet*>(branchJet->At(j));
						if (!jet2 || jet == jet2) continue;


						if (jet2->TauTag == 1) {
							if (!filledTau) {
								m_tau1tau2 = (jet->P4() + jet2->P4()).M();
								tau2_pt = jet2->PT;
								tau2_eta = jet2->Eta;
								tau2_phi = jet2->Phi;
								tau2_m = (jet2->P4()).M();
						//tau2_d1 = p->D1;
						//tau2_d2 = p->D2;   
								numTauJet2s++;	

								filledTau = true;
							}
						}
					}
				}
				if(filledTau) {
					fprintf(fout,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i\n", tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2, 
						n_subj[0],n_subj[1],n_subj[2], n_subj[3],n_subj[4],met_met, met_eta, met_phi, isSig);
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
			printf("No. of events with 1 gen tau- jet = %i\n",numGenTau1s);
			printf("No. of events with 1 gen tau+ jet = %i\n",numGenTau2s);
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
