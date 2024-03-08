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

    // Get pointers to branches used in this analysis
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchParticle = treeReader->UseBranch("Particle");
  TClonesArray *branchMET = treeReader->UseBranch("MissingET");
  TClonesArray *branchMu = treeReader->UseBranch("Muon");
  TClonesArray *branchEl = treeReader->UseBranch("Electron");
  //TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");

  float tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2, met_met, met_eta, met_phi, tau1_d1, tau1_d2, tau2_d1, tau2_d2;
  Float_t n_subj[5];
  
  std::cout << "Running on " << numberOfEntries << " events" << std::endl;
  int numTauJet1s = 0, numTauJet2s = 0, numGenTau1s = 0, numGenTau2s = 0;
  
//  numberOfEntries = 1000;
  for (Long64_t entry = 0; entry < numberOfEntries; ++entry) {
    /*
    if (entry % 2000 == 0) {
      std:cout << "Processing event " << entry << std::endl;
      printf("No. of events with at least 1 tagged tau jets = %i\n",numTauJet1s);
      printf("No. of events with at least 2 tagged tau jets = %i\n",numTauJet2s);
      printf("No. of events with 1 gen tau- jet = %i\n",numGenTau1s);
      printf("No. of events with 1 gen tau+ jet = %i\n",numGenTau2s);
      }*/
    treeReader->ReadEntry(entry);
    bool filled = false;
    bool filledTau = false;
    // numGenTau1s = 0;
    int numJets = 0;
    //printf("No. of gen particles in this event: %i\n",branchParticle->GetEntries());
  for (int m=0; m < branchParticle->GetEntries();++m){
     GenParticle *p = (GenParticle*) branchParticle->At(m);
     if(!p or p->Status!=1) continue;
     GenParticle *genMom = getMother(branchParticle, p);
     if(genMom->PID == 15) numGenTau1s++; 
     if(genMom->PID == -15) numGenTau2s++;

    }   
    //printf("No. of gen tau- babies = %i, no. of gen tau+ babies = %i\n",numGenTau1s,numGenTau2s);
   
  for (int i = 0; i < branchJet->GetEntries(); ++i) {

      Jet *jet = (Jet*) branchJet->At(i);
      //Electron *electron = (Electron*) branchEl->At(i);
      //Muon *muon = (Muon*) branchMu->At(i);
      //MissingET *met = (MissingET*) branchMET->At(i);
      //GenParticle *p = (GenParticle*) branchParticle->At(i);
      //GenJet *genjet = (GenJet*) branchGenJet->At(i);

      if (!jet or jet->TauTag != 1) continue;
       //printf("No. of MET in this event: %i\n",branchMET->GetEntries());
       MissingET *met = (MissingET*) branchMET->At(0);
      //if(isBkg) label = 0; 
      //else label = 1;

      if (jet->TauTag == 1) {
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
