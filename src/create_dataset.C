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

    // Get pointers to branches used in this analysis
  TClonesArray *branchJet = treeReader->UseBranch("Jet");

  float tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2;
  
  
  std::cout << "Running on " << numberOfEntries << " events" << std::endl;

//  numberOfEntries = 1000;
  for (Long64_t entry = 0; entry < numberOfEntries; ++entry) {

    if (entry % 2000 == 0) {
      std:cout << "Processing event " << entry << std::endl;
    }
    treeReader->ReadEntry(entry);
    bool filled = false;
    bool filledTau = false;
    int numTauJets = 0;
    int numJets = 0;

    for (int i = 0; i < branchJet->GetEntries(); ++i) {
      Jet *jet = (Jet*) branchJet->At(i);
      if (!jet or jet->TauTag != 1) continue;

      //if(isBkg) label = 0; 
      //else label = 1;

      if (jet->TauTag == 1) {
        ++numTauJets;
        tau1_pt = jet->PT;
        tau1_eta = jet->Eta;
        tau1_phi = jet->Phi;
        tau1_m = (jet->P4()).M();

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
              }
          filledTau = true;
        }
      }
    }
    if(filledTau) fprintf(fout,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%i\n", tau1_pt, tau1_eta, tau1_phi, tau2_pt, tau2_eta, tau2_phi, tau1_m, tau2_m, m_tau1tau2, isSig);
  }
return 1;
}
