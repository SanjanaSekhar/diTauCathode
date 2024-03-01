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
  //if(argc!=3) {
  //	std::cout << "Please enter the root file name and the label ( 0 for bkg and 1 for signal )" << std::endl;
  //	return 0;	
  //}
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
//  chain.Add("/Users/tav/Software/MG5_aMC_v3_5_3/vbfHToTauTau-M750/Events/run_01/tag_1_delphes_events.root");

  ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
  Long64_t numberOfEntries = treeReader->GetEntries();

    // Get pointers to branches used in this analysis
  TClonesArray *branchJet = treeReader->UseBranch("Jet");

  // TH1F histJetInvMass("jet_mass", ";M_{inv}(j_{1}, j_{2}) [GeV];Jets / 5 GeV", 150, 0.0, 1500.0);
  // TH1F histJetTauInvMass("jetTau_mass", ";M_{inv}(j_{tau, 1}, j_{tau, 2}) [GeV];Tau jets / 5 GeV", 150, 0.0, 1500.0);
  // TH1F histJetPt("jet_pt", ";p_{T}(j) [GeV];Jets / 5 GeV", 150, 0.0, 1500.0);
  // TH1F histJetTauPt("jetTau_pt", ";p_{T}(j_{tau}) [GeV];Jets / 5 GeV", 150, 0.0, 1500.0);
  // TH1F histJetEta("jet_eta", ";#eta(j);Jets / bin", 70, -5., 5.);
  // TH1F histJetTauEta("jetTau_eta", ";#eta(j_{tau}) ;Jets  / bin",70, -5., 5.);
  // TH1F histJetN("jet_phi", ";phi(j);Jets / 1", 60, -3.2, 3.2);
  // TH1F histJetTauN("jetTau_phi", ";phi(j_{tau});Jets  / bin", 60, -3.2, 3.2);

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
/*
 150.386200 0.018794 -0.292659 0.000000 0.000000 -0.000000 0.841084 0.000000 0.000000 1
248.562607 0.628225 -1.001127 0.000000 0.000000 -0.000000 14.215500 0.000000 0.000000 1
384.063141 -1.317290 1.378457 0.000000 0.000000 -0.000000 2.655327 0.000000 0.000000 1
384.063141 -1.317290 1.378457 0.000000 0.000000 -0.000000 2.655327 0.000000 0.000000 1
384.063141 -1.317290 1.378457 0.000000 0.000000 -0.000000 2.655327 0.000000 0.000000 1
409.222229 -0.750504 -1.340876 0.000000 0.000000 -0.000000 22.128954 0.000000 0.000000 1
409.222229 -0.750504 -1.340876 0.000000 0.000000 -0.000000 22.128954 0.000000 0.000000 1
409.222229 -0.750504 -1.340876 0.000000 0.000000 -0.000000 22.128954 0.000000 0.000000 1
129.401672 1.598770 -2.875708 0.000000 0.000000 -0.000000 20.087402 0.000000 0.000000 1
108.567940 1.533159 3.141070 108.567940 1.533159 3.141070 6.230235 6.230235 375.426056 1
45.346210 1.502949 2.150964 108.567940 1.533159 3.141070 3.040915 6.230235 375.426056 1
45.346210 1.502949 2.150964 108.567940 1.533159 3.141070 3.040915 6.230235 375.426056 1
45.346210 1.502949 2.150964 108.567940 1.533159 3.141070 3.040915 6.230235 375.426056 1
45.346210 1.502949 2.150964 108.567940 1.533159 3.141070 3.040915 6.230235 375.426056 1
172.505829 0.576267 1.229211 172.505829 0.576267 1.229211 3.536976 3.536976 220.740326 1
172.505829 0.576267 1.229211 172.505829 0.576267 1.229211 3.536976 3.536976 220.740326 1
172.505829 0.576267 1.229211 172.505829 0.576267 1.229211 3.536976 3.536976 220.740326 1
172.505829 0.576267 1.229211 172.505829 0.576267 1.229211 3.536976 3.536976 220.740326 1
51.749622 0.121425 -0.717920 51.749622 0.121425 -0.717920 3.509698 3.509698 147.287445 1
51.749622 0.121425 -0.717920 51.749622 0.121425 -0.717920 3.509698 3.509698 147.287445 1
138.746414 2.359561 0.681335 51.749622 0.121425 -0.717920 8.624253 3.509698 147.287445 1
138.746414 2.359561 0.681335 51.749622 0.121425 -0.717920 8.624253 3.509698 147.287445 1
126.307266 0.391632 2.947564 51.749622 0.121425 -0.717920 3.220380 3.509698 147.287445 1
126.307266 0.391632 2.947564 51.749622 0.121425 -0.717920 3.220380 3.509698 147.287445 1
*/
