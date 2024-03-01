#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TLorentzVector.h"

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#endif

/*
 example running
 root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/vbfHToTauTau-M750/Events/run_01/tag_1_delphes_events.root")'

 */

GenParticle* getMother(TClonesArray *branchParticle, const GenParticle *particle);

void printMotherHistory(TClonesArray *branchParticle, const GenParticle *particle);

void readDelphes(const char *inputFile) {
  int debug = 0;
  gSystem->Load("libDelphes");
  
  std::string fullpath = inputFile;
  std::istringstream iss(inputFile);
  std::string sampleName;
  
  std::getline(iss, sampleName, '/');
  int i = 0;
  while (i < 5) {
    std::getline(iss, sampleName, '/');
    ++i;
  }
  
  std::cout << "Sample used is " << sampleName << std::endl;
  

  TChain chain("Delphes");
  chain.Add(inputFile);
//  chain.Add("/Users/tav/Software/MG5_aMC_v3_5_3/vbfHToTauTau-M750/Events/run_01/tag_1_delphes_events.root");

  ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
  Long64_t numberOfEntries = treeReader->GetEntries();

    // Get pointers to branches used in this analysis
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchParticle = treeReader->UseBranch("Particle");

  TH1F histJetInvMass("jet_mass", ";M_{inv}(j_{1}, j_{2}) [GeV];Jets / 5 GeV", 150, 0.0, 1500.0);
  TH1F histJetTauInvMass("jetTau_mass", ";M_{inv}(j_{tau, 1}, j_{tau, 2}) [GeV];Tau jets / 5 GeV", 150, 0.0, 1500.0);
  TH1F histJetPt("jet_pt", ";p_{T}(j) [GeV];Jets / 5 GeV", 150, 0.0, 1500.0);
  TH1F histJetTauPt("jetTau_pt", ";p_{T}(j_{tau}) [GeV];Jets / 5 GeV", 150, 0.0, 1500.0);
  TH1F histJetEta("jet_eta", ";#eta(j);Jets / bin", 60, -3., 3.);
  TH1F histJetTauEta("jetTau_eta", ";#eta(j_{tau}) ;Jets  / bin", 60, -3., 3.);
  TH1F histJetN("jet_N", ";N(j);Jets / 1", 15, -0.5, 14.5);
  TH1F histJetTauN("jetTau_N", ";N(j_{tau});Jets  / bin", 15, -0.5, 14.5);
  TH1F histGenTauDauther("genTauDauther", ";#tau daughter PDG IDs;Generator particle  / bin", 15, -0.5, 14.5);
  histGenTauDauther.GetXaxis()->SetBinLabel(1,"ele");
  histGenTauDauther.GetXaxis()->SetBinLabel(2,"#nu_{ele}");
  histGenTauDauther.GetXaxis()->SetBinLabel(3,"mu");
  histGenTauDauther.GetXaxis()->SetBinLabel(4,"#nu_{mu}");
  histGenTauDauther.GetXaxis()->SetBinLabel(5,"#nu_{mu}");
  histGenTauDauther.GetXaxis()->SetBinLabel(6,"#gamma");
  histGenTauDauther.GetXaxis()->SetBinLabel(7,"#pi^{0}");
  histGenTauDauther.GetXaxis()->SetBinLabel(8,"#pi^{-/+}");
  histGenTauDauther.GetXaxis()->SetBinLabel(9,"K^{0}");
  histGenTauDauther.GetXaxis()->SetBinLabel(10,"K^{+/-}");
  histGenTauDauther.GetXaxis()->SetBinLabel(11,"K^{0}_{L}");
  histGenTauDauther.GetXaxis()->SetBinLabel(12,"K^{0}_{S}");
  histGenTauDauther.GetXaxis()->SetBinLabel(13,"#eta");
  histGenTauDauther.GetXaxis()->SetBinLabel(14,"#omega");
  
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
      if (!jet) continue;
      ++numJets;
      histJetPt.Fill(jet->PT);
      histJetEta.Fill(jet->Eta);

      if (jet->TauTag == 1) {
        ++numTauJets;
        histJetTauPt.Fill(jet->PT);
        histJetTauEta.Fill(jet->Eta);
      }

      for (int j = 0; j < branchJet->GetEntries(); ++j) {
        auto* jet2 = static_cast<Jet*>(branchJet->At(j));
        if (!jet2 || jet == jet2) continue;

        if (!filled) histJetInvMass.Fill((jet->P4() + jet2->P4()).M());
        filled = true;

        if (jet2->TauTag == 1) {
          if (!filledTau) histJetTauInvMass.Fill((jet->P4() + jet2->P4()).M());
          filledTau = true;
        }
      }
    }

    histJetTauN.Fill(numTauJets);
    histJetN.Fill(branchJet->GetEntries());
    
    // -------------------------------------------------------------------------------
    // Loop on the gen particles

    int numGenPart = 0;
    for (int i = 0; i < branchParticle->GetEntries(); ++i) {
      GenParticle *gen = (GenParticle*) (branchParticle->At(i));
      // dont look at non-final state genparts
      if (!gen) continue;
      if (gen->Status != 1)  continue;
      // count the remaining gen parts
      ++numGenPart;
      if (debug > 2) {
        std::cout << "Status = " <<  gen->Status << ", PID: " << gen->PID <<  std::endl;
        printMotherHistory(branchParticle, gen);
      }
      GenParticle *genMom = getMother(branchParticle, gen);
      if (genMom==0) continue;
      if (abs(genMom->PID) == 15) {
        int genPID = abs(gen->PID);
        if (genPID == 11) { // ele
          histGenTauDauther.Fill(0);
        } else if (genPID == 12) { // nu ele
          histGenTauDauther.Fill(1);
        } else if (genPID == 13) { // mu
          histGenTauDauther.Fill(2);
        } else if (genPID == 14) { // nu mu
          histGenTauDauther.Fill(3);
        } else if (genPID == 16) { // nu tau
          histGenTauDauther.Fill(4);
        }  else if (genPID == 22) { // gamma
          histGenTauDauther.Fill(5);
        } else if (genPID == 111) { // pi0
          histGenTauDauther.Fill(6);
        } else if (genPID == 211) { // pi+/-
          histGenTauDauther.Fill(7);
        } else if (genPID == 311) { // K0
          histGenTauDauther.Fill(8);
        } else if (genPID == 321 || genPID == 323) { // K+
          histGenTauDauther.Fill(9);
        } else if (genPID == 130) { // KL0
          histGenTauDauther.Fill(10);
        } else if (genPID == 310) { // Ks
          histGenTauDauther.Fill(11);
        } else if (genPID == 221) { // eta
          histGenTauDauther.Fill(12);
        } else if (genPID == 223) { // omega
          histGenTauDauther.Fill(13);
        } else {
          std::cout << "This particle comes from a tau decay, PID is " << gen->PID <<  std::endl;
        } // check if it's a tau or not
      }
      // Print statements here
      // print(,gen.PID,", E: ",gen.E,", PT: ",gen.PT,", Eta: ",gen.Eta,", M: ",gen.Mass,", M1: ",gen.M1,", M2: ",gen.M2,", D1: ",gen.D1,", D2: ",gen.D2)
    } // end lopp on genPart
  } // end loop on events

  std::string outputFileName = "/Users/tav/Documents/1Research/diTauCathode/Histos_" + sampleName + ".root";
  TFile outputFile(outputFileName.c_str(), "RECREATE");
  histJetInvMass.Write();
  histJetTauInvMass.Write();
  histJetPt.Write();
  histJetTauPt.Write();
  histJetEta.Write();
  histJetTauEta.Write();
  histJetN.Write();
  histJetTauN.Write();
  histGenTauDauther.Write();
  outputFile.Close();
}


// -------------------------------------------------------------------------------

GenParticle* getMother(TClonesArray *branchParticle, const GenParticle *particle){
  GenParticle *genMother = NULL;
  if (particle == 0 ){
    printf("ERROR! null candidate pointer, this should never happen\n");
    return 0;
  }
  
    // Is this the first parent with a different ID? If yes, return, otherwise
    // go deeper into recursion
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


void printMotherHistory(TClonesArray *branchParticle, const GenParticle *gen){
  int lastPID = 0;
  GenParticle *genCurrent =  (GenParticle*)  gen;
  while (abs(lastPID) != 2212 || (abs(lastPID) < 6) ) {
    GenParticle *genTemp = getMother(branchParticle, genCurrent);
    if (!genTemp) break;
    lastPID = abs(genTemp->PID);
    genCurrent = genTemp;
    std::cout << "  > Mother ID = " << lastPID << " stats = " << genTemp->Status << endl;
    if (lastPID == 15 )  std::cout << "    >> Found the tau!" << endl;
  }
}
