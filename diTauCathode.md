
## Bkg
 * ttbar->TauTau (sm-ckm_no_b_mass model)
   SM_ttbarTo2Tau2Nu_2J

 * DY->TauTau (sm-ckm_no_b_mass model)
   SM_dyToTauTau_0J1J2J
   	SM_dyToTauTau_0J1J2J_wPU

 * ggF Higgs->TauTau (loop_SM model)
   SM_ggHToTauTau_0J 

 * vbF Higgs->TauTau (sm-ckm_no_b_mass model)
   SM_vbfHToTauTau_2J
   
 * vbF HH (H->bb H->TauTau )
   SM_vbfHHToBBTauTau_2J


## Signal

 * nonRes LQ->TauTau (LQ_UFO model)
   LQ_nonResScalarLQ-M1000_2J

 * ggF Phi->TauTau (2HDM_NLO model [noborn=QCD])
   2HDM_ggPhiToTauTau-M90_0J --> 43693 events :(
   2HDM_ggPhiToTauTau-M750_0J

 * vbF Phi->TauTau (2HDM model)
   2HDM_vbfPhiToTauTau-M750_2J --> I want to compare this to SM_vbfHToTauTau-M750_2J
   2HDM_vbfPhiToTauTau-M90_2J

 * vbF H->bb + Phi->TauTau (2HDM model)
   2HDM_vbfHPhiToBBTauTau-M750_0J      --> this is not VBF

   SM_vbfHHToBBTauTau-M750_2J --> it's an SM Higgs with 750 for two cases, compare to the 2HDM version
   
 * vbF H Phi ( H->bb Phi->TauTau)
   2HDM_vbfHPhiToBBTauTau-M750_2J --> also has the subjets, 27000 weighted events :(
   2HDM_vbfHPhiToBBTauTau-M90_2J --> 72587
   
## Plotting
   root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/SM_vbfHHToBBTauTau_2J/Events/run_01_decayed_1/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/LQ_nonResScalarLQ-M1000_2J/Events/run_01/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/2HDM_ggPhiToTauTau-M750_0J/Events/run_01/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/2HDM_ggPhiToTauTau-M90_0J/Events/run_01/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/2HDM-vbfPhiToTauTau-M750_2J/Events/run_01/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/SM_vbfHToTauTau-M750_2J/Events/run_01/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/2HDM_vbfPhiToTauTau-M90_2J/Events/run_01/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/2HDM_vbfHPhiToBBTauTau-M750_0J/Events/run_01/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/SM_vbfHHToBBTauTau-M750_2J/Events/run_01/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/2HDM_vbfHPhiToBBTauTau-M750_2J/Events/run_01_decayed_1/tag_1_delphes_events.root")'
    root -q -b readDelphes.C'("/Users/tav/Software/MG5_aMC_v3_5_3/2HDM_vbfHPhiToBBTauTau-M90_2J/Events/run_01_decayed_1/tag_1_delphes_events.root")'
    

## Got sidetracked
 * ggF Higgs->TauTau (SMhgg model)
   SMhgg_ggHToTauTau_0J
   SMhgg_ggHToTauTau_1J
   SMhgg_ggHToTauTau_2J

## Other stuff
git clone https://github.com/cms-sw/genproductions.git
(if you need to use mg 2.4.2 then do the following
git clone https://github.com/cms-sw/genproductions.git -b mg242legacy)
cd genproductions/bin/MadGraph5_aMCatNLO/
./gridpack_generation.sh <name of process card without _proc_card.dat> <folder containing cards relative to current location>
e.g.

./gridpack_generation.sh wplustest_4f_LO cards/examples/wplustest_4f_LO 
Note: the output directory specified in the *_proc_card.dat should match the name of the process as used in the gridpack_generation.sh script
