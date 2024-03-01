# diTauCathode

## Setting up Delphes on LPC

Links:
* [Delphes](https://github.com/delphes/delphes/tree/master)
* [LCG Software and standalone ROOT in LPC](https://uscms.org/uscms_at_work/computing/setup/setup_software.shtml#lcgsoft)

```
git clone git@github.com:delphes/delphes.git
cd Delphes
source /cvmfs/sft.cern.ch/lcg/views/LCG_103swan/x86_64-centos7-gcc11-opt/setup.sh 
make
```

## Testing weak supervision

First we create .csv files containing the pT, eta, phi and m_tau of the 2 tau jets in signal and background samples. 
```
# note that creata_dataset.C should be placed inside the Delphes folder
# The file takes the root fle name and the label (0 for bkg, 1 for sig) as inputs
root -l -b -q create_dataset.C'("SM_ttbarTo2Tau2Nu_2J",0)'
```
