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

First we create .csv files containing the pT, eta, phi and m_tau of the 2 tau jets in signal and background samples. `create_dataset.C` takes the root fle name and the label (0 for bkg, 1 for sig) as inputs. Modify the `csv_path` accordingly.
```
git clone git@github.com:SanjanaSekhar/diTauCathode.git
root -l -b -q create_dataset.C'("SM_ttbarTo2Tau2Nu_2J",0)'
```
On LPC we can use the GPUs through Singularity containers. More info can be found [here](https://uscms.org/uscms_at_work/computing/setup/gpu.shtml). Specifically the Docker tags can be found [here](https://hub.docker.com/r/fnallpc/fnallpc-docker/tags). We can then run the weak supervision script.
```
singularity run --nv --bind `readlink $HOME` --bind `readlink ~/nobackup` --bind /cvmfs --bind /cvmfs/unpacked.cern.ch --bind /storage /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:pytorch-1.13.0-cuda11.6-cudnn8-runtime-singularity
Singularity> python src/weak_supervision.py
```
(More info to be added.)
