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
cd diTauCathode
root -l -b -q src/create_dataset.C'("SM_ttbarTo2Tau2Nu_2J",0)'
```
On LPC we can use the GPUs through Singularity containers. More info can be found [here](https://uscms.org/uscms_at_work/computing/setup/gpu.shtml). Specifically the Docker tags can be found [here](https://hub.docker.com/r/fnallpc/fnallpc-docker/tags). We can then run the weak supervision script.
```
singularity run --nv --bind `readlink $HOME` --bind `readlink ~/nobackup` --bind /cvmfs --bind /cvmfs/unpacked.cern.ch --bind /storage /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:pytorch-1.13.0-cuda11.6-cudnn8-runtime-singularity

Singularity> python src/weak_supervision.py --help
usage: weak_supervision.py [-h] [--name NAME] [--sig SIG] [--bkg BKG] [--early_stop EARLY_STOP] [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS] [--ending ENDING] [--load_model LOAD_MODEL]
                           [--epoch_to_load EPOCH_TO_LOAD] [--train_model TRAIN_MODEL] [--test_model TEST_MODEL] [--full_supervision FULL_SUPERVISION] [--sig_injection SIG_INJECTION] [--bkg_frac BKG_FRAC]
                           [--m_tt_min M_TT_MIN] [--m_tt_max M_TT_MAX]

Train sig vs bkg for identifying CATHODE vars

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           file name extension for residuals and pulls
  --sig SIG             name of the .csv file for the signal
  --bkg BKG             name of the .csv file for the bkg
  --early_stop EARLY_STOP
                        early stopping patience (no. of epochs)
  --batch_size BATCH_SIZE
                        batch size for training
  --n_epochs N_EPOCHS   no. of epochs to train for
  --ending ENDING       date
  --load_model LOAD_MODEL
                        load saved model
  --epoch_to_load EPOCH_TO_LOAD
                        load checkpoint corresponding to this epoch
  --train_model TRAIN_MODEL
                        train and save model
  --test_model TEST_MODEL
                        test model
  --full_supervision FULL_SUPERVISION
                        Run fully supervised
  --sig_injection SIG_INJECTION
                        percent of signal to inject into data
  --bkg_frac BKG_FRAC   n_bkg/n_sig
  --m_tt_min M_TT_MIN   lower boundary for sig region in ditau inv mass
  --m_tt_max M_TT_MAX   upper boundary for sig region in ditau inv mass
```
(More info to be added.)
