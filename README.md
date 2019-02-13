# EEG neurofeedback
Open-source EEG neurofeedback for meditation. Works with all popular EEG headsets, providing adaptive feedback for any kind of meditation and mental activity.

Base idea behind project is to fit brain pattern of mental activity on the fly (tuning phase) and then provide real-time sound feedback if required mental activity fades away (feedback phase). 

## Protocol
Current protocol goes like this:
* calibration as sequence of relax and target state periods (relax -> target -> relax -> target)
* machine learning is applied to eeg data and distinctive pattern is learnt
* feedback starts with increased sound volume, as target state fades away, that reminds meditator to go back into target state
* after some period of time feedback accuracy is checked, and if accuracy is low, then calibration repeats
* if accuracy is OK, then feedback contunues, and in some period of time starts relax state period, for meditator to have some rest and for machine learning to get new information, as during deepening of meditation brain patterns can change
```
          tuning phase                                  feedback phase
[relax → target → relax → target] → [feedback → accuracy check → feedback → relax] → feedback → ...
                                                        ↓
                                                   tuning phase → feedback phase → ...
```
Next protocol modification:
* add voice commands to tell when meditator distracts, allowing additional learning from these examples

## Headsets
Current headsets available are:
* Fake headset - providing synthetic data for testing
* [OpenBCI Cyton](https://shop.openbci.com/products/cyton-biosensing-board-8-channel?variant=38958638542) with 8 electrodes

Upcoming support for headsets:
* [MUSE 2016](https://choosemuse.com/)
* [MUSE 2](https://choosemuse.com/)
* [NeuroSky MindWave Mobile 2] (http://neurosky.com/)

## Machine learning and data processing
Current pipeline:
```[notch and bandpass filtering] → [1 second FFT spectrum with Hanning window] → [dimension reduction with PCA] → [Machine learning with Support Vector Machine classifier]```

Next ML todo:
* replace filtering and FFT with wavelet based decomposition, using optimization of mother wavelet 
* use pre-computed autoencoding recurrent neural nets for general features extraction

## Prerequisites
Mac OS X, Python 2.7 (Python 3 comparability is in progress)

https://github.com/OpenBCI/OpenBCI_Python - OpenBCI python lib 
```
pandas
numpy
bokeh
sklearn
sounddevice
```

## Installing
```
git clone https://github.com/poddubnyoleg/eeg_neurofeedback.git
```

## Running
* Open project parent folder and run ```bokeh serve eeg_neurofeedback --show``` where 'eeg_neurofeedback' is the name of project folder
* Make sure that all electrodes are connected using real-time graphs
* Press start button and follow sound instructions
