# EEG neurofeedback
Open-source EEG neurofeedback for meditation. Works with all popular EEG headsets, providing adaptive feedback for any kind of meditation and mental activity.

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
git clone https://github.com/OpenBCI/OpenBCI_Python.git
```

## Running
Open project parent folder and run
```bokeh serve eeg_neurofeedback --show```
where 'eeg_neurofeedback' is the name of project folder
