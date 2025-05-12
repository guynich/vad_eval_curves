AUC metrics example

Dataset Card on HuggingFace.
https://huggingface.co/datasets/guynich/librispeech_asr_test_vad

Model: Silero VAD
https://github.com/snakers4/silero-vad

# Introduction

This repo computes AUC metrics for the test dataset with Silero VAD model.

# Installation

This section describes installation for the working test example in this repo.

The first step is to clone this repo.
```console
cd
git clone git@github.com:guynich/vad_eval_curves.git
```

The main script has dependencies.  For these steps I used Ubuntu 22.04 and
Python `venv` virtual environment.  The script plots require tkinter.
```console
sudo apt install -y python3.10-venv
sudo apt-get install python3-tk

cd
python3 -m venv venv_vad_eval_curves
source ./venv_vad_eval_curves/bin/activate

cd vad_eval_curves

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

# Run the test script

```console
cd
source ./venv_vad_eval_curves/bin/activate
cd vad_eval_curves

python3 main.py
```

## Results

### test.clean

<img src="images/ROC_test_clean.png" alt="AUC plots for test.clean"/>

Speech features marked as low confidence are excluded in the following plot.  See
[Dataset Card](https://huggingface.co/datasets/guynich/librispeech_asr_test_vad)
for discussion.

<img src="images/ROC_test_clean_exclude_low_confidence.png" alt="AUC plots for test.clean excluding zero confidence data"/>

### test.other

<img src="images/ROC_test_other.png" alt="AUC plots for test.clean"/>

Speech features marked as low confidence are excluded in the following plot.  See
[Dataset Card](https://huggingface.co/datasets/guynich/librispeech_asr_test_vad)
for discussion.

<img src="images/ROC_test_other_exclude_low_confidence.png" alt="AUC plots for test.clean excluding zero confidence data"/>

```
{'test.clean': {'Overall PR AUC': np.float64(0.9916533637164553),
                'Overall ROC AUC': np.float64(0.9749639873286956)},
 'test.clean_confidence': {'Overall PR AUC': np.float64(0.9982586011226073),
                           'Overall ROC AUC': np.float64(0.992260886130102)},
 'test.other': {'Overall PR AUC': np.float64(0.9856187361132123),
                'Overall ROC AUC': np.float64(0.9690341897727688)},
 'test.other_confidence': {'Overall PR AUC': np.float64(0.9971949276641505),
                           'Overall ROC AUC': np.float64(0.9914300840606078)}}
```
