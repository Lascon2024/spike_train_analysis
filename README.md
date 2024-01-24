# spike_train_analysis
Materials for Spike Train Analysis tutorials

## Setting up the environment
  - > mamba create --name lascon2024_spike_train_analysis python=3.11 numpy scipy statsmodels matplotlib tqdm ipykernel
  - > conda activate lascon2024_spike_train_analysis
  - > pip install elephant
  - > pip install viziphant
  - > python -m ipykernel install --user --name lascon2024_spike_train_analysis
  - If jupyter-lab is not installed:
    - > conda install jupyterlab
  - If jupyter-lab raises ImportError:
    - > conda install chardet

## Running the notebooks
  - Make sure that the kernel "lascon2024_spike_train_analysis" is used to run the notebook.
