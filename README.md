# MarginalContrastiveDiscrimination

Implementation of the Marginal Contrastive Discrimination (MCD) method for conditional density estimation (CDE) in python. It is compatible with any regressor or binary classifier which implements "\_\_init\_\_()", "fit(X, y)", "predict(X)"/"predict\_proba(X)" (eg: scikit-learn, xgboost, catboost, FFNN_InitFitPredict). It also provides a straight-forward and unified API for the CDE task inspired from scikit-learn for both density estimators ("\_\_init\_\_()", "fit(X, y)", "pdf\_from\_X(X)") and density models (generate\_X(), generate\_y\_from\_X(X), "evaluate_divergences(estimator)"). 

# The implementation encapsulates
- ...

# Repository content

- ...
- requirements.txt lists required libraries.

# Requirements: numpy, scikit-learn,

```bash
conda create --name MCD_demo python=3.7
conda activate MCD_demo
conda install ipykernel scikit-learn=0.23.2 scipy=1.6.2 matplotlib=3.4.3 -c anaconda -y 
conda deactivate
wget 'https://github.com/benjaminriu/MarginalContrastiveDiscrimination.git'
cd FFNN_InitFitPredict
```

If using MCD with the FFNN_InitFitPredict repository, then do:

```bash
conda activate MCD_demo
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y 
conda deactivate
wget 'https://github.com/benjaminriu/FFNN_InitFitPredict.git'
```

### if no gpu available:
- just replace 
```bash
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
```
- with 
```bash
conda install pytorch=1.9.1 -c pytorch -y
```

If using MCD with CatBoost, then do:

```bash
conda activate MCD_demo
conda install catboost=0.26.1 -c conda-forge -y
conda deactivate
```

# How to cite

If you use the MarginalContrastiveDiscrimination repository in your research (or use MCD in general), you can cite it as follows:
```
@article{riu2022mcd,
  title={MCD : Marginal Contrastive Discrimination},
  author={Riu, Benjamin},
  journal={arXiv preprint : arXiv:2106.11959},
  year={2022}
}
```

# To do next list and key missing features
- Add a negative sampling functions for model with known density but no available sampler,
- Add multi-dimensional target support for density models and MCD
- ...
