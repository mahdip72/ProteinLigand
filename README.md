# Zero-Shot Protein-Ligand Binding Site Detection

## Abstract

Accurate identification of protein–ligand binding residues is critical for mechanistic biology and drug discovery, yet performance varies widely across ligand families and data regimes. We present a systematic evaluation framework that stratifies ligands into three settings, overrepresented (many examples), underrepresented (tens of examples), and zero-shot (unseen at training). We developed a three-stage, sequence-based modeling suite that progressively adds ligand conditioning and zero-shot capability, and used an evaluation framework to assess the suite. Stage 1 trains per-ligand predictors using a pretrained protein language model (PLM). Stage 2 introduces ligand-aware conditioning via an embedding table, enabling a single multi-ligand model. Stage 3 replaces the table with a pretrained chemical language model (CLM) operating on SMILES, enabling zero-shot generalization. We show Stage 2 improves Macro F1 on the overrepresented test set from 0.4769 (Stage 1) to 0.5832 and outperforms sequence- and structure-based baselines. Stage 3 attains zero-shot performance (F1 = 0.3109) on 5612 previously unseen ligands while remaining competitive on represented ligands. Ablations across five PLM scales and multiple CLMs reveal larger PLM backbones consistently increase Macro F1 across all regimes, whereas scaling the CLM yields modest or inconsistent gains, which need further investigation. 

## Installation

[Provide installation instructions here. Include a bash script for setup.]

```bash
# Example bash script placeholder
#!/bin/bash
# Add installation commands here, e.g.,
# pip install -r requirements.txt
```

## Downloading Datasets

We provide [datasets](https://mailmissouri-my.sharepoint.com/:f:/r/personal/mpngf_umsystem_edu/Documents/Github/Zero-Shot%20Protein-Ligand%20Binding%20Site/BioLip2?csf=1&web=1&e=sj0hsh) for the three evaluation stages described in the manuscript: overrepresented (ligands with >=100 samples), underrepresented (20-99 samples), and zero-shot (<20 samples). These datasets have been slightly modified for compatibility with the UNIMOL 2 chemical encoder (see manuscript Appendix for details). We also release the original unmodified datasets for future research.

- Original unmodified datasets: [link](https://mailmissouri-my.sharepoint.com/:f:/r/personal/mpngf_umsystem_edu/Documents/Github/Zero-Shot%20Protein-Ligand%20Binding%20Site/BioLip2/Original%20datasets?csf=1&web=1&e=o0GNGA)

- Modified datasets: [link](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EibmSj_qXixMn1TZB9vOzNMBzZveaY_P2XzGx-jlbgmqiw?e=ABKN12)


## Inference

[Instructions for performing inference. Include example commands and required inputs/outputs.]

- Stage 2 checkpoint: [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/mpngf_umsystem_edu/Ef-_BQfoVchFjLdSPPOs1w4BSzLocvT-sfPXOm06cK-J8g?e=OHbGyL)

- Stage 3 checkpoint: [link](https://mailmissouri-my.sharepoint.com/:u:/g/personal/mpngf_umsystem_edu/ERN92UnmoYZIgyh0p-7SnZcBLex5-0FSzW6uWd9QS-jaUQ?e=9kIJOi)


## Results

Benchmarking the three Stages on the test sets. Macro F1 scores are based on the average of F1 for each type of ligand.

| Method | Training Dataset | Overrepresented Macro F1 | Underrepresented Macro F1 | Zero-shot Macro F1 | Zero-shot F1 |
|--------|------------------|--------------------------|---------------------------|-------------------|--------------|
| Stage 1 | Overrepresented (separated) | 0.4769 | - | - | - |
| Stage 2 | Overrepresented | 0.5826 ±0.0035 | - | - | - |
| Stage 2 | Overrepresented + Underrepresented | 0.5832 ±0.0014 | 0.3752 ±0.0049 | - | - |
| Stage 3 | Overrepresented + Underrepresented | 0.5526 ±0.0012 | 0.3603 ±0.0029 | 0.2338 ±0.0051 | 0.3109 ±0.0087 |

Comparison of our method’s best performance for each ligand with other available methods on selected ligands in the overrepresented test set based on F1 score. The main values are taken from the original papers, and ∗ indicates methods evaluated on our test sets.

| Ligand | Stage 2 (Our) | Prot2Token | TargetS | LMetalSite | ZinCap | MIB2 | Boltz-2x |
|--------|---------------|------------|---------|------------|--------|------|----------|
| Ca²⁺ | 0.6958 ±0.0011 | 0.6566∗ | 0.392∗ | 0.526 (0.7370∗) | - | - | 0.380∗ |
| Mg²⁺ | 0.5637 ±0.0036 | 0.4603∗ | 0.433∗ | 0.367 (0.5560∗) | - | - | 0.339∗ |
| Zn²⁺ | 0.8180 ±0.0017 | 0.7594∗ | 0.660∗ | 0.760 (0.8299∗) | 0.451∗ | - | 0.557∗ |
| Mn²⁺ | 0.7663 ±0.0113 | 0.7376∗ | 0.579∗ | 0.662 (0.8048∗) | - | - | 0.419∗ |

## 📜 Citation

If you use this code or the pretrained models, please cite the following paper:

```bibtex
will be added soon
```