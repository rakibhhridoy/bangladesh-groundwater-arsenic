# Bangladesh Groundwater Arsenic: Mechanism Partitioning and Screening

Code and model weights for:

> **Phosphate Dominance, Geochemical Bistability, and Seasonal Dynamics in Bangladesh Groundwater Arsenic: National-Scale Mechanism Partitioning and Rapid Risk Screening**
>
> Md Rakib Hasan, Mst Anika Khatun Rupa, Fazla Zawadul Arabi, Shoumik Zubyer, Anwar Zahid
>
> *Water Research* (2026)

## Overview

This repository contains the analysis code and trained model for a national-scale study of groundwater arsenic mechanisms in Bangladesh. We analysed 1,807 groundwater samples from 988 locations across all seven physiographic zones, collected in both dry and wet seasons.

**Key findings:**
- Phosphate competition is the dominant national-scale arsenic driver (rho = 0.326 vs Fe rho = 0.164)
- Mechanism shifts from phosphate to reductive dissolution below ~80 m (Holocene-Pleistocene boundary)
- Monsoon amplifies arsenic risk in 77% of zone-depth combinations (+13.6 million people exposed)
- Geochemical bistability with a phosphate tipping point at 0.37 mg/L
- ArseniCFM screening model achieves AUC = 0.781 with spatial block CV; 50 samples match full training

## Repository Structure

```
src/
  h1_geochemistry/     # H1: Multi-mechanism analysis (phosphate vs reductive)
  h2_depth/            # H2: Depth-stratified geochemical transitions
  h3_seasonal/         # H3: Seasonal forcing, SEM, entropy diagnostics
  h4_arsenicfm/        # H4: ArseniCFM multi-modal screening model
models/
  arsenicfm_best.pt    # Trained ArseniCFM weights (94K params)
embeddings/
  prithvi_pca64_embeddings.npy      # Prithvi-100M embeddings (1807 x 64)
  sentinel_statistical_embeddings.npy  # Sentinel-2 statistical features (1807 x 64)
  prithvi_pca_model.pkl             # Fitted PCA model (768 -> 64 dim)
figures/
  make_figures.py      # Scripts to reproduce all manuscript figures
```

## Data Availability

The geochemical dataset (1,807 samples x 45 features) is archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

*DOI will be updated upon acceptance.*

## Installation

```bash
git clone https://github.com/rakibhhridoy/bangladesh-groundwater-arsenic.git
cd bangladesh-groundwater-arsenic
pip install -r requirements.txt
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- scikit-learn, shap, libpysal, esda
- earthengine-api (for satellite data extraction)
- See `requirements.txt` for full list

## Reproducing Results

### H1: Geochemical Mechanism Analysis
```bash
python src/h1_geochemistry/main.py
```

### H2: Depth-Stratified Analysis
```bash
python src/h2_depth/main.py
```

### H3: Seasonal Forcing Analysis
```bash
python src/h3_seasonal/main.py
```

### H4: ArseniCFM Model Training and Evaluation
```bash
python src/h4_arsenicfm/main.py
```

### Generate Prithvi-100M Embeddings
```bash
python src/h4_arsenicfm/prithvi_encoder.py --install  # first time only
python src/h4_arsenicfm/prithvi_encoder.py --test
```

## ArseniCFM Architecture

Three-encoder multi-modal fusion model (~94,000 parameters):
- **Satellite encoder:** Prithvi-100M 64-dim PCA embeddings
- **Geochemical encoder:** 18-feature MLP (32-dim output)
- **Context encoder:** 4-feature MLP (depth, season, lat, lon; 16-dim output)
- **Fusion:** Cross-modal attention -> 112-dim representation
- **Outputs:** As concentration (MSE), WHO exceedance (BCE), mechanism class (CE), depth zone (CE)

## Citation

```bibtex
@article{Hasan2026_arsenic,
  title   = {Phosphate Dominance, Geochemical Bistability, and Seasonal Dynamics
             in Bangladesh Groundwater Arsenic: National-Scale Mechanism
             Partitioning and Rapid Risk Screening},
  author  = {Hasan, Md Rakib and Rupa, Mst Anika Khatun and Arabi, Fazla Zawadul
             and Zubyer, Shoumik and Zahid, Anwar},
  journal = {Water Research},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
