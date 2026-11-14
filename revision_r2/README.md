# Revision analyses (GSD-D-26-01044, major revision)

Scripts for the analyses added or recomputed in the revised manuscript
*Depositional Environment and Depth-Stratified Geochemical Association of Arsenic
and Phosphate in Bangladesh Groundwater* (Groundwater for Sustainable Development,
under review). They supersede the corresponding parts of `src/` where the two differ.

## Changes from the earlier analysis

- **Wells are identified by BWDB station identifier.** The earlier code grouped
  samples by exact coordinates. All cluster bootstraps and grouped
  cross-validation now resample whole wells using `r01c_station_wellkey.py`.
- **The crossover depth is withdrawn.** `src/h2_depth/mechanism_shift.py` reports
  a single phosphate-to-iron crossover depth. The revised paper withdraws it,
  because its location moves between 63 and 151 m with the analysis settings
  (`r07_crossover_sensitivity.py`). Do not use that script's crossover depth.

## Scripts and what they reproduce

Run them in this order from the repository root. Outputs go to
`revision_r2/output/`.

| Script | Manuscript / Supplement |
|---|---|
| `analysis/r01c_station_wellkey.py` | Well key (station identifier) used by r02–r07 |
| `analysis/r01b_station_id.py` | Paired and single-campaign wells (878 / 28 / 18), inconsistent records; Section 2.1, Table S17 |
| `analysis/r02_geology_vs_depth.py` | Deposit type against depth on equal terms, shallow-well test, depth-matched pairs, Mantel–Haenszel; Section 3.2, Tables S7, S11 |
| `analysis/r03_depthwise_and_crossover.py` | Depth-wise As–PO4 and As–Fe correlations with well-bootstrap CIs, alternative depth schemes, physiographic units, BGS comparison; Table 4, Tables S8, S10, Section 3.6 |
| `analysis/bgs_replication.py` | Loader for the DPHE/BGS (2001) survey, imported by r03 |
| `analysis/r04_saturation_indices.py` | PHREEQC (wateq4f) saturation indices, association by siderite state; Section 3.3.2, Tables S12, S13 |
| `analysis/r05_phosphate_method.py` | Phosphate method checks (arsenate interference, redox structure, repeatability, Hoque 2010 comparison); Section 3.9 |
| `analysis/r06_gb_validation.py` | Gradient boosting under random, by-well and spatial-block folds; within-class models and SHAP rank stability; Tables S14, S15 |
| `analysis/r07_crossover_sensitivity.py` | Settings grid for the withdrawn crossover depth; Table S9 |
| `analysis/r08_boundary_distance.py` | Distance from each well to the nearest mapped unit boundary; Table S16 |
| `analysis/r09_supp_tables.py` | Writes Supplementary Tables S6–S18 as LaTeX from the outputs above |
| `analysis/r10_digitise_logs.py` | Digitises the 69 BWDB exploratory borehole logs (BWDB 2023, pdf pp. 40–50) into layer tables: depth scale from OCR of the axis labels, lithology from the legend colours; Section 2.3 |
| `analysis/r10c_validate_logs.py` | Checks the digitised logs against the aquifer boundaries stated in the report text at seven sites; Table S19 |
| `analysis/r11_borehole_logs.py` | Matches nested wells to the logs, places each well in the first aquifer or beneath an aquitard, and repeats the deposit-type comparison on the logged wells; Section 3.2, Table S19 |
| `analysis/r10b_check_sheet.py` | Check sheets: each original log beside its digitised column (all 69, and a 10-log sample for manual checking) |
| `figures/make_figure1_r2.py` | Fig. 1 (study area and interpolated arsenic surface) |
| `figures/make_figure_deposit_map.py` | Fig. 3 (arsenic and phosphate on the depositional map) |
| `figures/make_figure5_r2.py` | Fig. 6 (continuous depth, no crossover, bootstrap bands); needs the H2 tables from `src/h2_depth/main.py` and `lib/make_figures_hq.py` |

`expected/` holds the summary files from the authors' run, so a rerun can be
checked line by line against them.

Run the borehole-log scripts in the order r10, r10c, r11, r10b. `r10` needs
`pdfimages` (poppler) and `tesseract` on the path. `expected/r10_log_layers.csv`
and `expected/r10_log_meta.csv` are the digitised logs from the authors' run; copy
them to `revision_r2/output/` to run r10c and r11 without digitising.

## Inputs

Paths are relative to the repository root, or to `$PAPER1_ROOT` if that is set.

| Path | Source |
|---|---|
| `data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv` | Zenodo, https://doi.org/10.5281/zenodo.19148957 |
| `data/raw/usgs_ofr97470h/geo8bg.shp` | USGS surface geology of Bangladesh, https://doi.org/10.3133/ofr97470H |
| `data/external/BGS_DPHE_2001/BGS_NHS_national_3534.csv`, `BGS_special_study_areas_271.csv` | DPHE/BGS (2001) National Hydrochemical Survey, https://www.bgs.ac.uk/groundwater/quality/groundwater-and-health/arsenic-contamination-of-groundwater/bangladesh-reports |
| `data/raw/shapefiles/*.geojson` | National boundary, divisions, rivers and neighbouring countries, used for map figures only |
| `data/external/BWDB_2023/16594.pdf` (or `$BWDB_PDF`) | BWDB (2023) Report on Groundwater Quality Assessment of Bangladesh Down to the Depth of 335 Meters, https://www.bwdb.gov.bd/archive/pdf/16594.pdf; borehole logs for r10 |
| `revision_r2/inputs/wells_with_usgs_geology.csv` | Samples joined to the USGS deposit polygons (included) |
| `revision_r2/inputs/hoque2010_A5_1_hydrochem.csv` | Hydrochemistry of 74 wells transcribed from Hoque (2010, PhD thesis, University College London) (included) |

## Requirements

On top of `requirements.txt`: `geopandas`, `shapely`, `phreeqpython`, `statsmodels` and `Pillow`;
`pdfimages` (poppler) and `tesseract` for r10.
