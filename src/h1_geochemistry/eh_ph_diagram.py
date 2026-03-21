"""
PHASE 3B: Eh-pH DIAGRAM
Redox-pH characterization of groundwater

Essential plot for arsenic geochemistry showing:
- Eh (redox potential): Derived from ORP measurements
- pH: Direct measurement
- As(III) and As(V) stability fields: Calculated from thermodynamic data

Key insight: As(III) is more mobile in reducing conditions
             As(V) stable in oxidizing conditions

Eh-pH diagram shows which form dominates where

Output figure:
- Figure_17_eh_ph_diagram.png

Output table:
- Table_25_redox_classification_eh_ph.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from config import TABLES_DIR, FIGURES_DIR, OUTPUT_DIR, DPI, FIGURE_SIZE, DATA_FILE

def convert_orp_to_eh(orp_mv, ag_agcl_ref=True):
    """
    Convert ORP (mV) to Eh (V) using AgAgCl reference electrode

    Standard cell potentials:
    - Ag/AgCl reference: Eh = ORP + 0.197 (vs SHE) [or 0.200 approx]
    - Pt electrode: Eh = ORP directly

    Bangladesh well data likely uses Ag/AgCl, so add ~200 mV offset
    """

    if ag_agcl_ref:
        eh_v = (orp_mv + 200) / 1000  # Convert to V
    else:
        eh_v = orp_mv / 1000

    return eh_v

def add_as_stability_fields(ax):
    """
    Add As(III) and As(V) stability field boundaries to Eh-pH diagram

    Thermodynamic boundaries based on 10^-6 M As (background level)
    pH range: 4-10
    """

    # Approximate Eh values (V) for As transitions at pH 7
    # At pH 7, As(III)/As(V) boundary approximately:
    # Eh = +0.56V - 0.0592*pH = +0.56 - 0.0592*7 ≈ 0.15V

    # More detailed boundaries (simplified, actual depends on As concentration)
    ph_range = np.array([4, 5, 6, 7, 8, 9, 10])

    # As(III)/As(V) boundary (approximate)
    # Reaction: H3AsO4 + 2H+ + 2e- ↔ H3AsO3 (As(V) → As(III))
    # Eh = +0.56 - 0.0592*pH (at T=25°C)
    eh_as_boundary = 0.56 - 0.0592 * ph_range

    # O2/H2O boundary (aerobic/anaerobic)
    # O2 + 4H+ + 4e- ↔ 2H2O
    # Eh = +1.23 - 0.0592*pH
    eh_o2_boundary = 1.23 - 0.0592 * ph_range

    # H+/H2 boundary (strongly reducing)
    # 2H+ + 2e- ↔ H2
    # Eh = 0 - 0.0592*pH
    eh_h2_boundary = -0.0592 * ph_range

    # Add boundaries
    ax.plot(ph_range, eh_as_boundary, 'r--', linewidth=2, label='As(III)/As(V) boundary', alpha=0.7)
    ax.plot(ph_range, eh_o2_boundary, 'b-', linewidth=2, label='O2/H2O (aerobic/anaerobic)', alpha=0.5)
    ax.plot(ph_range, eh_h2_boundary, 'k-', linewidth=2, label='H+/H2 (strongly reducing)', alpha=0.5)

    # Shade As(III) and As(V) regions
    ax.fill_between(ph_range, eh_as_boundary, -0.5, alpha=0.1, color='red', label='As(III) dominant')
    ax.fill_between(ph_range, eh_as_boundary, 1.5, alpha=0.1, color='blue', label='As(V) dominant')

    return None

def classify_redox_ph_type(df):
    """
    Classify samples into Eh-pH regimes for interpretation

    Types:
    - Oxidizing/Oxic: Eh > +0.2V, typically pH 6-8
    - Suboxic: Eh 0 to +0.2V
    - Anaerobic/Reducing: Eh < 0V
    """

    # Convert ORP to Eh
    if 'ORP' in df.columns:
        df['Eh_V'] = df['ORP'].apply(convert_orp_to_eh)
    else:
        df['Eh_V'] = np.nan

    # Classify
    redox_type = []
    for idx in df.index:
        eh = df.loc[idx, 'Eh_V']
        ph = df.loc[idx, 'pH']

        if pd.isna(eh) or pd.isna(ph):
            redox_type.append('Unknown')
        elif eh > 0.3:
            redox_type.append('Oxidizing')
        elif eh > 0.0:
            redox_type.append('Suboxic')
        else:
            redox_type.append('Anaerobic/Reducing')

    df['Redox_Type'] = redox_type

    # As speciation prediction
    as_spec = []
    for idx in df.index:
        eh = df.loc[idx, 'Eh_V']
        ph = df.loc[idx, 'pH']

        if pd.isna(eh) or pd.isna(ph):
            as_spec.append('Unknown')
        else:
            # Boundary: Eh = 0.56 - 0.0592*pH
            eh_boundary = 0.56 - 0.0592 * ph

            if eh > eh_boundary:
                as_spec.append('As(V) dominant')
            else:
                as_spec.append('As(III) dominant')

    df['As_Speciation'] = as_spec

    return df

def run_eh_ph_analysis(df):
    """Main Eh-pH diagram analysis"""

    print("\n" + "="*80)
    print("PHASE 3B: Eh-pH DIAGRAM ANALYSIS")
    print("="*80)

    # Prepare data
    df_ehph = df[['pH', 'ORP', 'As']].dropna()

    print(f"\nAnalyzing {len(df_ehph)} samples with complete Eh-pH data")

    # Convert ORP to Eh
    df_ehph['Eh_V'] = df_ehph['ORP'].apply(convert_orp_to_eh)

    # Classify redox types
    df_classified = classify_redox_ph_type(df[['pH', 'ORP', 'As']])

    # Statistics by redox type
    print("\n1. Redox-pH Classification:")
    print("-"*70)

    for rt in ['Oxidizing', 'Suboxic', 'Anaerobic/Reducing', 'Unknown']:
        subset = df_classified[df_classified['Redox_Type'] == rt]
        if len(subset) > 0:
            print(f"   {rt:25}: {len(subset):5} samples, Mean As={subset['As'].mean():7.2f} µg/L")

    # As speciation prediction
    print("\n2. Predicted As Speciation:")
    print("-"*70)

    for spec in ['As(III) dominant', 'As(V) dominant', 'Unknown']:
        subset = df_classified[df_classified['As_Speciation'] == spec]
        if len(subset) > 0:
            pct = 100 * len(subset) / len(df_classified)
            print(f"   {spec:20}: {len(subset):5} samples ({pct:5.1f}%), Mean As={subset['As'].mean():7.2f} µg/L")

    # Save classification
    df_classified[['pH', 'ORP', 'As', 'Eh_V', 'Redox_Type', 'As_Speciation']].to_csv(
        TABLES_DIR / 'Table_25_redox_classification_eh_ph.csv', index=False
    )
    print(f"\n   Saved: Table_25_redox_classification_eh_ph.csv")

    # Create Eh-pH diagram
    create_eh_ph_figure(df_ehph)

    print("\n" + "="*80)
    print("✓ Eh-pH DIAGRAM ANALYSIS COMPLETE")
    print("="*80)

    return df_classified

def create_eh_ph_figure(df_ehph):
    """Create publication-quality Eh-pH diagram"""

    print("\n3. Creating Eh-pH diagram...")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Main scatter plot: pH vs Eh, colored by As
    scatter = ax.scatter(df_ehph['pH'], df_ehph['Eh_V'],
                        c=df_ehph['As'], cmap='YlOrRd', s=50,
                        alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add stability field boundaries
    add_as_stability_fields(ax)

    # Formatting
    ax.set_xlabel('pH', fontweight='bold', fontsize=12)
    ax.set_ylabel('Eh (Volts)', fontweight='bold', fontsize=12)
    ax.set_title('Eh-pH Diagram for Groundwater Arsenic\nStability Fields for As(III) and As(V)',
                fontweight='bold', fontsize=13)
    ax.set_xlim([3, 11])
    ax.set_ylim([-0.6, 1.2])
    ax.grid(alpha=0.3)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=10)

    # Color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Arsenic (µg/L)', fontweight='bold')

    # Annotations
    ax.text(3.5, 1.0, 'OXIDIZING\n(As V)', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(3.5, -0.4, 'REDUCING\n(As III)', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Figure_17_eh_ph_diagram.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   Saved: Figure_17_eh_ph_diagram.png")

if __name__ == "__main__":
    print("Loading groundwater data...")
    df = pd.read_csv(DATA_FILE)

    df_classified = run_eh_ph_analysis(df)
    print("\nEh-pH diagram analysis complete.")
