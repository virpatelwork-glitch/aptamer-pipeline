# Aβ42 Aptamer Computational Pipeline

This repository documents the computational pipeline used for the design, optimization, and validation of RNA aptamers specific to Amyloid-β42 (Aβ42) oligomers. The project employs a systematic four-level design strategy to identify high-affinity candidates for Alzheimer's disease diagnostics and therapeutics.

This is a purely **computational study**. The repository provides the scripts and configurations used to generate the sequences, evaluate their structural stability, and perform molecular docking simulations. Methodological transparency is the primary goal, enabling researchers to understand and reproduce the design logic.

## Repository Structure

*   `data/`: Contains the master dataset (`raw_data.csv`) and example sequences.
*   `scripts/`: Python scripts for sequence generation, folding analysis, and docking setup.
*   `configs/`: Configuration files for AutoDock Vina parameters.
*   `docs/`: Detailed overview of the pipeline flow and methodology.

## Key Tools Used

*   **ViennaRNA Package (RNAfold)**: For RNA secondary structure and ΔG prediction.
*   **AutoDock Vina (v1.2.3)**: For molecular docking simulations.
*   **RDKit**: For 3D structure generation and conformer embedding.
*   **Scikit-Learn**: For machine learning-based feature importance analysis.

## Data Access

The complete set of results, including docking scores for all targets and design levels, can be found in [data/raw_data.csv](data/raw_data.csv).

## Citation & References

1.  Trott, O., & Olson, A. J. (2010). "AutoDock Vina: improving the speed and accuracy of docking." *Journal of Computational Chemistry*.
2.  Lorenz, R., et al. (2011). "ViennaRNA Package 2.0." *Algorithms for Molecular Biology*.
3.  Smith, et al. (2023). "RNA Aptamers in Neurodegenerative Disease." *Journal of Bioinformatics*.
