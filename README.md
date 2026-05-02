# TAILGR: Tail-Aware Local Graph Refinement for Protein Function Prediction


## Data Preparation
====================

Download the dataset from:
https://drive.google.com/drive/folders/1hnx75yGfrSdFL5ERxTfEzUVJ6F724ZJp?usp=drive_link

After downloading, unzip all files and place them under a unified "data" folder:

project_root/
    data/
        ...


EC Dataset Construction
====================

Run the following scripts in order:

python build_ec_mapping.py

python build_ec_dataset.py

python build_ec_directed_spectral_embedding.py

This will generate:
- EC dataset
- EC directed spectral embedding


GO Embedding Construction
====================

Run:

python build_go_directed_spectral_embedding.py

Modify the parameters in the script to process different GO sub-ontologies:
- MF (Molecular Function)
- BP (Biological Process)
- CC (Cellular Component)


Model Training
====================

Baseline models:

Run: 
baseline_evaluation_withEC.ipynb

TAILGR model:

Run:
Model_TAILGR.ipynb


Result Analysis
====================

Run:
predict_result_analysis.ipynb


Notes
====================

- Make sure all data files are placed under the "data" folder before running any scripts.
- GO embeddings should be generated separately for MF, BP, and CC.
- EC and GO datasets are processed independently.
