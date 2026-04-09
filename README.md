# SADM-reproducibility-ARRAY-D-25-02292R1
Reproducibility package for SADM framework validation
# SADM Framework: Reproducibility Package and Supplementary Data

This repository contains the official evaluation scripts, raw generation logs, and statistical validation data supporting the empirical findings of our paper regarding the **SADM (Strategic AI Design Management)** framework. 
To ensure full transparency and reproducibility, we have provided all necessary materials to replicate the Computer Vision (CV) objective metrics and the Expert Panel's subjective Intraclass Correlation Coefficient (ICC) reliability tests discussed in the manuscript.

## 📂 Repository Contents & Paper Mapping
The repository is organized into two main categories: **Objective Algorithm Evaluation (CV Metrics)** and **Subjective Expert Evaluation (Statistical Reliability)**.

### 1. Objective Algorithm Evaluation (Computer Vision Metrics)
These files support the empirical validation of the Iteration-Controller (Section 4.1.2) and the temporal coherence improvements (Table 4, Section 4.3.1).

* **`evaluation_script.md` / `Calculate_LPIPS.ipynb`**
  * **Description:** The core Python scripts/Jupyter Notebook used to calculate the visual similarity and temporal coherence of the dynamic outputs. It includes the implementation for **CLIP (ViT-B/32)**, **SSIM**, and frame-to-frame **LPIPS** (AlexNet).
  * **Paper Reference:** Section 3.3.3 (Evaluation Metrics), Table 4.

* **`CLIP_Similarity_Scores.csv`**
  * **Description:** The complete generation logs and sensitivity analysis data across 10 iterations. This file contains the exact semantic (CLIP) and structural (SSIM) scores used to justify the **80% (0.8) operational threshold** and the **10-round iteration cap**.
  * **Paper Reference:** Section 4.1.2 (Sensitivity Analysis and Parameter Justification).

* **`Calculate LPIPS_runtime.log` & `test_3_runtime.log`**
  * **Description:** Raw console outputs and calculation logs generated during the evaluation process, provided for maximum transparency and computational proof.

### 2. Subjective Expert Evaluation & Statistical Proofs
These files support the inter-rater reliability claims and the subjective superiority of the SADM framework (Section 3.3.3 and Table 4).

* **`Expert Panel Scoring Rubric.csv`**
  * **Description:** The comprehensive 5-point Likert scale utilized by the independent expert panel. It includes all specific qualitative anchors and descriptors for Creativity, Technical Quality, Thematic Relevance, and Stylistic Cohesion.
  * **Paper Reference:** Appendix A, Section 3.3.3.

* **`Raw Data_Expert Evaluation.csv`**
  * **Description:** The foundational dataset containing 50 rows of raw evaluations (10 projects × 5 experts × 4 dimensions). This is the exact dataset used to run the ICC reliability tests.

* **`Summary of Underlying ANOVA for ICC(2,k) Calculation.csv`**
  * **Description:** The underlying Analysis of Variance (ANOVA) metrics ($MS_R$, $MS_C$, $MS_E$) proving the mathematical derivation of the reported ICC scores.
  * **Paper Reference:** Section 3.3.3 (Inter-rater reliability: Overall ICC = 0.76, Creativity = 0.902).

* **`Aggregated Mean Scores by Expert Panel.csv`**
  * **Description:** The final aggregated mean scores comparing the SADM (Experimental) workflow against the Traditional (Control) workflow.
  * **Paper Reference:** Table 4 (Quantitative Analysis Results).

* **`How to Use`**
 * **CV Evaluation:** Open Calculate_LPIPS.ipynb in Jupyter or Google Colab. Ensure your local generated videos/images are placed in the correct target directory as specified in the script, then run the cells to output the Mean ± SD for LPIPS, CLIP, and SSIM.

* **ICC Calculation:** You can use standard statistical software (like SPSS) or the Python pingouin library on Raw Data_Expert Evaluation.csv to reproduce the Two-Way Mixed, Absolute Agreement, Average Measures ICC (ICC 2,k) reported in the paper.

---

## ⚙️ Environment Setup & Dependencies

To run the evaluation scripts (`Calculate_LPIPS.ipynb` or the Python script), please ensure you have the following dependencies installed:

```bash
# Core CV and Machine Learning Libraries
pip install torch torchvision
pip install transformers          # For OpenAI CLIP
pip install lpips                 # For Perceptual Distance
pip install scikit-image opencv-python

# Statistical and Data Processing Libraries (for ICC calculation)
pip install pandas numpy pingouin
