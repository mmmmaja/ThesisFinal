- 5 outcomes
- 6 data compositions (single omics, all together, not CSF)
- Keep only MIFS
- 6 ML models (maybe less): Characterizing them and explaining the choices of methods. Why those

**Figures :**
- Workflow of the study
- Supplementary feature selection plot
MAIN : ROC curves per data composition and per outcome
  -> Really explain well in the legend/text
  -> First idea : 6 ROC (per data composition) Only report one ROC plot with the best algorithm per outcome
  -> Supplementary table with all the accuracies/MCC/F1 score per algorithm and we highlight the best performing model represented in the ROC plots
Feature importance for each best model
 -> All OMICS in the main, the rest as supplementary
 -> Needs to be standardized across methods (Shapley values ?)
 -> Heatmap ? See PPMI paper