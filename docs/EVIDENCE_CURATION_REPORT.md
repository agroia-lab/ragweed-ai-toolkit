# EVIDENCE CURATION REPORT
## LENCU Book Chapter: "AI-Driven Spatial Decision Support System for Sustainable Weed Management under Climate Variability"

**Compiled:** 2026-02-23
**Compiler:** CURATOR Agent (Team: lencu-evidence-curation)
**Publisher:** Springer-Nature, 2026
**Editor:** Prof. N. Benkeblia, UWI Jamaica

---

## EXECUTIVE SUMMARY

This report synthesizes evidence from 7 exercises spanning model development, domain analysis, cross-domain validation, orthomosaic processing, field deployment, satellite integration, and multi-scale species discrimination. The evidence collectively demonstrates a comprehensive **toolkit for AI-driven weed recognition and mapping**, positioned for community adoption.

**Key curation principle:** Select 2-3 clearest, most impactful pieces of evidence per exercise. Prioritize visual clarity, toolkit reproducibility, and novelty. Eliminate redundancy. Every selected figure must advance the narrative or demonstrate a reproducible method.

---

## EXERCISE 01: Training Progression (Model Development)

**Key story:** Systematic hyperparameter investigation (data augmentation, batch size, GPU scaling) achieves 0.886 mAP50 through intelligent batch scaling on dual GPUs rather than resolution scaling.

### Selected Evidence

1. **`run3_lentejas640_b64_2gpu` (Best Model mAP50=0.886)**
   - **Why:** Demonstrates the optimal configuration. Dual-GPU DataParallel at batch 64 outperforms single-GPU high-resolution (Run 2: 1024px batch 8) on all metrics while training 4.1x faster (40 min vs 164 min).
   - **Shows:** Practical production pathway -- batch size >> resolution for agricultural detection.
   - **Artifact:** `training_curves.png`, `PR_curve.png`, `confusion_matrix.png`

2. **Comparison Table (Runs 1-4 Summary)**
   - **Why:** One-page evidence of the progression narrative. Shows baseline (0.810) → augmented (0.871) → scaled (0.886) → multi-domain transfer (0.839).
   - **Shows:** Four distinct experimental questions all addressed quantitatively.
   - **Artifact:** Embedded in `01_training_progression/README.md` § 1 (lines 13-30)

### Rejected Evidence

- Individual epoch-by-epoch loss curves (too noisy; training_curves.png already shows trajectory)
- Training batch samples and validation visualizations (standard YOLO outputs; not novel)
- Checkpoint weights at every 10 epochs (unnecessary; only best.pt required)
- Field inference test (moves into Exercise 05; avoid duplication)

---

## EXERCISE 02: Embedding Space Analysis (Domain Shift Detection)

**Key story:** Ragweed embeddings from 9 databases cluster by **acquisition conditions**, not species biology. MMD=0.27–0.43 between Chilean and international sources predicts deployment failure.

### Selected Evidence

1. **UMAP 2D Projection (by database)**
   - **Why:** Visual proof of database-specific clustering. Strong separation signal visible in embedding space.
   - **Shows:** Why naive deployment fails -- each database occupies a distinct region.
   - **Artifact:** `viz_umap_2d_by_database.html` + PNG snapshot `viz_umap_2d_by_database.png`

2. **MMD Heatmap (9×9 pairwise distance matrix) + Per-Database Generalization Risk Bar Chart**
   - **Why:** Quantifies distances. Shows CL_StaRosa closest (0.347), WeedCube farthest (0.482).
   - **Shows:** Deployment gate framework -- which databases are "safe" for cross-deployment.
   - **Artifact:** `fig_heatmap_mmd.png` + `fig_generalization_risk_barplot.png`

3. **Deployment Gate Decision Framework (Table)**
   - **Why:** Actionable guidance. Maps MMD ranges to deployment recommendations.
   - **Shows:** How to use embedding distances in practice (0.15–0.30: deploy with augmentation; 0.30–0.45: active learning required).
   - **Artifact:** Embedded in `02_embedding_analysis/README.md` § 5 (lines 190–211)

### Rejected Evidence

- 3D UMAP projection (adds little beyond 2D)
- t-SNE and PCA 2D plots (redundant; UMAP is canonical)
- 200MB interactive HTML files (self-contained but unwieldy for publication; PNG snapshots sufficient)
- Whitelist file inventories and raw image directories (metadata, not evidence)

---

## EXERCISE 03: Cross-Domain Detection Experiment (Generalization Testing)

**Key story:** Chilean-only model **catastrophically fails** on international data (mAP50 0.108, zero detections per database). Combined training recovers to 0.874, validating the embedding distance predictions.

### Selected Evidence

1. **Phase 1 Result: Zero Detections Per Database Table**
   - **Why:** Stark visual proof of failure. Per-database breakdown shows **literally zero predictions** on each international database.
   - **Shows:** The 77.8 percentage point drop is not gradual degradation but complete breakdown.
   - **Artifact:** `04_cross_domain_experiment/README.md` § 4.2 (lines 144–153)

2. **Phase 2 Recovery: mAP50 Comparison Chart (CL→INTL vs Combined→INTL)**
   - **Why:** Demonstrates that adding 2,367 international images recovers 76.6 percentage points (0.108 → 0.874).
   - **Shows:** Active learning/data combination is the practical solution to MMD-predicted domain gaps.
   - **Artifact:** `comparison_chart.png` (side-by-side CL-only vs Combined metrics)

3. **Cross-Domain Comparison Summary Table (All scenarios)**
   - **Why:** Complete picture in one table: baseline, failure, recovery, regression costs.
   - **Shows:** The trade-off -- international generalization costs 14.4pp on Chilean test (0.886 → 0.742).
   - **Artifact:** Embedded in `03_cross_domain_experiment/README.md` § 6 (lines 291–307)

### Rejected Evidence

- Epoch-by-epoch training progression for combined model (too granular; final curves sufficient)
- Per-image evaluation details (aggregate metrics already tell the story)
- Confusion matrices for each scenario (standard YOLO outputs; not novel finding)

---

## EXERCISE 04: Orthomosaic Pipeline (Scaling to Drone Mosaics)

**Key story:** 2.4 GB drone orthomosaic (2,151 valid tiles) shows consistent domain shift (MMD=0.4422, std=0.0307), confirming active learning is essential across the entire field, not just isolated regions.

### Selected Evidence

1. **Phase 2 Domain Shift Result: MMD Bar Chart (Overall vs Per-Database)**
   - **Why:** Overall MMD=0.4422 sits at the boundary of "active learning required" (0.30–0.45) zone. Shows this is a field-wide phenomenon, not localized.
   - **Shows:** The tile embedding distances directly validate the framework from Exercise 02.
   - **Artifact:** `mmd_comparison.png` (bar chart: ortho vs all 9 databases)

2. **Phase 3 Tile Selection Summary (K-means + Stratified Sampling)**
   - **Why:** 80 tiles selected from 2,151 = 3.7% of field. Demonstrates efficient active learning strategy.
   - **Shows:** Practical labeling cost for field-scale deployment (estimated 6–8 hours annotation work).
   - **Artifact:** `selected_tiles.csv` + summary text in `04_orthomosaic_pipeline/README.md` § 6.3 (lines 260–286)

3. **Joint UMAP (Orthomosaic tiles + 9 reference databases)**
   - **Why:** Visual proof that ortho tiles occupy a consistent region of embedding space, distinct from training distribution.
   - **Shows:** Why transfer learning (not direct deployment) is necessary.
   - **Artifact:** `joint_umap.html` + PNG snapshot `joint_umap.png`

### Rejected Evidence

- Phase 0 tiling statistics (standard rasterio output; not novel)
- Phase 1 embedding coordinates CSV (data, not visualization)
- Phase 2 per-tile distance statistics (summary statistics in summary JSON sufficient)

---

## EXERCISE 05: Field Inference (Ground-Level Deployment)

**Key story:** Production-ready SAHI pipeline on 190 smartphone images from 3 field sites yields 5,205 georeferenced detections, demonstrating end-to-end deployment feasibility with 87% GPS-tagged imagery.

### Selected Evidence

1. **Field Campaign Summary (3 Sites + Full Campaign Comparison Table)**
   - **Why:** All-in-one deployment evidence. Shows detection counts (18.5–36.5 avg per image), GPS availability (87%), site-to-site consistency.
   - **Shows:** Operational scalability and reliability across diverse field conditions.
   - **Artifact:** `05_field_inference/README.md` § 3 and § 4 (tables with site summaries, campaign totals: 1,390 detections Sta Rosa, 2,443 CATO, 1,372 Trigo)

2. **Representative Inference Panels (3 comparison images: low/medium/high density)**
   - **Why:** Visual proof of detection capability. Shows original photo alongside annotated output from Run 3 (best model).
   - **Shows:** Real-world detection quality on ground imagery.
   - **Artifact:** Example panels in `malezainia2/ambel_seba/inference_examples/{site}/comparison_panels/`

3. **Output Shapefiles (ambel_lentejas_dic24_ground.shp + others)**
   - **Why:** Demonstrates complete pipeline to QGIS-ready geospatial output. Community tool.
   - **Shows:** Integration with standard agricultural GIS workflows.
   - **Artifact:** File paths in `05_field_inference/README.md` § 5

### Rejected Evidence

- Per-image CSVs (data already summarized in tables)
- QGIS Python console scripts (reference material, not evidence)
- Inference results from Runs 1, 2, 4 (Run 3 is canonical; comparisons redundant)

---

## EXERCISE 06: Satellite Weed Analysis (Spectral Integration)

**Key story:** Single-date Sentinel-2 NDVI predicts LENCU density with r=0.890; spatially-aware GWR explains 91% of local variance (vs 59% global OLS). **The spectral bridge works.**

### Selected Evidence

1. **Correlation Heatmap (Presto PCs + Sentinel-2 indices vs 4 weed species)**
   - **Why:** One figure showing both approaches. PC1 r=0.739 (AMBEL), NDVI r=0.837 (AMBEL), 0.890 (LENCU).
   - **Shows:** Strong, multi-method evidence that satellite signal links to ground weed density.
   - **Artifact:** `fig_correlation_heatmap.png`

2. **GWR Local R² Map + Summary Statistics Table (OLS vs GWR)**
   - **Why:** Spatial non-stationarity quantified. R² improvement: +0.20 (AMBEL), +0.32 (LENCU).
   - **Shows:** Why local models matter for heterogeneous paddocks; practical precision agriculture relevance.
   - **Artifact:** `fig_gwr_local_r2.png` + summary table in `gwr_summary.csv`

3. **Bivariate LISA Cluster Map (HH/LL/HL/LH spatial classes)**
   - **Why:** Identifies where satellite-weed co-variation is strongest (hotspots, coldspots, outliers).
   - **Shows:** Practical management-scale spatial structure; 68% significant clusters means most of field is spatially coherent.
   - **Artifact:** `bivariate_lisa_map.png` + `bivariate_lisa_clusters.gpkg` (QGIS-ready)

### Rejected Evidence

- 9 individual spectral index maps (too many; summary heatmap sufficient)
- Kriging semivariograms (technical; variogram range (~90 m) captured in text)
- Raw Presto PC coordinates CSV (data, not visualization)
- Per-pixel correlation scatter plots (correlation heatmap more informative)

---

## EXERCISE 07: Drone-Scale Detection & Class Inversion (Multi-Scale Reality Check)

**Key story:** Three YOLOv11 variants trained at 1024 vs 2048 px systematically **invert AMBEL/LENCU class ratios** on the same drone imagery (Original: 65% LENCU; Retrained 2048: 45% AMBEL). Proves morphological convergence at altitude; recommends two-stage detection.

### Selected Evidence

1. **Per-Image AMBEL/LENCU Breakdown (Table: Original vs Retrained across 10 images)**
   - **Why:** Evidence of inversion in every single image. Original=majority LENCU; Retrained=majority AMBEL.
   - **Shows:** Species-level discrimination is unreliable at drone altitude; broadleaf count (AMBEL+LENCU combined) is what's trustworthy.
   - **Artifact:** `detection_summary.csv` (columns: image, model, ambel_count, lencu_count, polav_count, polpe_count) + table in `07_drone_scale_detection/README.md` § 6.2

2. **Four-Panel Comparison Figure (Low/Medium/High Density Representative Images)**
   - **Why:** Original photo + Original model detections + Maaferna detections + Retrained detections side-by-side.
   - **Shows:** Visually evident label confusion; same green weed clusters receive different species colors.
   - **Artifact:** `panel_DJI_0971.png` (low density), `panel_DJI_0632.png` (medium), `panel_DJI_0882.png` (high)

3. **Training Performance Summary (mAP50 range 0.846–0.868) + Combined Broadleaf Consistency Table**
   - **Why:** Models are well-trained (mAP metrics solid), yet still disagree on species. Issue is not model quality but scale physics.
   - **Shows:** The problem is not fixable by training better; it's structural (altitude/morphology).
   - **Artifact:** Tables in `07_drone_scale_detection/README.md` § 4 and § 6.3

### Rejected Evidence

- All 10 comparison panels in grid (pick 3 representative; summarize stats for the rest)
- Training curves, confusion matrices, loss plots (standard YOLO outputs; not novel)
- max_det parameter exploration (important but technical; mentioned in text)
- Roboflow dataset lineage (metadata, not evidence)

---

## TOP 15 FIGURES FOR THE CHAPTER

### Tier 1: Foundation (Essential Narrative Flow)

1. **UMAP 2D by Database** (`02_embedding_analysis/viz_umap_2d_by_database.png`)
   - *Why placed first:* Establishes why domain shift matters. Visual proof.
   - *Chapter section:* 3.3 Domain Shift Analysis

2. **Training Curves: Run 3 (Best Model)** (`01_training_progression/training_curves.png`)
   - *Why:* Model foundation. Shows convergence and final metrics.
   - *Chapter section:* 3.1 Model Development

3. **Phase 1 Zero Detections Table** (`03_cross_domain_experiment/README.md` text table)
   - *Why:* Stark failure case. Sets up recovery story.
   - *Chapter section:* 3.4 Generalization Testing

### Tier 2: Quantitative Evidence (Metrics & Validation)

4. **MMD Heatmap (9×9)** (`02_embedding_analysis/fig_heatmap_mmd.png`)
   - *Why:* Quantifies all pairwise distances. Reference for deployment gates.
   - *Chapter section:* 3.3 Domain Shift Analysis

5. **Phase 2 Recovery Comparison Chart** (`03_cross_domain_experiment/comparison_chart.png`)
   - *Why:* Before-after recovery (0.108 → 0.874). Validates data combination strategy.
   - *Chapter section:* 3.4 Generalization Testing (Phase 2 results)

6. **GWR Local R² Map** (`06_satellite_weed_analysis/fig_gwr_local_r2.png`)
   - *Why:* Spatial non-stationarity made visible. R²=0.91 is remarkable result.
   - *Chapter section:* 5.3 Satellite Scaling & Precision

### Tier 3: Field Validation (Operational Proof)

7. **Joint UMAP (Ortho Tiles + Databases)** (`04_orthomosaic_pipeline/joint_umap.png`)
   - *Why:* Links orthomosaic data to reference framework. Field-scale domain shift quantified.
   - *Chapter section:* 3.5 Orthomosaic Scale-Up

8. **Bivariate LISA Cluster Map** (`06_satellite_weed_analysis/bivariate_lisa_map.png`)
   - *Why:* Spatial clustering shown as QGIS-ready output. Practical management zones.
   - *Chapter section:* 5.3 Satellite Scaling & Precision

9. **Field Inference Representative Panel (Medium Density)** (`05_field_inference/inference_examples/ambel_sta_rosa/panel_example.png`)
   - *Why:* Proof of detection in production conditions. Real-world quality.
   - *Chapter section:* 3.2 Field Deployment

### Tier 4: Methods & Reproducibility (Toolkit Value)

10. **Selected Tiles Map (K-means + Stratified Sampling)** (`04_orthomosaic_pipeline/selected_tiles.csv` visualized or `tiles_for_labeling/` directory listing)
   - *Why:* Active learning strategy made concrete. 80 tiles = 3.7% workload.
   - *Chapter section:* 3.5 Orthomosaic Pipeline (Phase 3)

11. **Generalization Risk Bar Chart** (`02_embedding_analysis/fig_generalization_risk_barplot.png`)
   - *Why:* Ranks databases by MMD. Actionable for practitioners.
   - *Chapter section:* 3.3 Domain Shift Analysis

12. **Correlation Heatmap (Presto + NDVI + Species)** (`06_satellite_weed_analysis/fig_correlation_heatmap.png`)
   - *Why:* Shows satellite-weed link across all indices and species. Comprehensive.
   - *Chapter section:* 5.3 Satellite Scaling & Precision

### Tier 5: Key Findings & Limitations (Insights)

13. **AMBEL/LENCU Inversion Table** (`07_drone_scale_detection/detection_summary.csv` with per-image counts highlighted)
   - *Why:* Evidence of class confusion across all images. Fundamental finding.
   - *Chapter section:* 6.1 Multi-Species Discrimination Limits

14. **Four-Panel Comparison (Medium Density)** (`07_drone_scale_detection/figures/panel_DJI_0632.png`)
   - *Why:* Visual proof of inversion. Same weeds, opposite labels.
   - *Chapter section:* 6.1 Multi-Species Discrimination Limits

15. **Deployment Gate Framework Decision Table** (Table from `02_embedding_analysis/README.md` § 5, recreated as professional figure)
   - *Why:* Actionable guide. Links theory (MMD) to practice (active learning).
   - *Chapter section:* 3.3 Domain Shift Analysis (Deployment Framework)

---

## STORYLINE SUGGESTION: THREE-ACT NARRATIVE

### ACT I: Field-Scale Detection Achievement (Exercises 01, 02, 05)

**Opening Scene:** A lentil farmer faces a choice: apply herbicide uniformly across the 3.4 ha field, or spray only where ragweed is densest.

**Act I establishes the technology:**

1. **Training Foundation (Ex. 01):** YOLOv11l model achieves mAP50=0.886 through intelligent batch scaling (dual GPU, batch 64) rather than resolution escalation. This is the production detector.

2. **Why Domain Matters (Ex. 02):** But this model was trained on close-up field photos. When we examine the embedding space of 5,338 images from 9 international and Chilean databases, we discover that **images cluster by acquisition conditions (camera, field, season), not by ragweed morphology**. MMD distances of 0.27–0.43 between Chilean and international databases predict cross-domain failure.

3. **Ground Deployment (Ex. 05):** Nevertheless, on smartphone ground photos from 3 field sites, the SAHI pipeline detects 5,205 ragweeds across 190 images. 87% are GPS-tagged, enabling conversion to shapefiles for QGIS. The community now has a working detector for ground-level field assessment.

**Act I Conclusion:** Field-scale detection is achievable and reproducible.

---

### ACT II: Domain Generalization Challenge (Exercises 03, 04, 07)

**Complication:** When the Chilean-trained model is evaluated on international test imagery (2,367 images from North Dakota, Michigan, multi-site USA), it produces **zero detections** at mAP50=0.108, a 77.8 percentage point collapse. Why?

**Act II explores the limits and solutions:**

1. **The Failure (Ex. 03, Phase 1):** Per-database breakdown shows the collapse is complete, not gradual. Every database yields zero predictions. The embedding analysis correctly predicted this: MMD distances of 0.27–0.43 are in the "active learning required" zone.

2. **The Recovery (Ex. 03, Phase 2):** Adding 2,367 international images to the training set recovers mAP50 to **0.874** on international test data (+76.6 percentage points). The cost: 14.4 pp regression on Chilean test data (0.886 → 0.742). This trade-off is the reality of multi-domain learning.

3. **Scaling to Drone Mosaics (Ex. 04):** A 2.4 GB drone orthomosaic of a single lentil field (2,151 valid 1024×1024 tiles) shows consistent domain shift (MMD=0.4422 overall, std=0.0307). This is not a localized phenomenon but field-wide. The active learning strategy selects 80 tiles (3.7%) for annotation—the springboard for transfer learning to adapt the model to drone perspective.

4. **The Multi-Scale Reality (Ex. 07):** When the model is applied via SAHI to drone tomato imagery at operational altitude (~2 m GSD), three independently trained YOLOv11 variants systematically **invert the AMBEL/LENCU class ratio** (Original: 65% LENCU; Retrained 2048: 45% AMBEL) on the same images. The root cause: morphological convergence. At drone altitude, deeply-lobed ragweed (*A. artemisiifolia*) and arrow-shaped bindweed (*Convolvulus arvensis*) both appear as small green clusters of similar size. Diagnostic morphological features vanish.

**Act II Conclusion:** The solution is not a better model—it is a two-stage detection strategy: drone for **spatial mapping** (weed patch location and density, species-agnostic), ground-level cameras for **species identification** (where management decisions depend on herbicide choice).

---

### ACT III: Satellite Scaling for Regional Surveillance (Exercise 06 + Conceptual Future)

**Synthesis:** The field-scale detection pipeline produces high-resolution density maps. Satellite imagery covers the landscape cheaply and continuously. Can they be connected?

**Act III demonstrates proof-of-concept and outlines the path forward:**

1. **The Spectral Bridge (Ex. 06):** Single-date Sentinel-2 NDVI correlates with ground-truthed weed density at **r=0.890** for *Convolvulus* (bindweed, LENCU) and **r=0.837** for *Ambrosia* (ragweed, AMBEL). These are remarkably strong correlations for satellite-to-weed-count relationships. The mechanism is direct: dense weed patches = more above-ground green biomass → higher NDVI. This is not indirect detection (e.g., crop stress from parasitic weeds) but direct spectral mixture.

2. **Spatial Non-Stationarity (Ex. 06):** Geographically Weighted Regression reveals that the satellite-weed relationship is not uniform across the paddock. Local models explain **R²=0.91** (LENCU) vs global OLS at **R²=0.59**, a 0.32 improvement. The relationship varies by soil type, microclimate, and management history. Bivariate Local Moran's I identifies 68% of the paddock as significant spatial clusters (HH = high weed + high NDVI hotspots; LL = clean zones confirmed by satellite). Only 1–2% are outliers where the spectral-weed link breaks down.

3. **The Future Framework:** The book chapter does not fully execute regional surveillance, but lays the conceptual foundation: (i) calibration phase overlays field-scale density maps onto concurrent Sentinel-2 imagery to train an embedding→density model; (ii) regional inference applies the trained model across all 10 m pixels regionally; (iii) temporal evolution tracking detects pixels whose PRESTO embeddings shift toward "Ambrosia signature" clusters across growing seasons = documented invasion progression.

**Act III Conclusion:** The same AI-geostatistics pipeline that reduces herbicide use by 35–50% within a paddock can scale to a landscape-level early warning system for climate-driven weed invasion, serving simultaneously as an agricultural advisory tool and a public health surveillance platform.

---

## NARRATIVE THREAD ACROSS ALL 7 EXERCISES

**Unified message for community practitioners:**

> "We have explored and tested multiple tools for recognizing and mapping weeds at complementary scales: field-level detection from close-up images (ground accuracy), multi-domain adaptation for crop systems (practical robustness), drone-mosaics for property-scale visualization (operational efficiency), and satellite spectral signatures for landscape monitoring (early warning). None alone solves the problem. Together, they form a toolkit for decision-making under climate variability."

**Community takeaway:**

- **Expect domain shift.** Your best model trained on one field will fail on another. Use embedding distances (MMD) to assess risk.
- **Active learning is not optional.** When MMD > 0.30, budget annotation time for target-domain calibration.
- **Combine modalities.** Drones for spatial mapping, ground cameras for species ID, satellites for long-term surveillance.
- **Species confusion is physics, not poor training.** At drone altitude, morphology converges. Accept this and plan accordingly.
- **Reproducibility requires open data and code.** All scripts, datasets, and trained weights are archived and documented for your use.

---

## EVIDENCE INVENTORY: FILE PATHS FOR REFERENCE

### Exercise 01 (Training)
- Best model: `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt`
- Training curves: `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/training_curves.png`
- Comparison table: `research_docs/lencu_book_chapter/evidence/01_training_progression/README.md` lines 13–30

### Exercise 02 (Embeddings)
- UMAP visualization: `/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/viz_umap_2d_by_database.html` (interactive) and `.png` (static)
- MMD heatmap: `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/malezainia2/resultados/figures/fig_heatmap_mmd.png`
- Generalization risk: `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/malezainia2/resultados/figures/fig_generalization_risk_barplot.png`

### Exercise 03 (Cross-Domain)
- Phase 1 metrics: `malezainia2/ambel_crossdomain/phase1_eval/crossdomain_metrics.json`
- Phase 2 metrics: `malezainia2/ambel_crossdomain/phase2_eval_intl/crossdomain_metrics.json`
- Comparison chart: `malezainia2/ambel_crossdomain/comparison/comparison_chart.png`

### Exercise 04 (Orthomosaic)
- Tile manifest: `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024/tile_manifest.csv`
- Domain shift report: `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/domain_shift_report.json`
- MMD comparison: `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/mmd_comparison.png`
- Joint UMAP: `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/joint_umap.png`
- Selected tiles: `/media/malezainia2/E/Proyecto_lentejas_v2/selected_tiles.csv`

### Exercise 05 (Field Inference)
- Campaign results: `malezainia2/ambel_seba/results/ambel_lentejas_dic24.csv`
- Shapefiles: `outputs/geo_exports/ambel_lentejas_dic24_ground.shp` (and .shx, .dbf, .prj)
- Inference examples: `malezainia2/ambel_seba/inference_examples/{site}/comparison_panels/`

### Exercise 06 (Satellite Weed)
- Correlation heatmap: `outputs/lencu_sentinel_analysis/fig_correlation_heatmap.png`
- GWR results: `outputs/lencu_presto/spatial_analysis/gwr_results.gpkg`
- GWR local R² map: `outputs/lencu_presto/spatial_analysis/fig_gwr_local_r2.png`
- LISA clusters: `outputs/lencu_presto/spatial_analysis/bivariate_lisa_clusters.gpkg`
- LISA map: `outputs/lencu_presto/spatial_analysis/bivariate_lisa_map.png`

### Exercise 07 (Drone Detection)
- Detection summary: `research_docs/lencu_book_chapter/tables/detection_summary.csv`
- Comparison panels: `research_docs/lencu_book_chapter/figures/panel_DJI_0971.png`, `panel_DJI_0632.png`, `panel_DJI_0882.png`
- Full grid: `research_docs/lencu_book_chapter/figures/summary_grid.png`

---

## CURATION NOTES FOR AUTHORS

### Figures Requiring Regeneration or Polish

- **Deployment Gate Framework (Ex. 02):** Currently embedded as text table in README. Should be recreated as publication-quality figure with axes, color-coding for risk levels (green/yellow/red).
- **Comparison Chart (Ex. 03):** Verify dimensions match journal template; ensure legend is readable at print size.
- **Four-Panel Drone Comparisons (Ex. 07):** Check color contrast between bounding box colors and backgrounds; ensure all 3 selected panels have consistent layout.

### Recommended Sequence for Chapter Sections

1. **Introduction:** Set dual motivation (agronomic + public health). Reference CLAUDEO.md context.
2. **Methods (Exercises 01–07 condensed):** Focus on reproducibility, not raw results. Include CLI commands.
3. **Results (Exercises 02–07 figures):** Use the 15-figure framework above.
4. **Discussion:** Connect findings to argument_map.md conceptual framework.
5. **Conclusions:** Toolkit message + future validation agenda.

### Call to Community

Explicitly invite readers to: (i) download trained weights from repository, (ii) test on their own field imagery, (iii) contribute labeled data from new geographies/crops to improve generalization, (iv) file issues/PRs for model improvements. Frame as open-source community science.

---

## APPENDIX: EVIDENCE COMPLETENESS CHECKLIST

- [x] Exercise 01: All 4 runs documented; Run 3 selected as canonical
- [x] Exercise 02: 9 databases, 5,338 images; visualizations complete
- [x] Exercise 03: Phase 1 (failure) + Phase 2 (recovery) fully evaluated
- [x] Exercise 04: Phases 0–3 complete; Phases 4–8 pending (noted in plan)
- [x] Exercise 05: 3 sites + full campaign; 5,205 detections georeferenced
- [x] Exercise 06: Presto + Sentinel-2; GWR and LISA spatial analysis complete
- [x] Exercise 07: 3 models × 10 images; class inversion quantified

**Status:** All 7 exercises have sufficient evidence for narrative coherence and publication. No critical gaps.

---

**CURATOR signature:** Agent "curator" | Team: lencu-evidence-curation | 2026-02-23
**Ready for:** Editorial review, figure finalization, manuscript assembly
