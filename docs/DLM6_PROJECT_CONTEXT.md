# DLM6 Project Context

## Publication Details

- **Title**: AI-Driven Spatial Decision-Support for Sustainable Weed Management under Climate Variability
- **Volume**: *Smart-Agriculture and Technology-Innovation: Facing the Dynamic of Climate Change*
- **Editor**: Prof. Noureddine Benkeblia, University of the West Indies, Mona Campus
- **Publisher**: Springer Nature
- **Deadline**: February 23, 2026
- **Format**: Book chapter (contributed volume), LaTeX (`svmult.cls`)
- **Citation style**: APA 7th Edition (overrides Springer default per editor instruction)
- **Language**: American English (consistent throughout text, tables, and figure captions)

## Chapter Orientation

This chapter is a **toolkit-oriented contribution**. It presents a modular suite of AI tools for weed management decision-support under climate variability, demonstrated through the case of *Ambrosia artemisiifolia* (common ragweed) in Chilean agriculture. The chapter is accompanied by an **open GitHub repository** containing reproducible code for each module.

The contribution is **not** a fine-grained correlational analysis between environmental variables. It is a demonstration of novel, integrated tools applied to a species for which no deep learning detection models exist in the published literature, packaged for community adoption. Exploratory results are presented honestly, with unresolved challenges (anomaly confounding, domain transferability) framed as open research directions.

## Scientific Foundation (Validated)

The following claims have been verified through systematic literature search across eight research domains (February 2026):

1. **No deep learning detection model** for *A. artemisiifolia* exists in peer-reviewed literature (confirmed through Hammad et al., 2026; Dammer et al., 2013; comprehensive database searches).
2. **No published study** has combined species-specific DL weed detection, geostatistical density mapping, and satellite foundation model embeddings for any weed species. The individual components exist; their integration has not been attempted.
3. **Closest precedent**: Rasmussen et al. (2021) correlated drone-derived binary *Cirsium arvense* coverage with Sentinel-2 NDVI, achieving 24% herbicide savings. This used color-based classification (not species-specific DL), binary presence/absence (not continuous density), and correlative analysis (not predictive model training).
4. **The crop-conditioned anomaly detection framework** — detecting weed contamination as deviations from expected crop signatures within known crop types — is the conceptual innovation for satellite-scale scaling. This reframes the spatial resolution limitation: the crop signal is context to exploit, not noise to overcome.
5. **Anomaly confounding** is an acknowledged open challenge: spectral anomalies can arise from disease, nutrient deficiency, waterlogging, soil heterogeneity, and herbicide damage, not only weed presence. Temporal signature specificity (via PRESTO self-attention) is the proposed discrimination pathway.

## Key Empirical Results (León et al.)

- **Detection**: YOLOv11x + SAHI, *A. artemisiifolia* F1 = 0.84 in lentils at seedling stage (León et al., 2025b)
- **Spatial mapping**: Kriged continuous density surface, 66.5 plants/m², Moran's I = 0.667 (León et al., 2025b)
- **Bivariate spatial analysis**: LISA AMBEL density ↔ NDVI r = 0.818 (León et al., 2025b)
- **Management zones**: 31.9% critical zone, 35–50% herbicide reduction potential (León et al., 2025b)
- **Domain gap**: 25–60% performance collapse across Chile–Denmark transfer, 99.6% recovery with multi-domain learning (León et al., 2024b)
- **End-to-end pipeline**: 847 ha validated, 45–55% treated area reduction (León et al., 2025a)
- **Climate baseline**: 8-year climate data integrated with field observations (León et al., 2025b)
- **Datasets**: Seedling and mature *Ambrosia* across lentils, wheat, fallow, maize, tomato; geolocated across Chilean regions

## Public Health Dimension

*A. artemisiifolia* pollen causes an estimated €7.4 billion/year in European healthcare costs affecting 13.5 million patients (Müller-Schärer et al., 2020). Urban CO₂ amplifies ragweed biomass by 189% relative to rural sites (Ziska et al., 2003). Sensitization is projected to double from 33 to 77 million Europeans by 2041–2060 (Lake et al., 2017). The tools presented in the chapter enable **proactive source-level surveillance** of agricultural *Ambrosia* populations — monitoring the cause (field-level establishment and expansion) rather than the symptom (airborne pollen in cities).

## Related Publications in the Portfolio

- **León et al. (2024a)**: IntechOpen — IWM framework, SSWM concept, cover crops, YOLOv4 (F1 = 80–96%)
- **León et al. (2024b)**: AIA Journal — Domain gap in weed detection CNNs, Chile vs. Denmark
- **León et al. (2025a)**: Draft book chapter — End-to-end pipeline (YOLOv11 + SAHI → kriging → agentic LLM DSS via WhatsApp), 847 ha
- **León et al. (2025b)**: Preprint — YOLOv11x + SAHI → kriging → LISA → fuzzy management zones, 3.42 ha lentil field
- **DLM3 (CRC Press)**: *Data Collection, Analysis, and Agentic Modelling for Sustainable Weed Management* — under correction
- **Frontiers in Plant Science**: *Embodied-AI for Sustainable and Intelligent Phytoprotection* — abstract pending
- **AgroIA system (v0.4.0)**: RAG-based agricultural companion, 190 pesticide labels, weather spray windows, GraphRAG, WhatsApp interface

## Workflow

1. **Claude (this project)** produces `.md` scaffold files (CLAUDE.md, task lists, section specifications)
2. **Lorenzo** integrates scaffold files into the GitHub repository
3. **Claude Code (Cursor)** executes writing and analysis tasks following scaffold instructions
4. **Resulting LaTeX** is returned to this project for review, refinement, and compliance checking

## Springer Nature Editorial Requirements

### Text
- American English throughout (*Merriam-Webster's Collegiate Dictionary* as reference)
- Species names in italics: *Ambrosia artemisiifolia*, *Orobanche*, *Cirsium arvense*
- All abbreviations (YOLO, UAV, DSS, SAHI, PRESTO, NDVI, LISA, mAP) defined at first occurrence
- Decimal heading numbering: 1, 1.1, 1.1.1 (no skipped levels)
- Cross-references by number: "see Sect. 1.2", not by title
- SI units throughout; decimal points (not commas); commas for thousands
- Code in sans-serif or nonproportional font (Arial or Courier)
- Footnotes (not endnotes); footnotes do not substitute for references

### Citations (APA 7th Edition — Mandatory)
- One author: (Author, Year) or Author (Year)
- Two authors: (Author & Author, Year) — ampersand inside parentheses; "and" in running text
- Three or more: (First Author et al., Year)
- Multiple citations: alphabetical, semicolon-separated: (Chen et al., 2022; Reddy, 2021)
- Article titles in sentence case; journal names in full, italicized, title case
- DOIs as full URLs: `https://doi.org/xxxxx`
- Reference list at end of chapter, alphabetical, heading: **References**

### Figures
- Numbered sequentially: Fig. N.1, Fig. N.2 (N = chapter number)
- Cited in text in order; no "the following figure" phrasing
- Captions at end of text file, not in figure file
- Separate files named `León-FigN.n` (EPS with embedded fonts, or TIFF)
- Resolution: diagrams ≥ 1,200 dpi; photos ≥ 300 dpi; combined ≥ 600 dpi
- Dimensions: 78 mm or 117 mm wide; height ≤ 198 mm
- Lettering: Helvetica or Arial, 8–12 pt
- RGB (8 bits/channel); legible in grayscale; captions do not reference color
- Copyright permissions via copyright.com for any third-party material

### Tables
- Numbered sequentially: Table N.1, Table N.2
- Built with table function (not tabs or spaces)
- Saved within text file (not as separate files)
- Caption above table; source citation at end of caption if applicable

### File Submission
- LaTeX file named `León-ChapN`
- Figure files named `León-FigN.n`
- PDF proof with all fonts embedded
- Signed Consent to Publish form
- Copyright permission files if third-party material used
