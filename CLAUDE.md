# CLAUDE.md — DLM6 Weed Toolkit Chapter

## Project

This repository is the companion toolkit for a Springer Nature book chapter:
- **Title**: AI-Driven Spatial Decision-Support for Sustainable Weed Management under Climate Variability
- **Volume**: Smart-Agriculture and Technology-Innovation: Facing the Dynamic of Climate Change
- **Editor**: Prof. Noureddine Benkeblia, University of the West Indies
- **Publisher**: Springer Nature
- **Deadline**: February 23, 2026
- **Format**: LaTeX (svmult.cls), APA 7th Edition citations

## Key Message

"We explored and tested multiple tools for recognizing and mapping weeds at complementary
scales. None alone solves the problem. Together, they form a toolkit for decision-making
under climate variability — and we put this repository into the community's hands."

## Three-Act Structure

- **Act I — Field Detection**: YOLO + SAHI detection → geostatistical mapping → management zones
- **Act II — Domain Generalization**: Cross-domain evaluation, MMD gates, multi-domain recovery
- **Act III — Satellite Integration**: PRESTO embeddings, NDVI correlation, GWR, LISA clusters

## Key Results (Verified)

| Metric | Value | Source |
|--------|-------|--------|
| Best mAP50 (AMBEL detection) | 0.886 | Training Run 3 |
| Cross-domain collapse (CL→INTL) | mAP50 = 0.108 | Phase 1 |
| Multi-domain recovery | mAP50 = 0.874 | Phase 2 |
| NDVI–AMBEL correlation | r = 0.837 | Single-date Sept 24 |
| NDVI–LENCU correlation | r = 0.890 | Single-date Sept 24 |
| PRESTO PC1–AMBEL | r = 0.739 | Jul–Dec 2024 |
| PRESTO PC1–LENCU | r = 0.717 | Jul–Dec 2024 |
| Bivariate Moran's I (PC1×AMBEL) | 0.706 (p=0.001) | 999 permutations |
| GWR R² AMBEL | 0.882 | PC1+PC2+PC3, BW=49nn |
| GWR R² LENCU | 0.910 | PC1+PC2+PC3, BW=49nn |
| LISA significant pixels | 68% | PC1×AMBEL |

## Editorial Rules (MANDATORY for all text output)

- **Language**: American English (Merriam-Webster)
- **Citations**: APA 7th Edition. Ampersand inside parentheses: (Author & Author, Year). "and" in running text.
- **Species names**: Always italics: *Ambrosia artemisiifolia*, *Convolvulus arvensis*
- **Headings**: Decimal numbering (1, 1.1, 1.1.1). Cross-references by number: "see Sect. 1.2"
- **Abbreviations**: Define at first use (YOLO, SAHI, UAV, DSS, PRESTO, NDVI, LISA, mAP, GWR, OLS, MMD)
- **Figures**: Fig. N.1, Fig. N.2 (N = chapter number). Captions at end of text file. Separate files named León-FigN.n
- **Tables**: Table N.1, Table N.2. Built in LaTeX, not as images.
- **Figure specs**: 78 mm or 117 mm wide; ≤198 mm height; Helvetica/Arial 8–12 pt; ≥300 dpi photos; ≥1200 dpi diagrams; RGB; legible in grayscale
- **References**: End of chapter, alphabetical. DOIs as https://doi.org/xxxxx. Journal names in full.

## File Locations

| What | Where |
|------|-------|
| Chapter LaTeX | `chapter/main.tex` |
| Figures | `chapter/figures/` |
| Tables | `chapter/tables/` |
| Evidence docs | `evidence/` |
| Tool scripts | `tools/` |
| Configs | `configs/` |
| Data pointers | `data/README.md` |
| Narrative roadmap | `docs/CHAPTER_STORYLINE.md` |
| Curation report | `docs/EVIDENCE_CURATION_REPORT.md` |
| Full project context | `docs/DLM6_PROJECT_CONTEXT.md` |

## Workflow

1. Read `docs/CHAPTER_STORYLINE.md` for the three-act narrative and figure plan
2. Read `docs/EVIDENCE_CURATION_REPORT.md` for per-exercise best evidence
3. Read `docs/DLM6_PROJECT_CONTEXT.md` for verified numbers and scientific foundation
4. Write LaTeX sections following the three-act structure
5. All figures referenced must exist in `chapter/figures/`
6. Lorenzo reviews LaTeX via the Claude project on claude.ai
