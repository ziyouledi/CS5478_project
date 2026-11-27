# CS5478 Final Report (NeurIPS 2024 Style)

- **Page limit**: 5--8 pages excluding references and appendices (target: ~7.5--8.0 pages).
- **Code repository**: `https://github.com/ziyouledi/CS5478_project`

## Structure

- `main.tex`: Master document (NeurIPS 2024 style, `[final]` mode).
- `neurips_2024.sty`: NeurIPS-compatible style file (5.5 in x 9 in text block, 10 pt Times, `natbib`).
- `references.bib`: BibTeX bibliography.
- `sections/`: Modular section files (`00_abstract.tex` through `06_appendix.tex`).
- `figures/`: Paper-ready PDF/PNG figures.

## Building

### Option 1: Overleaf (recommended if local TeX is not installed)
```bash
make overleaf
```
Upload `CS5478_Team34_Report_Overleaf.zip` directly to Overleaf and compile with `pdfLaTeX`.

### Option 2: Local TeX Live / MacTeX
```bash
make pdf
```

## Page Budget Tracker (Main Text)

| Section | Target Pages | Status |
| --- | ---: | --- |
| Title + Abstract | 0.4 | Template ready |
| 1. Introduction (+ Fig. 1) | 1.0 | Template ready |
| 2. Related Work | 0.7 | Template ready |
| 3. Method | 1.5 | Template ready |
| 4. Experiments (4.1--4.5 incl. Plan C) | 3.6 | Template ready |
| 5. Discussion, Limitations & Conclusion | 0.7 | Template ready |
| **Total Main Text** | **~7.9** | Within 5--8 page limit |
