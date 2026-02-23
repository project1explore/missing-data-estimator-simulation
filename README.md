# Missing-Data Estimator Simulation Study

Reproducible in-silico medical-statistics project with:

- literature-backed research gap
- explicit simulation protocol
- extended Monte Carlo study
- LaTeX manuscript with tables + graphs
- compiled PDF output

## Repository structure

- `docs/` — research gap, plan, literature notes
- `src/run_study.py` — main simulation engine
- `src/make_tables.py` — LaTeX tables + study-count macros
- `src/make_python_figures.py` — renders plots with Python/Matplotlib to PDF
- `results/` — raw + summarized outputs + generated figure PDFs
- `paper/manuscript.tex` — manuscript source
- `paper/manuscript.pdf` — compiled paper

## Run everything

```bash
# from the repository root
./run_all.sh
```

This will:
1. run the simulation study (`n_reps=500`) in parallel across all available CPU cores,
2. build tables (including Type I error column in all tables),
3. render figures in Python (Matplotlib) as PDF,
4. compile the manuscript PDF.

## Notes

- Python dependencies for plotting: `matplotlib` (installed in user site packages).
- Figures are generated as `results/fig_*_python.pdf` and included in LaTeX with `\includegraphics`.
