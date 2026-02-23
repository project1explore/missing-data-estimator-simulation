#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p results
python3 src/run_study.py --n-reps 500 --seed 20260221
python3 src/make_tables.py
python3 src/make_python_figures.py

cd paper
pdflatex -interaction=nonstopmode manuscript.tex >/tmp/genesis_pdflatex_1.log
pdflatex -interaction=nonstopmode manuscript.tex >/tmp/genesis_pdflatex_2.log

echo "Done. PDF: paper/manuscript.pdf"
