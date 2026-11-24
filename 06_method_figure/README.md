# Figure 1. Study design

The author supplied the initial layout from the ChatGPT web interface (original_chatgpt.png). The vector version retains its section headings, counts, layout and analysis branches. On 19 November 2026, the focal-pair label was aligned with the revised title: student ratings of online interaction and instructor help with questions. The supplied original_chatgpt.png remains an unchanged source draft. The shape and text instructions are in method_overview.tex. The source survey and split counts can be checked in ../03_results/data_audit.json; the 3,000 stratified replicates are documented in ../03_results/run.json.

The manuscript uses method_overview.pdf at 170 mm wide. The raster submission figure, method_overview_1200dpi.png, is 8032 × 4961 pixels, RGB, with a density of 1200 pixels per inch. It was rendered directly from the vector diagram.

To rebuild the figure in a working copy of this directory, use pdfLaTeX (standalone, TikZ, Helvetica) and Poppler:

~~~text
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode method_overview.tex
pdftoppm -r 1200 -singlefile -png method_overview.pdf method_overview_1200dpi
~~~

The four plots generated from survey results are in ../04_figures. The methods diagram describes the analysis design.
