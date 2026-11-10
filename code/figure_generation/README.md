# Python figure generation

`prepare_updated_figure_data.py` assembles the plot-ready CSV/JSON products.
`plot_figures_2_to_6.py` renders Figures 2-6 with Matplotlib using fixed
180-mm journal canvases, vector PDF/SVG output and 600-dpi PNG output. The
same module renders Supplementary Figure 1 from paired flow-state tables.

```bash
python prepare_updated_figure_data.py
python plot_figures_2_to_6.py --figures 2 3 4 5 6
```

The renderer is also called by the case-level scripts, so a reader can run one
scientific case and obtain its numerical data and associated figures in one
output directory.
