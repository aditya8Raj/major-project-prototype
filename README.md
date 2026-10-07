# Bio-Inspired Neuromorphic Vision for Nocturnal Pest Detection: software prototype

Group CMP/23-137, C.V. Raman Global University, Bhubaneswar. Guide: Dr. Monalisa Mishra.
Team: Aditya Raj, Aryan Pratap, Ayush Swain, Swayam Swain.

This repository holds the software prototype described in Chapter 4 and Appendix A of the Interim Project Report.

**Important:** every result here comes from a *synthetic* night scene with *assumed* sensor parameters. No real event camera or field data was used. The simulation numbers are not field results. The 0.95 F1 in the project proposal is a literature target and was not measured.

## What it does
A simulator makes night scenes (moths, leaves, stars) at three light levels and models an event camera and an ordinary frame camera. The pipeline then runs: noise filter, fast and slow time surfaces, a fruit-fly T4/T5-inspired motion map, event grouping, a gradient-boosted classifier (standing in for the planned graph neural network) and a tracker. It is compared with a frame-camera baseline.

## Files (in `source_code/`)
| File | Purpose |
|---|---|
| `simulator.py` | Synthetic night scene; event-camera and frame-camera models |
| `pipeline.py` | Noise filter, time surfaces, T4/T5-inspired motion map, event grouping, tracker, frame baseline |
| `run_experiments.py` | Training, testing, ablation; writes `results/results.json`, `stats.json`, `figure_data.npz` |
| `make_figures.py`, `make_block_diagram.py` | Figures of Chapters 3 and 4 (written to `figures/`) |
| `tune_noise_filter.py` | Noise-filter tuning on training seeds only |
| `make_demo_video.py` | Renders the demo video (needs `ffmpeg`; written to `demo/`) |
| `extended_experiments.py` | Sensitivity experiments of Chapter 7; writes `results/extended_results.json` (about 45 minutes) |
| `polarity_experiment.py` | Repeats the main experiment with ON/OFF leak noise; writes `results_symmetric_leak/` |
| `make_pipeline_figures.py`, `make_dataflow_figure.py`, `make_extended_figures.py` | Figures of Chapters 6 and 7 |
| `results/` | Results used in the report; `results_baf_tuning.txt` records the noise-filter tuning |

## Run
```
pip install -r requirements.txt
cd source_code
python run_experiments.py     # about ten minutes on four cores
python make_figures.py
python make_block_diagram.py
python make_demo_video.py     # optional
```
Random seeds are fixed. Training scenes use seeds 100-105 and test scenes 200-205.

## Known limits
Events do not beat frames in the dark in this simulation. The T4/T5 stage adds only a small gain. The noise filter was tuned on training seeds only (see report, Section 4.9).
