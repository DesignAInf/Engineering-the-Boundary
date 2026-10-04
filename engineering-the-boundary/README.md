# Engineering the Boundary

**Markov Blanket Design as an Inferential Advantage in Active Inference**

Luca M. Possati — University of Twente

This repository contains the paper and all the code needed to reproduce its simulations and figures.

## Central result

For an active inference agent, the location of its Markov blanket is not a precondition of inference but one of its variables.

1. **The boundary enters the free energy.** Describing oneself as separated from the world across a partition costs a free-energy excess, the *factorisation gap*, which belief updating, precision (attention) and model expansion cannot remove; only changing the partition or the couplings across it can.
2. **Self-evidencing therefore engineers the boundary.** If the couplings that cross the boundary descend free energy on a slow timescale, they decouple interior from exterior by an anti-Hebbian rule until the blanket is exact; where this is impossible, Bayesian structural perception redraws the boundary.
3. **Engineering the boundary pays, under stated conditions.** An agent that can perceive and reshape its blanket outperforms one that cannot, in a way attention cannot reproduce, whenever the world's coupling structure changes more slowly than the agent can detect the change and amortise the cost of rebuilding.

The results connect the free energy principle with design theory (design as the engineering of Markov blankets) and with ontology (things as maintained partitions).

## Repository layout

```
paper/
  engineering_the_boundary.tex          complete, self-contained LaTeX source (bibliography embedded)
  engineering_the_boundary_entropy.tex  version formatted for submission to Entropy (numeric citations, back matter)
  engineering_the_boundary.pdf          compiled paper
  refs.bib                              BibTeX database
  figs/                                 all figures (PDF)
code/
  paths.py                              output locations (override with ETB_FIGDIR / ETB_RESDIR)
  sim1_gaussian_fields.py               Simulation 1  - breached and re-engineered blankets in Gaussian fields
  sim2_emergence.py                     Simulation 2  - tightening, redrawing, Occam cost, Ising systems
  mini_dmbd.py                          NumPy re-implementation of the core of DMBD (Beck & Ramstead 2025)
  sim3_inversion.py, sim3_figure.py,    Simulation 3  - inversion with a DMBD-style detector
  sim3_flat_prior.py
  sim4_foraging_agents.py, sim4_run.py, Simulation 4  - foraging with fixed / engineerable blankets
  sim4_figures.py, sim4_calibration.py
  sim5_pavilion.py, sim5_run.py,        Simulation 5  - a pavilion on a heat-conducting lattice in random weather
  sim5_figures.py
  original_dmbd/run_original_dmbd.py    repeats the detection step of Simulation 3 with the original PyTorch DMBD code
results/                                raw results and summaries underlying every number in the paper
run_all.sh                              reproduces everything
```

| Simulation | Figures |
|---|---|
| 1 | `fig_sim1.pdf` |
| 2 | `fig_emergence.pdf` |
| 3 | `fig_inversion.pdf` |
| 4 | `fig_trace.pdf`, `fig_bars.pdf`, `fig_phase.pdf` |
| 5 | `fig_thermal.pdf` |

## Reproducing the results

Requirements: Python ≥ 3.10, NumPy, SciPy, Matplotlib (`pip install -r requirements.txt`).

```bash
./run_all.sh                                    # overwrites results/ and paper/figs/
ETB_RESDIR=/tmp/r ETB_FIGDIR=/tmp/f ./run_all.sh  # writes elsewhere, keeps the archived results
```

The full run takes roughly 20–30 minutes on a two-core machine. A complete re-run reproduced every
number reported in the paper; in the Simulation 5 sweep, two of the 96 runs can differ slightly between
machines (floating-point differences that flip a discrete reconfiguration decision), which moves two sweep
cells by about one percentage point without changing any statement in the paper. Set `OMP_NUM_THREADS=1` if you run the
multiprocessing scripts by hand: BLAS thread oversubscription slows them down by an order of magnitude.

To compile the paper: `cd paper && pdflatex engineering_the_boundary.tex && pdflatex engineering_the_boundary.tex`
(no BibTeX run is needed; the bibliography is embedded).

## Checking the detector against the original DMBD

Simulation 3 uses `mini_dmbd.py`, a simplified NumPy re-implementation of the core of Dynamic Markov
Blanket Detection (one role per class, static labels, point-estimated parameters, element-specific
loadings). To repeat the detection step with the original code:

```bash
git clone https://github.com/bayesianempirimancer/pyDMBD
pip install torch numpy scipy matplotlib
cp code/original_dmbd/run_original_dmbd.py code/sim3_inversion.py code/mini_dmbd.py code/paths.py pyDMBD/
cd pyDMBD && python run_original_dmbd.py
```

## Citation

If you use this work, please cite it as indicated in `CITATION.cff`.

## License

Code: MIT License (see `LICENSE`). Paper text and figures: © 2026 Luca M. Possati, all rights reserved.
