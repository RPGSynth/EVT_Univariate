# 🌞 PhD Paper Scripts Repository 🌞

Welcome to the sunny codebase behind my **first PhD paper** on extreme events.  
Every folder is a small lab: some hold production-ready GEV code, others are sandboxes for simulations, neural kernels, or data-import helpers. Enjoy the stroll!

---

## ✨ What’s Inside

| Area | What it is | Key entry points |
| --- | --- | --- |
| **EVT_Classes/** | Legacy/statistical workflows plus neural-kernel experiments. | `GEV.py`, `utils.py`, `SIM/` simulator + notebook, `NN/` experiments. |
| **Refactor/** | Modern GEV API powered by JAX/JAXopt. | `gevPackage/` modules; demo in `Refactor/main.py`. |
| **Exploration/** | Work-plan notes for the broader project. | `Exploration/README.md`. |
| **Import/** | Download helpers for climate datasets (EOBS, CMIP-S). | `import_EOBS.py`, `import_CMISP.py`, bash `sh/` script. |
| **outputs** | Example animations/plots from notebooks. | `EVT_Classes/SIM/*.gif`, `output.png`. |

---

## 🚀 Quickstart (Refactored GEV API)

1. **Environment**
   ```bash
   conda create -n phd python=3.11
   conda activate phd
   pip install jax jaxlib jaxopt numpy scipy matplotlib pandas

## ⭐ Centerpiece: Weighted Neural Kernel for Return Levels
The repo’s signature idea is a **weighted neural-kernel approach** that learns distance→weight functions to bias GEV fits toward relevant space–time neighborhoods. You’ll find the working code and demos here:

- **Simulation + notebooks:** `EVT_Classes/SIM/sim.ipynb` (uses a trained `DynamicWeightNet` to build weights and fit GEVs).

If you’re here for the novelty, Open `EVT_Classes/SIM/sim.ipynb` to see the weights driving the GEV fits and return-level plots.
