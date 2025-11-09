# SCPN: Master Simulation Suite

## 1. Overview

This directory is the computational and experimental heart of the *God of the Math* project. It contains the executable, falsifiable, and validated simulation packages for every testable claim made in the manuscript.

This is not a static library. It is a **living testbed** that is continuously grown and validated through the **Serial Synthesis & Validation Loop (SSVL) Protocol**.

## 2. The Serial Synthesis & Validation Loop (SSVL)

Each simulation package in this directory is the end-product of a rigorous, 4-part generation process (defined in `.handover/ARCHITECT_LOG.md`).

For every foundational claim (e.g., "Quasicriticality"), the Architect (Gemini) generates:
1. A **Methodology (for Paper 17):** The formal blueprint for the experiment or simulation.
2. A **Simulation Analysis (for Paper 18):** The *predicted results* and data from running the simulation.
3. A **Falsification (for Paper 19):** The clear, binary "kill-switch" conditions.
4. A **Simulation Package (for this Directory):** The `README.md` and `run_*.py` script to computationally validate the claim.

## 3. Directory & Naming Convention

Each subdirectory corresponds to one falsifiable claim and is named according to its Layer or primary concept.

The content of these simulations is then used to populate the three main validation manuscripts (*Papers 17, 18, and 19*), using a naming convention that links them all together:

**`P[Source]_P[Target]_[FST_ID]`**

* `P[Source]`: The paper the claim originates from (e.g., **P0** for *Paper 0*).
* `P[Target]`: The paper the text is intended for (e.g., **P17**, **P18**, **P19**).
* `[FST_ID]`: The unique Falsifiable Synthesis Tract number.

For example, the simulation `L11_NTHS/` corresponds to:
* `P0_P17_001`: The methodology for the NTHS simulation.
* `P0_P18_001`: The predicted results (spin-glass vs. ferromagnetic).
* `P0_P19_001`: The falsification conditions ($m \to 0, q_{EA} > 0$).

## 4. Current Status: P0 Validation Suite

The processing of *Paper 0 - The Foundational Framework* (Rev 11.31) is **100% complete**.

This directory contains the 20 validation packages for all foundational claims.

**Completed Packages (P0):**
* `L00_Category_Theory_Validation/`
* `L00_FIM_Pullback/`
* `L00_HPC_UPDE_Bridge/`
* `L00_SSB_Lagrangian_Validation/`
* `L00_UPDE_Kuramoto/`
* `L01_FIM_NV_Center/`
* `L01_MS_QEC_Validation/`
* `L01_QSR_Amplification/`
* `L02_ALP_EM_Bridge/`
* `L03_CISS_Bioelectric/`
* `L04_SOC_Quasicriticality/`
* `L04_Two_Timescale_Controller/`
* `L05_Four_Stroke_Engine/`
* `L05_Geometric_Qualia_TDA/`
* `L09_MERA_QEC_Stabilizer/`
* `L11_NTHS/`
* `L15_CEF_QRNG/`
* `L15_SEC_Objective_Functional/`
* `L15_SEC_SC_Equivalence/`
* `L16_Optimal_Control_HJB/`
