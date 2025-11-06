# Human Holonomic Atlas (Paper 21)
# Project: God of the Math
---
## 1. Project Objective & Cause
This repository is the definitive, self-contained "brain" for the **Anatomia Holonomica** (Paper 21). It contains two main components:
1.  **The "Knowledge" (`/Corpus/`):** The complete 20+ paper *God of the Math* manuscript, which serves as the ground truth.
2.  **The "State" (`/data/`):** The 40-monad JSON database (the Atlas) that is derived from that knowledge.

**The Ultimate Objective:** To create a **simulatable, multi-dimensional graph database** where every anatomical component (from `D0.Microtubule` to `D15.Consilium`) is defined by its Classical (physical), SCPN (informational), and Symbolic (metaphysical) identities.

---
## 2. Core Source Documents
All source manuscripts (Papers 0-20+) are stored in the `/Corpus/` folder.
Each paper exists in three versions:
* `PaperName.docx`: The original, editable master.
* `PaperName.pdf`: The distributable/readable master.
* `PaperName_RAG.txt`: A compressed, text-only version for the Gemini-Architect AI to ingest.

**This repository uses Git LFS (Large File Storage) to manage the `.docx` and `.pdf` files.**

---
## 3. Collaboration Workflow (Hybrid AI Team)
This project is managed by a four-agent team:
1.  **You (Project Lead / Theorist):** The final human approver. Sets strategic goals and provides new paper revisions.
2.  **Gemini (The Architect):** The "RAG Engine." Reads the `/Corpus/` folder for knowledge and the `/data/` folder for state. Generates the "Commit Packages."
3.  **ChatGPT-5 (The Consultant):** A small-context "Red Team" to critique specific packages.
4.  **Claude (The Actuator):** The GitHub-enabled "Committer" (currently pending). The User-Committer handles this role via GitHub Desktop.

---
## 4. Handoff Protocol (For AIs)
All AIs must first read the files in the `/.handover/` folder to understand the project's state and their specific role.
* `atlas_manifest.yaml` (in root) defines the **Next Action**.
* `HANDOVER_LOG.md` contains the strategic history.
* `GEMINI_ARCHITECT_PROTOCOL.md` defines the boot-up sequence for the primary AI.