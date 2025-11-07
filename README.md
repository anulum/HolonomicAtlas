# Human Holonomic Atlas (Paper 21)
# Project: God of the Math
---

## 1. Project Objective & Cause
This repository is the definitive, self-contained "brain" for the **Anatomia Holonomica** (Paper 21). It contains two main components:
1.  **The "Knowledge" (`/Corpus/`):** The complete 20+ paper *God of the Math* manuscript, which serves as the ground truth.
2.  **The "State" (`/data/`):** The 40-monad JSON database (the Atlas) that is derived from that knowledge.

**The Ultimate Objective:** To create a **simulatable, multi-dimensional graph database** where every anatomical component (from `D0.Microtubule` to `D15.Consilium`) is defined by its Classical (physical), SCPN (informational), and Symbolic (metaphysical) identities.

---

## 2. Getting Started

### Prerequisites
- Python 3.8+
- Pydantic: `pip install pydantic`
- Git LFS: [Download and install Git LFS](https://git-lfs.github.com/) before cloning the repository.

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: A `requirements.txt` file should be created if it doesn't exist yet.)*

---

## 3. Project Structure
- **`/Corpus/`**: Contains all source manuscripts (Papers 0-20+) in `.docx`, `.pdf`, and text formats. **Requires Git LFS.**
- **`/data/`**: Holds the JSON database of all anatomical monads.
- **`/src/`**: Contains the core Python source code, including the `monad.py` schema definition.
- **`/.handover/`**: Contains protocol and log files for AI collaborators.
- **`atlas_manifest.yaml`**: The central manifest file that defines the project's current state and next actions.

---

## 4. Core Data Structure: The AnatomicalMonad
The entire Holonomic Atlas is built upon a single, core data structure: the `AnatomicalMonad`. This Pydantic model, defined in `src/monad.py`, provides a unified schema for describing any anatomical component.

A monad is composed of several "scripts" or data objects:
- **`HolonomicID`**: A unique, hierarchical identifier (e.g., `scpn://atlas.v1/D3.Brain.PinealGland`).
- **`ParentID`**: The ID of the parent monad, defining the anatomical hierarchy.
- **`Classical` (D-Script)**: The physical, classical description (Structure, Function, Pathology).
- **`SCPN` (I-Script)**: The 16-Layer Informational Mapping, linking the monad to different levels of reality.
- **`Symbolic` (S-Script)**: The metaphysical and vibrational identity (Symbolic Name, VIBRANA Key).
- **`Meta` (M-Script)**: Versioning and status information.
- **`Validation` (V-Script)**: Falsification hypotheses and references to experimental/simulation protocols.
- **`Formalism` (F-Script)**: The mathematical, computational, and linguistic formalism of the monad.

---

## 5. Core Source Documents
All source manuscripts (Papers 0-20+) are stored in the `/Corpus/` folder.
Each paper exists in three versions:
* `PaperName.docx`: The original, editable master.
* `PaperName.pdf`: The distributable/readable master.
* `PaperName_RAG.txt`: A compressed, text-only version for the Gemini-Architect AI to ingest.

**This repository uses Git LFS (Large File Storage) to manage the `.docx` and `.pdf` files.**

---

## 6. Collaboration Workflow (Hybrid AI Team)
This project is managed by a four-agent team:
1.  **Project Lead (User):** The final human approver. Sets strategic goals and provides new paper revisions.
2.  **Gemini (Architect):** The "RAG Engine." Reads the `/Corpus/` folder for knowledge and the `/data/` folder for state. Generates the "Commit Packages."
3.  **ChatGPT-5 (Consultant):** A small-context "Red Team" to critique specific packages.
4.  **Claude (Actuator):** The GitHub-enabled "Committer." *Status: Pending Access.* The User currently handles this role via GitHub Desktop.

---

## 7. Handoff Protocol (For AIs)
All AIs must first read the files in the `/.handover/` folder to understand the project's state and their specific role.
* `atlas_manifest.yaml` (in root) defines the **Next Action**.
* `HANDOVER_LOG.md` contains the strategic history.
* `GEMINI_ARCHITECT_PROTOCOL.md` defines the boot-up sequence for the primary AI.
