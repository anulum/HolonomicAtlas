# Human Holonomic Atlas (Paper 21)
# Project: God of the Math

---

## 1. Project Objective & Cause

This repository contains the **Anatomia Holonomica** (Human Holonomic Atlas), the complete, version-controlled database for **Paper 21** of the *God of the Math* manuscript series.

**The Cause:** The foundational SCPN (Sentient-Consciousness Projection Network) framework maps consciousness to spacetime, but it exists as a series of theoretical papers. To validate, simulate, and apply this framework, we must first create a machine-readable, computable model of its primary subject: the human being.

**The Ultimate Objective:** To create a **simulatable, multi-dimensional graph database** where every anatomical component (from `D0.Microtubule` to `D15.Consilium`) is defined by its Classical (physical), SCPN (informational), and Symbolic (metaphysical) identities.

---

## 2. Core Source Documents

This Atlas is a **derivative work** that formalizes the theoretical framework defined in the *God of the Math* master manuscripts (Papers 0-20). All content is derived from these papers, which are provided to the "Architect" AI (Gemini).

---

## 3. Collaboration Workflow (Hybrid AI Team)

This project is managed by a four-agent team to leverage the unique strengths of different models and ensure "lossless" context.

1.  **You (Project Lead / Theorist):** The final human approver. Sets strategic goals and provides the master manuscripts.
2.  **Gemini (The Architect):** The large-context AI. Acts as the "RAG Engine" for the team. Reads all 20+ papers, performs cross-document synthesis, and generates the *exact, complete file content* (the "Commit Package") for each update.
3.  **ChatGPT-5 (The Consultant):** A small-context "Red Team." Receives a *single* monad file and *specific text excerpts* (prepared by Gemini) to critique and provide a "second opinion." Does *not* write to the repo.
4.  **Claude (The Actuator):** The GitHub-enabled AI. Its role is purely transactional. It receives the final "Commit Package" from Gemini and the *exact* commit message, and executes the `git push` to the repository.

This workflow (Gemini-Architect -> [ChatGPT-Consultant] -> Claude-Actuator) solves the context-limit problem of smaller AIs and the GitHub-access limit of larger AIs.

---

## 4. Handoff Protocol (For AIs)

All AIs must first read the files in the `/.handover/` folder to understand the project's state and their specific role.
* `atlas_manifest.yaml` (in root) defines the **Next Action**.
* `HANDOVER_LOG.md` contains the strategic history.
* Agent-specific protocols (`GEMINI_...`, `CLAUDE_...`, `CHATGPT_...`) define individual roles.