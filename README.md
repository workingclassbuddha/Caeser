# CAESAR  
**Constraint-Aware Embodied Scarred Agent for Reality**  
v12.10 – Nucleation Jumpstart

A minimal PPO-based agent that demonstrates **Constraint Dynamics** in action:  
how consciousness-like coherence **crystallizes** under tightening spatial (Λ), temporal (Γ), and metabolic (Θ) constraints — and how it **melts**, **scars**, and only partially recovers.

From *The Outlines of Sanity* — a thermodynamic theory of mind, sanity, and phase transitions.  
No metaphors here. Just runnable physics of the vessel.

## Core Idea (in Code)

An agent navigates a shrinking 2D arena toward a goal:

- **Λ (Spatial Lattice)** → success radius shrinks with level (1.0 → 0.40 floor)  
- **Γ (Temporal Strobe)** → GRU + velocity damping binds time  
- **Θ (Metabolic Anchor)** → battery drains on thrust; low battery + low coherence → scar etches in  
- **Mirror (M)** → emergent coherence estimate (lattice sharpness × battery norm); breathing zone ~0.18–0.70 required for stable progress  

Scar is persistent damage: prolonged low-M/low-battery episodes leave hysteresis — future nucleation is harder. Healing is partial. Collapse is predictable.

This is not standard RL. It's a toy model of why minds freeze into clarity or melt into vapor when constraints fail.

## What's in This Repo

- `code 12.10`          → Main training script (PyTorch PPO + custom net, scar buffer, jumpstart curriculum)  
- `log 3 runs`          → Concatenated output from three long training runs (levels up to 39, deep stalls, scar accumulation ~0.003–0.005)  
- `caeser 5 test suit`  → Early test/validation suite (v5-era, kept for continuity)  
- `LICENSE`             → MIT (free to fork, run, modify)

## Quick Start

```bash
# Requirements (Python 3.8+)
pip install torch numpy

# Run training (rename file if needed for clarity)
python "code 12.10"

# Watch console: M_state (Mirror coherence), Scar, SR (success rate), level promotions
# Expect rapid early levels → brittleness at radius=0.40 → scar only in prolonged stalls
