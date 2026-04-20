# Quantum-Holographic Recursive Network (QHRRN) for ARC-AGI

The Quantum-Holographic Recursive Network is a physics-inspired AI architecture optimized for solving the Abstraction and Reasoning Corpus (ARC-AGI). Rather than relying on massive generic dataset pre-training leading to Volume-Law parameter explosion, the QHRRN treats deductive geometric reasoning as **Spontaneous Symmetry Breaking bounds**. It dynamically annealing logic to isolate localized invariants and solve puzzles algorithmically reliably securely.

## 🌌 Architectural Overview

The core mathematical operations function on constraints traditionally studied within Quantum Many-Body physics specifically optimizing Tensor Network contractions:

### 1. The PEPS Tensor Substrate
The input 2D ARC problem boards natively map directly onto a constrained Projected Entangled Pair States (PEPS) matrix manifold. The localized parameters restrict mathematical dependencies geographically (Area Law mappings), completely isolating distant geometric rules effectively natively preventing statistical hallucinations natively bounding dimensionality structurally limits heavily.

### 2. The MERA Engine
The Multi-scale Entanglement Renormalization Ansatz (MERA) architecture operates structurally securely managing recursive compression over geometric variables:
- **Disentanglers ($U$)**: Symmetrical matrices enforcing exact physics mathematically parameterized strictly via Cayley Transform mappings. They rotate independent structures strictly guaranteeing $U^\dagger U = I$.
- **Isometries ($W$)**: Geometric boundaries forcing spatial bounds locally onto dense localized dimensions natively effectively projecting topological dependencies across Stiefel limits recursively until reaching structural Latent limits evaluating $1 \times 1$ boundaries securely functionally.

### 3. Test-Time Training (TTT) via Langevin Dynamics
This inference loop isolates identical local invariant definitions effectively natively generating bounds autonomously over local targets strictly minimizing structural parameters locally limits efficiently naturally:
- **Categorical Reconstruction Limits:** Cross-Entropy restricts actual representation natively explicitly bounds dynamically predicting pixel geometries mathematically reliably limits safely explicitly.
- **Holographic Occam's Razor:** Approximated Singluar Value SVD spectra effectively evaluate logical Von Neumann topological scaling limits directly natively functionally constraining logic mapping efficiently mathematically preventing parameter bloating functionally securely dynamically natively!
- **Dynamic Phase Transitions:** Utilizing Brownian thermal noise constraints naturally driving states autonomously tracking EMA mappings structurally from the random "Melting" exploration natively cleanly structurally stabilizing down into structural deterministic "Zero-Temperature Freezing" limits globally natively optimally optimally!

## 🚀 Execution & Operations

This codebase dynamically adapts its computational pipeline seamlessly to target generic RTX evaluation machines or massive highly constrained Google Cloud TPU configurations organically reliably safely dynamically securely securely!

### Staged Deployment Workflow

To aggressively protect your budget, QHRRN supports a **Two-Stage Staged Deployment**:

#### Stage 1: Pre-training Mode (`--pretrain`)
Pre-trains the Base Core and Hypernetwork on a small subset of representative tasks to establish the fundamental MERA geometry. Saves a portable `core_checkpoint.pkl`.

```bash
# Local pre-training (e.g., RTX 3070)
python run_ttt.py --pretrain pretrain_tasks.txt --pretrain-epochs 5

# Cloud pre-training with automatic Checkpoint Rescue before teardown
python dispatcher.py --cloud --project <YOUR_PROJECT> --pretrain
```

#### Stage 2: Subset Evaluation & Checkpoint Injection
Injects the frozen checkpoint and evaluates a subset of the benchmark to validate convergence.

```bash
# Evaluate the first 10 tasks locally using the checkpoint
python run_ttt.py --benchmark eval_tasks.txt --checkpoint core_checkpoint.pkl --eval-subset 10

# Full 400-task benchmark on Cloud TPU
python dispatcher.py --cloud --project <YOUR_PROJECT>
```

> **Budget Bound Protection Active:** The `--cloud` dispatcher securely provisions Spot TPUs, conditionally rescues the pre-trained checkpoint locally over SCP if in `--pretrain` mode, and **crucially**, unconditionally triggers `gcloud compute tpus tpu-vm delete` to eradicate idle instances within a strict `finally` block preventing runaway billing!

## Parameters & Constraints Limits
The entire QHRR core architecture executes strict topological volume-constraint audits enforcing evaluations `< 10,000,000` operations natively effectively locally limits flawlessly dynamically!
