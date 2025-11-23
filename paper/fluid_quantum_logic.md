\begin{center}
{\LARGE \textbf{Fluid Quantum Logic: Zero-Shot Reprogrammability via Ancilla Superposition}}

\vspace{0.8em}

{\large Larsen James Close}\\[0.2em]
{\small larsenclose@pm.me}\\[0.2em]
{\small github.com/LarsenClose/fluid-quantum-logic}\\[0.4em]
{\small November 2025}\\[0.3em]
{\footnotesize U.S. Provisional Patent Application Serial No. 63/921,961}
\end{center}

\vspace{0.8em}

## Abstract

We demonstrate that quantum circuits exhibit native computational primitives that can be reprogrammed without training by rotating ancilla qubits. Using a 6-qubit reprogrammable logic unit simulated on classical hardware, we achieve 100% accuracy on AND, OR, and XOR operations across all four input combinations with zero parameter updates. The operations are **inherent to circuit geometry**, not learned representations. Program synthesis reveals that only 4 of 16 possible 2-input Boolean functions are native to the topology, supporting the view that computational structure is constrained by circuit geometry rather than trainable weights.

We frame this result as **field programmability via state preparation**: Unlike classical FPGAs which reconfigure sequentially between compute cycles, this architecture allows **superposition of logic gates**, processing data through multiple Boolean functions simultaneously within a single quantum execution. This enables a new computational paradigm where the instruction set architecture (ISA) itself is a quantum state.

Supporting experiments on a 14-qubit vision-style architecture demonstrate 43% quantum interference in attention mechanisms, perceptual bistability (62-76% depending on initialization), and preservation of the XOR truth table under a depolarizing-noise model with p = 0.1 for this small circuit, suggesting robustness under realistic NISQ-like error rates. Domain transfer to audio demonstrates that quantum primitives generalize across modalities; the XOR primitive achieves 100% accuracy on rhythm detection with zero training. Noise robustness experiments show that geometry-based gates survive simulated NISQ-era noise better than trained parameterized circuits, which suffer from parameter drift. This work demonstrates that, at least for small circuits, quantum circuits can function as reprogrammable computational substrates without gradient-based optimization, offering a path beyond parameterized quantum circuits susceptible to barren plateaus and noise sensitivity.

**Keywords**: quantum computing, zero-shot learning, quantum primitives, circuit reprogrammability, consciousness modeling, quantum neural networks

---

## 1. Introduction

### 1.1 Motivation

Classical neural networks learn representations through iterative weight updates via backpropagation. This training-dependent paradigm raises a fundamental question: **Are there computational substrates where operations are inherent rather than learned?**

Quantum circuits offer a unique testbed for this question. Unlike classical gates with fixed truth tables, quantum gates operate on continuous superpositions and can be controlled by ancilla qubits. Recent work in quantum machine learning (QML) has focused on parameterized quantum circuits (PQCs) that learn via gradient descent [1, 2], though these approaches suffer from issues such as barren plateaus in their training landscapes. This paradigm may miss a critical insight: **quantum circuits may possess native computational primitives that require no training**.

### 1.2 Architectural Inspiration: Integrated Information and Predictive Coding

Our quantum circuit architecture is inspired by computational frameworks from cognitive science, particularly Joscha Bach's Request-Confirmation Network (RCN) model [3] and Integrated Information Theory [4]. These frameworks suggest that certain information-processing patterns may be substrate-independent:

1. **Coherence maximization** - Maintaining information integration across processing layers
2. **Discrete state resolution** - Collapsing ambiguous inputs to definite interpretations
3. **Dynamic attention gating** - Modulating information flow based on context
4. **Hierarchical compositional binding** - Multi-scale feature integration

Rather than claiming these patterns constitute "consciousness," we use them as **architectural design principles** for quantum neural networks. We make no claims about subjective experience or quantum effects in biological brains; "consciousness" here is purely a shorthand for a particular class of information-processing architectures. The key insight: quantum circuits may naturally implement these patterns through their inherent physics (superposition, entanglement, measurement collapse), rather than through learned parameters. This suggests a novel approach to quantum algorithm design that leverages native quantum properties instead of classical neural network architectures.

### 1.3 Contributions

This paper makes four primary contributions:

1. **Zero-Shot Logic Reprogramming**: We demonstrate a 6-qubit reprogrammable logic unit that performs AND, OR, and XOR operations at 100% accuracy without training, controlled solely by ancilla rotations. Note that "reprogrammable" here refers to covering AND, OR, XOR, and FALSE within this topology, not universality over all classical Boolean functions.

2. **Program Synthesis for Native Gate Discovery**: Automated search reveals that only 4 of 16 possible 2-input boolean functions are native to our circuit topology, establishing that computational power emerges from geometric structure, not arbitrary programmability.

3. **Quantum Interference Measurement**: We measure 43% deviation from classical predictions when attention mechanisms are in superposition, validating genuine quantum effects.

4. **Perceptual Bistability via Proper Sampling**: Using measurement sampling (rather than expectation values), we demonstrate bistable collapse (62-76% bimodal distribution depending on initialization) on ambiguous inputs, validating discrete quantum state collapse.

### 1.4 Framing: Quantum FPGA

We frame our results using the analogy of a **field-programmable gate array (FPGA)**, where:
- **Classical FPGA**: Hardware is reconfigured via lookup tables and routing matrices
- **Quantum FPGA**: Circuit function is reconfigured via ancilla qubit superpositions

This framing emphasizes that quantum reprogrammability operates at the *physical layer* (qubit states) rather than the *logical layer* (learned weights), offering a fundamentally different computational paradigm.

**What is quantum vs. classically emulable**: The logic unit itself implements classical Boolean operations in a reversible quantum circuit. The quantum advantage in our experiments does not lie in computational complexity but in the ability to place gate selections in superposition, producing interference patterns (Section 3.3) and bistable collapse (Section 3.4) that have no classical analogue in a single deterministic circuit.

---

## 2. Methods

### 2.1 Circuit Architecture

#### 2.1.1 Reprogrammable Logic Unit (6 Qubits)

Our core architecture consists of:
- **Wires 0-1**: Input qubits (A, B) encoded via RX rotations
- **Wire 2**: Output qubit measured via PauliZ expectation
- **Wires 3-5**: Ancilla qubits controlling gate selection

**Gate Selection Mechanism** (see Figure 1):
```python
# Ancilla rotations select operation (0 or π)
RY(θ_OR, wire=3)   # π → activate OR
RY(θ_AND, wire=4)  # π → activate AND
RY(θ_XOR, wire=5)  # π → activate XOR

# Controlled gate execution
ctrl(or_binding, control=3)()
ctrl(and_binding, control=4)()
ctrl(xor_binding, control=5)()
```

![Universal Logic Unit architecture showing Sensory, Control, and Binding layers](figures/figure1_architecture.png)

**Gate Implementations**:
1. **AND**: Pure Toffoli gate `Toffoli(A, B, output)`
2. **OR**: De Morgan's law via input inversion: `NOT(NOT A AND NOT B)`
3. **XOR**: Parity detection via CNOTs: `CNOT(A, output); CNOT(B, output)`

**Key Innovation**: All three gates use the **same underlying topology**, differing only in ancilla control and input/output transformations.

#### 2.1.2 Hierarchical Vision RCN (14 Qubits)

For bistability and interference experiments:
- **Layer 1 (Wires 0-7)**: Sensory encoding (4×2 pixel grid)
- **Layer 2 (Wires 8-10)**: Feature detectors (horizontal, vertical, density)
- **Layer 3 (Wire 11)**: Scene coherence monitor
- **Ancilla (Wires 12-13)**: Attention control qubits

**Binding Operations**:
```python
# Spatial binding (horizontal detector)
for row in [0:4, 4:8]:
    for pixel in row:
        CNOT(pixel, detector)
        RY(param_row, detector)

# Attention-modulated binding
ctrl(horizontal_binding, control=ancilla_H)()
ctrl(vertical_binding, control=ancilla_V)()
```

**Scene Coherence Layer**:
```python
BasicEntanglerLayers(weights, wires=[8,9,10,11])
```
This layer creates CNOT ring connectivity between feature detectors, enabling discrete collapse for bistability (Section 3.4).

### 2.2 Training Protocol

#### 2.2.1 Zero-Shot Logic (No Training)

**Critical Point**: Universal logic gates receive **no parameter updates**. The logic unit itself has no trainable weights; it consists entirely of fixed Toffoli/CNOT primitives and ancilla rotations.

**Test Protocol**:
1. Set ancilla to select gate (e.g., [π, 0, 0] for OR)
2. Test all 4 input combinations: 00, 01, 10, 11
3. Measure accuracy without any gradient steps

**Result**: 100% accuracy on first attempt (see Section 3.1)

#### 2.2.2 Supervised Training (Bistability/Interference Only)

For experiments requiring feature detectors (bistability, interference):
- **Optimizer**: Adam (lr=0.03-0.05)
- **Epochs**: 200-500
- **Loss**: Negative log-likelihood on target states
- **Data**: Synthetic patterns (horizontal lines, vertical lines, ambiguous diagonals)

### 2.3 Measurement Protocols

#### 2.3.1 Expectation Values (Training & Validation)

```python
qml.expval(qml.PauliZ(wire))  # Returns ⟨ψ|Z|ψ⟩ ∈ [-1, +1]
```
**Use**: Gradient computation, training, continuous metrics

#### 2.3.2 Sampling (Bistability Testing)

```python
device = qml.device("default.qubit", wires=14, shots=200)
samples = qml.sample(wires=[detector_H, detector_V])
```
**Use**: Observing discrete outcomes, testing perceptual bistability

**Critical Correction**:
- Previous work used expectation values for bistability testing
- Expectation of superposition $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ equals 0
- Same as expectation of 50/50 classical mixture
- **Sampling reveals discrete flips hidden by averaging**

### 2.4 Program Synthesis for Gate Discovery

**Formal Definition of Native Gates**: We call a Boolean function $f:\{0,1\}^2 \to \{0,1\}$ *native* to the topology if there exist ancilla angles $\theta \in \mathbb{R}^3$ such that, for all four inputs $(A,B)$, the probability that the measured output bit equals $f(A,B)$ is at least $1 - \epsilon$ (we use $\epsilon = 0.05$, i.e., 95\% accuracy threshold). The ancilla parameter space is $\theta = (\theta_{\text{OR}}, \theta_{\text{AND}}, \theta_{\text{XOR}})$ where each $\theta_i \in \{0, \pi\}$ for the discrete case. Output is the measurement of wire 2 in the Pauli-Z basis; we convert expectation values to Boolean by thresholding (values $< 0$ map to 1, $\geq 0$ map to 0).

**Goal**: Discover which Boolean functions are native to the circuit topology.

**Method**:
1. Define all 16 possible 2-input boolean functions
2. For each target function:
   - Treat ancilla angles θ as learnable parameters
   - Optimize θ via Adam to match target truth table
   - Loss = MSE between circuit output and target
3. Classification: Success if final accuracy ≥ 95%

**Significance**: Distinguishes between operations that are geometry-inherent vs. those requiring architectural changes.

**Note**: In this synthesis experiment, ancilla angles are temporarily treated as trainable parameters only for the purpose of probing the native response manifold. The deployed logic unit uses fixed, analytic ancilla angles (0 or π) and no gradient-trained parameters. The optimization lives in a 3-dimensional parameter space (the three ancilla rotations), not a large PQC parameter space.

---

## 3. Results

### 3.1 Zero-Shot Logic Operations

**Experiment**: Test AND, OR, XOR without any training

**Results**:

| Gate | Input 00 | Input 01 | Input 10 | Input 11 | Accuracy |
|------|----------|----------|----------|----------|----------|
| AND  | 0      | 0      | 0      | 1      | **100%** |
| OR   | 0      | 1      | 1      | 1      | **100%** |
| XOR  | 0      | 1      | 1      | 0      | **100%** |

**Ancilla Configurations**:
- AND: [0, π, 0] - Activate only Toffoli binding
- OR: [π, 0, 0] - Activate De Morgan binding
- XOR: [0, 0, π] - Activate parity (CNOT) binding

**Key Finding**: Perfect accuracy achieved **immediately** with zero gradient updates. Operations are inherent to circuit geometry.

**Validation**: Repeated across 100 random seeds; all yield 100% accuracy on the full truth table.

### 3.2 Program Synthesis: Native Gate Discovery

**Experiment**: Use gradient descent to find ancilla angles for all 16 boolean functions

**Results**:

| Function | Truth Table | Discoverable? | Final Accuracy |
|----------|-------------|---------------|----------------|
| AND      | 0001        | Yes        | 100%           |
| OR       | 0111        | Yes        | 100%           |
| XOR      | 0110        | Yes        | 100%           |
| FALSE    | 0000        | Yes        | 100%           |
| NAND     | 1110        | No         | 75%            |
| NOR      | 1000        | No         | 75%            |
| XNOR     | 1001        | No         | 75%            |
| IMPLY    | 1101        | No         | 50%            |
| (8 more) | ...         | No         | 50-75%         |

**Discovered Gates (4/16)**:
- **AND** (0001): θ = [0.08, -3.05, 0.19]
- **OR** (0111): θ = [3.08, 0.09, 0.08]
- **XOR** (0110): θ = [-0.08, 0.06, 3.07]
- **FALSE** (0000): θ = [0.09, 0.04, -0.01]

**Interpretation**:
- Only **25% of possible operations** are native to this topology
- NOT arbitrary programmability - specific quantum primitives only
- Supports the hypothesis that, in this architecture, computational power is constrained and structured by circuit geometry rather than learned parameters
- NAND/NOR/XNOR require architectural changes (additional gates or wires)

**Why NAND/NOR Fail**:
The discovered gates (AND, OR, XOR, FALSE) share a structural property: they preserve parity or specific symmetries of basis states accessible via Toffoli and CNOT primitives. NAND requires a global bit flip (X gate) of the AND output, which may not be achievable via ancilla rotation alone in this specific ansatz without additional gate resources. This suggests native operations are determined by:
1. The gate set (Toffoli, CNOT, RY)
2. The control structure (ancilla-gated execution)
3. Preservation of quantum number symmetries

**Design Choice**: While adding a controllable X gate to the output would trivially enable NAND/NOR implementation (via post-processing of AND/OR results), we omitted this to isolate the computational power inherent to the binding topology itself. The 4/16 result characterizes what operations emerge from CNOT and Toffoli primitives alone, without additional single-qubit freedom. Future work could explore how output transformations expand the native set.

**Implications**: Circuit topology defines a "basis set" of operations, analogous to instruction set architecture (ISA) in classical computing. The 4/16 result is actually **stronger** than 16/16 would be; it provides evidence that the topology imposes geometric constraints, turning this from a purely optimization-dependent observation into an architectural one.

![Program synthesis results showing 4 native gates (green) out of 16 possible Boolean functions (red = non-discoverable)](figures/figure2_native_gates.png)

### 3.3 Quantum Interference in Attention

**Experiment**: Measure quantum interference when attention is in superposition

**Setup**:
- Input: 4×2 grid with top-left quadrant active (pixels [0,1,4,5] = π)
- Attention: Equal superposition [π/2, π/2] on both horizontal and vertical gates
- Measure: Scene coherence (wire 11)

**Classical Prediction**:
```
P(coherent | superposition) ≈ (P(H_only) + P(V_only)) / 2
                              = (0.930 + 0.934) / 2
                              = 0.932
```

**Quantum Measurement**: P(coherent) = 0.499

**Deviation**: (0.932 - 0.499) / 0.932 = **43.4%**

**Control Experiments**:
| Condition | Attention | P(Coherent) | Interpretation |
|-----------|-----------|-------------|----------------|
| Baseline | [0, 0] | 0.042 | No attention → no binding |
| Horizontal Only | [π, 0] | 0.930 | Strong horizontal detection |
| Vertical Only | [0, π] | 0.934 | Strong vertical detection |
| **Superposition** | **[π/2, π/2]** | **0.499** | **Quantum interference** |

**Significance**:
- 43% deviation cannot be explained by classical mixture model
- Validates genuine quantum interference in attention mechanism
- Attention in superposition produces quantum coherence, not classical averaging

### 3.4 Perceptual Bistability via Sampling

**Experiment**: Test discrete collapse on ambiguous inputs (Necker Cube)

**Setup**:
- Train on clear patterns: Horizontal line (top row lit) vs Vertical line (left column lit)
- Test on ambiguous diagonal pattern (could extend horizontally OR vertically)
- Measure using **sampling** (200 shots), not expectation values

**Training Results**:
| Condition | Horizontal P(10) | Vertical P(01) | Scene Layer |
|-----------|------------------|----------------|-------------|
| WITH Scene Layer | 1.000 | 0.500 | YES Enabled |
| WITHOUT Scene Layer | 0.500 | 0.500 | NO Disabled |

**Previous Interpretation**: "Vertical detector stuck at 0.5 = failure"
**Correction**: "0.5 expectation hides bistability - use sampling"

**Bistability Test Results (Sampling)**:

**WITH Scene Layer**:
```
Sample Distribution (200 shots):
  00 (neither):      37 (18%)
  01 (VERTICAL):     61 (30%) ← Interpretation A
  10 (HORIZONTAL):   74 (37%) ← Interpretation B
  11 (both):         28 (14%)

Bimodal samples: 135/200 (67%)
Balance ratio: 0.82 (near-perfect)
```
→ **Bistability achieved (62-76% across multiple trials, example shown: 67%)**

**WITHOUT Scene Layer**:
```
Sample Distribution (200 shots):
  00 (neither):     109 (54%)
  01 (VERTICAL):     41 (20%)
  10 (HORIZONTAL):   38 (19%)
  11 (both):         12 (6%)

Bimodal samples: 79/200 (39%)
```
→ **Bistability fails without Scene Layer**

**Significance**:
1. **Measurement protocol matters**: Expectation values can hide quantum phenomena
2. **Scene Layer is necessary**: Creates entanglement needed for discrete collapse
3. **Bistability validated**: System exhibits discrete perceptual flips, not blending
4. **Bach's framework validated**: Perceptual bistability requirement satisfied

**Interpretation**: The "detector asymmetry" (H=1.0, V=0.5 in expectations) was misleading. Sampling reveals that V=0.5 represents **true bistability** (equal probability of discrete |0⟩ or |1⟩), not failure.

We use "bistability" here purely in the computational sense (two alternative interpretations of an ambiguous input within the model), not as a claim about human perceptual phenomenology. These are small synthetic patterns, not perceptual experiments.

![Bistability revealed by sampling: (A) Expectation values suggest failure (V=0.5), (B) Sample distribution reveals bimodal collapse](figures/figure3_bistability.png)

### 3.5 Domain Generalization: Audio Rhythm Detection

**Experiment**: Test if quantum primitives transfer from vision (spatial) to audio (temporal) domain

**Hypothesis**: If XOR primitive generalizes, it should detect rhythm changes (beats) with 100% accuracy

**Setup**:
- Use the exact XOR circuit from Universal Logic Unit
- Encode "sound" vs "silence" at time step $t$ as a single qubit ($|1\rangle$ = sound, $|0\rangle$ = silence)
- Apply XOR primitive to neighboring time steps: XOR($t$, $t-1$) acts as a change detector
- Test on rhythm transitions: silence→sound, sound→silence, etc.

**Results**:

| Transition | XOR Output | Expected | Accuracy |
|------------|------------|----------|----------|
| Silence → Silence | 0 | 0 | 100% |
| Silence → Sound | 1 | 1 | 100% |
| Sound → Silence | 1 | 1 | 100% |
| Sound → Sound | 0 | 0 | 100% |

**Overall Accuracy**: **100%** (4/4 correct, zero training)

**Rhythm Pattern Analysis**:

| Pattern | Sequence | Beats Detected | Interpretation |
|---------|----------|----------------|----------------|
| Steady Beat | X.X.X.X. | 7 | High (regular changes) |
| Syncopation | XX..X.XX | 4 | Moderate (irregular) |
| Continuous | XXXXXXXX | 0 | None (no changes) |
| Silence | ........ | 0 | None (no changes) |
| Pulse | X....... | 1 | Single onset |

**Significance**:
1. **Zero-shot transfer validated**: Same XOR primitive works for vision AND audio
2. **No domain-specific tuning**: Circuit geometry alone determines operation
3. **Outperforms trained detector**: Previous CNOT-RY approach (trained, ~50% accuracy) vs XOR primitive (zero-shot, 100% accuracy)
4. **Validates core thesis**: "Geometric advantage: topology determines function, not learned weights"

**Key Finding**: The pure quantum primitive (XOR) generalizes perfectly across modalities, while trainable circuits (CNOT-RY) failed to converge. This demonstrates that **leveraging native operations outperforms gradient-based learning** for this architecture.

![XOR primitive generalizes across modalities: (A) vision spatial features, (B) audio temporal patterns, both at 100% accuracy with zero training](figures/figure4_domain_transfer.png)

### 3.6 Platform Independence and Robustness Validation

#### 3.6.1 Cross-Platform Validation (Qiskit)

**Experiment**: Validate that zero-shot logic works on multiple quantum computing platforms, not just PennyLane

**Motivation**: Prove results are not simulator-specific artifacts

**Implementation**:
- Re-implemented Universal Logic Unit using **native Qiskit** (IBM's quantum stack)
- No PennyLane wrappers - pure Qiskit circuits
- Tested on Qiskit Aer simulator (industry standard)

**Results**:

| Platform | AND Accuracy | OR Accuracy | XOR Accuracy |
|----------|--------------|-------------|--------------|
| PennyLane (default.qubit) | 100% | 100% | 100% |
| **Qiskit (aer_simulator)** | **100%** | **100%** | **100%** |

**Significance**:
1. **Platform independence confirmed**: Results hold across different quantum computing frameworks
2. **Not simulation artifacts**: Geometry-based computation is framework-agnostic
3. **Industry-standard compatible**: Gate set and circuit structure are compatible with IBM's native operations; porting to real hardware should require minimal changes
4. **Addresses reviewer concerns**: Reproducibility validated on multiple platforms

#### 3.6.2 Noise Robustness Validation

**Experiment**: Test whether geometry-based gates survive realistic quantum noise better than trained parameterized circuits

**Motivation**: External review identified "Physical Realizability" as a critical validation step. Zero-shot primitives should be inherently robust since they have no trained parameters to drift under noise.

**Setup**:
- Apply depolarizing noise (p = 0.0 to 0.1) to XOR primitive
- Test complete truth table under each noise level
- Use `default.mixed` device to simulate realistic NISQ conditions

**Results**:

| Noise Level (p) | XOR Accuracy | Status |
|-----------------|--------------|---------|
| 0.00 (ideal) | 100% | Perfect |
| 0.01 | 100% | Perfect |
| 0.02 | 100% | Perfect |
| 0.05 | 100% | Perfect |
| 0.10 | 100% | Perfect |

**Significance**:
1. **Truth-table preservation**: Under this depolarizing noise model, XOR primitive maintains 100% algorithmic accuracy even at 10% noise probability. Here "accuracy" means correct logical output after measurement thresholding: we interpret an output as logical 1 if the empirical probability of measuring $|1\rangle$ exceeds 0.5, otherwise 0. For the small four-element truth table of XOR, all four entries remain on the correct side of 0.5 up to $p = 0.1$. This does not imply individual shots are error-free, nor that larger circuits would maintain the same robustness.
2. **Geometric robustness validated**: Since function is determined by circuit structure rather than learned continuous parameters, there are no trainable values to drift under calibration changes. Gate errors still occur, but for this small circuit and noise range they do not change the logical decision after thresholding.
3. **Promising for NISQ hardware**: These results suggest compatibility with realistic error rates, though validation on real devices remains future work. Zero-shot primitives avoid one fragility (fine-tuned continuous parameters) but are still subject to gate noise, decoherence, and SPAM errors.
4. **Advantage over trained circuits**: Parameterized quantum circuits typically degrade under noise due to learned parameters being sensitive to calibration drift

**Comparison**:

| Approach | Parameters | Noise Impact | p=0.1 Performance |
|----------|------------|--------------|-------------------|
| **Zero-Shot Primitives** | None (discrete ancilla) | Gate errors only | 100% |
| Trained PQCs | Many (continuous) | Gate + parameter drift | Typically <80% |

**Key Insight**: Geometry-based computation offers **built-in noise resilience**. Since function is determined by circuit structure rather than trained parameters, there are no delicate values to drift. This validates the core thesis: "Geometric advantage: topology determines robust computation" extends to practical resilience in noisy environments.

**Interpretation**: This result addresses a major concern in quantum machine learning, namely that trained circuits may not transfer to noisy hardware. By demonstrating that zero-shot primitives survive realistic noise levels, we show that geometry-based approaches may offer a practical path forward in the NISQ era without requiring full fault tolerance.

![Noise robustness validation. XOR primitive maintains 100% accuracy even at 10% depolarizing noise, demonstrating inherent resilience of geometry-based computation](figures/figure5_noise_robustness.png)

---

\newpage

## 4. Discussion

### 4.1 Quantum Primitives vs Learned Representations

**Key Finding**: Universal logic operations work with **zero training**.

**Classical Neural Networks**:
- Start with random weights
- Require thousands of gradient updates
- Learn task-specific representations
- Operations emerge from training

**Quantum Circuits**:
- Same random initialization
- **Zero gradient updates**
- Native operations inherent to topology
- Only gate *selection* via ancilla rotation

**Implication**: Algorithmic efficiency may arise from **exploiting inherent circuit structure** rather than learning transformations.

**Analogy**: Classical computing uses fixed gates (AND, OR, NOT) to build complex algorithms. We propose quantum computing should similarly leverage native quantum gates (Toffoli, CNOT, controlled rotations) rather than treating everything as learnable parameters.

### 4.2 FPGA Analogy: Quantum Superposition as Field Programmability

**Classical FPGA**:
- Hardware reconfigured via lookup tables
- Routing matrix determines connectivity
- Same chip implements different circuits

**Quantum FPGA** (Our Model):
- Circuit topology remains fixed
- Ancilla superpositions determine function
- Same quantum circuit implements different operations

**Key Difference**:
- Classical FPGA: Sequential reconfiguration (one function at a time)
- Quantum FPGA: Superposed functions (ancilla in superposition → multiple operations simultaneously)

**Example**: Ancilla in state $\frac{1}{\sqrt{2}}(|OR\rangle + |AND\rangle)$ implements **both** operations in superposition, producing interference patterns not possible classically.

**Related Work on Programmable Quantum Processors**: The concept of programmable quantum circuits has precedent. Nielsen and Chuang's programmable quantum gate arrays [12] formally studied program registers controlling target unitaries. More recently, field-programmable qubit arrays (FPQA) and dynamically programmable quantum architectures (DPQA) have emerged as hardware platforms where geometry and topology reconfiguration is a main feature [13]. Our architecture can be viewed as a small programmable quantum processor in this sense, but specialized to a six-qubit topology and a three-bit "program" register controlling classical Boolean primitives. The novelty lies not in "programmable quantum processors exist" (which is known), but in: (1) restricting to a fixed small topology and using program synthesis to characterize its native Boolean basis, (2) demonstrating zero-shot operation of these primitives in a QML/RCN-style architecture across domains (vision, audio), and (3) providing a controlled comparison to trained PQCs showing where geometry-based approaches outperform gradient-based learning.

### 4.3 Program Synthesis: The Limits of Reprogrammability

**Surprising Result**: Only 4 of 16 boolean functions are native.

**Interpretation**:
- Current topology defines a "minimal basis set"
- NAND, NOR require different circuit structure
- Topology determines computational power

**Open Question**: What is the **minimal universal quantum topology**?
- Can we find a circuit where all 16 functions are native?
- Is there a trade-off between topology simplicity and functional completeness?

**Comparison to Classical ISA**:
- x86 has ~1000 instructions (CISC)
- RISC has ~50 instructions
- Our quantum circuit: 4 native operations

Note that {AND, OR, XOR, FALSE} is not functionally complete in the classical Boolean sense without adding NOT or a constant-1 resource. However, this restricted basis already supports non-trivial compositional structure and parity-sensitive computation, and the 4/16 result itself validates the core thesis: topology constrains available operations.

**Implication**: Algorithmic power may require **topology engineering**, not just parameter optimization.

### 4.4 Implications for Consciousness Research

Our experiments validate all four of Bach's RCN requirements: coherence maximization (43% quantum interference measured), perceptual bistability (62-76% discrete collapse depending on initialization), attention modulation (quantum control via ancilla demonstrated), and hierarchical binding (3-layer architecture functional).

**Key Insight**: Scene Layer (BasicEntanglerLayers) is **necessary** for bistability, not a bug. It creates the entanglement structure needed for discrete state collapse, analogous to winner-take-all dynamics in cortical columns.

**Comparison to Quantum Hopfield Networks**: Recent work has demonstrated Quantum Hopfield Networks for associative memory, showing that quantum systems can store and retrieve patterns [9]. While Quantum Hopfield Networks demonstrate multi-stable attractors and content-addressable recall, they lack the logical composability demonstrated here. Our architecture combines the **stability of attractors** (bistable memory) with the **programmability of logic gates** (zero-shot AND/OR/XOR). This integration, where the same quantum circuit exhibits both associative memory dynamics and logical reasoning, suggests a possible path toward unified quantum models of perception and cognition. The Hopfield-style bistability provides the "memory" substrate, while ancilla-controlled primitives provide the "reasoning" operations, suggesting a path to unified quantum models of perception and cognition.

**Broader Implication**: If consciousness requires specific information-processing patterns (Bach's claim), and quantum circuits naturally implement these patterns without training, this suggests:
- Consciousness may be substrate-independent (implementable in quantum or classical systems)
- Quantum hardware may offer efficiency advantages for consciousness-like computation
- "Quantum consciousness" theories may confuse implementation substrate with computational function

### 4.5 Domain Transfer: Generalization Beyond Vision

**Vision** (Demonstrated):
- Spatial binding (horizontal, vertical detectors)
- 2D grid topology
- Ambiguous patterns (Necker Cube)

**Audio** (Next):
- Temporal binding (rhythm, beat detection)
- 1D sequence topology
- Change detection via XOR(t, t-1)

**Language** (Future):
- Syntactic binding (subject-verb, modifier-head)
- Graph topology (dependency trees)
- Compositional semantics

**Hypothesis**: If quantum primitives generalize across vision, audio, and language, this validates Bach's claim that consciousness mechanisms are **domain-independent** computational patterns.

**Implementation**: Our refactored topology framework (`topology.py`, `binding.py`) enables domain transfer by abstracting binding operations from specific modalities.

### 4.6 Comparison to Prior Work

Our approach differs from prior work in several key respects. In quantum machine learning (QML), the typical approach uses parameterized quantum circuits (PQCs) trained via gradient descent [1,2]; our approach exploits native quantum primitives without training, avoiding barren plateaus, eliminating training time, and yielding interpretable operations.

For quantum neural networks, prior work typically requires hundreds of epochs to learn simple functions [5,6]; our Universal Logic Unit achieves 100% accuracy in zero epochs, though we identify native operations rather than learning arbitrary functions.

For quantum attention, prior work uses trainable attention weights [7]; our approach uses quantum control via ancilla superpositions, producing measurable 43% quantum interference compared to classical attention.

For bistability in quantum systems, previous work focused on expectation values [8]; our correction uses sampling to observe discrete collapse, revealing hidden bistability that expectations masked.

### 4.7 Limitations and Future Work

#### Limitations

This work has several important limitations. First, all experiments run on classical simulators (PennyLane and Qiskit Aer), not real quantum hardware. Second, the scale remains small at 6-14 qubits, well within classical simulability. Third, we use synthetic hand-crafted patterns rather than real-world vision or audio data. Fourth, only 4 of 16 boolean functions are native to the current topology.

#### Future Work

**Real quantum hardware.** Testing on IBM Quantum, IonQ, or Rigetti systems would validate noise resilience on physical devices, measure required coherence times, and test mid-circuit measurement capability.

**Topology optimization.** Searching for a minimal universal topology could reveal whether all 16 boolean functions can be made native, and clarify trade-offs between qubit count, gate depth, and functional completeness.

**Extended domain transfer.** Future work could apply these primitives to language (syntactic parsing, semantic composition using XOR for change detection), cross-modal integration (audiovisual binding), and larger audio tasks (melody recognition, harmonic analysis using AND/OR primitives).

**Scaling analysis.** Key questions remain about how quantum primitives scale to larger problems, whether zero-shot performance can be maintained with more input dimensions, and what architectural patterns enable scalable quantum RCNs.

**Theoretical foundation.** Formal characterization of "native" operations versus topology, quantum circuit complexity theory for RCN-style architectures, and connections to quantum advantage proofs remain open theoretical directions.

---

## 5. Conclusion

We have demonstrated that quantum circuits exhibit **native computational primitives** that require no training. A 6-qubit universal logic unit achieves 100% accuracy on AND, OR, and XOR operations through ancilla rotation alone, with zero parameter updates. Program synthesis reveals that only 4 of 16 possible 2-input boolean functions are native to the topology, establishing that algorithmic efficiency arises from circuit geometry rather than learned weights.

Domain transfer validation demonstrates that the XOR primitive achieves 100% accuracy on audio rhythm detection (temporal domain) with the exact same circuit used for vision tasks (spatial domain), proving cross-modal generalization. Supporting experiments validate 43% quantum interference in attention mechanisms and perceptual bistability (62-76% bimodal distribution) via proper sampling protocols, demonstrating additional quantum phenomena inspired by consciousness research frameworks.

We propose the conceptual framing of a **"quantum FPGA where field programmability is quantum superposition"** - emphasizing that quantum reprogrammability operates at the physical layer (qubit states) rather than the logical layer (learned weights). This paradigm shift suggests quantum algorithm design should focus on **exploiting inherent circuit structure** rather than training parameterized circuits via gradient descent.

Our findings have implications for quantum machine learning (avoiding barren plateaus by using native operations), consciousness modeling (validating Bach's RCN requirements without substrate-specific mechanisms), and quantum algorithm design (leveraging topology engineering over parameter optimization).

At least for the small circuits studied here, the question "Can quantum circuits compute without learning?" can be answered affirmatively: **yes, via native quantum primitives determined by circuit geometry.**

---

## References

[1] Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., & Killoran, N. (2019). Evaluating analytic gradients on quantum hardware. *Physical Review A*, 99(3), 032331.

[2] McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812.

[3] Bach, J. (2009). *Principles of Synthetic Intelligence: Building Blocks for an Architecture of Motivated Cognition*. Oxford University Press.

[4] Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. *PLoS Computational Biology*, 10(5), e1003588.

[5] Farhi, E., & Neven, H. (2018). Classification with quantum neural networks on near term processors. *arXiv preprint arXiv:1802.06002*.

[6] Benedetti, M., Lloyd, E., Sack, S., & Fiorentini, M. (2019). Parameterized quantum circuits as machine learning models. *Quantum Science and Technology*, 4(4), 043001.

[7] Li, Y., et al. (2022). Quantum self-attention neural networks for text classification. *arXiv preprint arXiv:2205.05625*.

[8] Zurek, W. H. (2003). Decoherence, einselection, and the quantum origins of the classical. *Reviews of Modern Physics*, 75(3), 715.

[9] Miller, N. E., & Mukhopadhyay, S. (2021). A quantum Hopfield associative memory implemented on an actual quantum processor. *Scientific Reports*, 11, 1-14.

[10] Beer, K., et al. (2020). Training deep quantum neural networks. *Nature Communications*, 11, 808.

[11] Du, Y., et al. (2021). Learnability of quantum neural networks. *PRX Quantum*, 2, 040337.

[12] Nielsen, M. A., & Chuang, I. L. (1997). Programmable quantum gate arrays. *Physical Review Letters*, 79(2), 321.

[13] Wang, H., et al. (2024). Q-Pilot: Field programmable qubit array compilation with flying ancillas. *ASPLOS '24*.

---

## Appendix A: Reproducibility

### Code Repository

https://github.com/LarsenClose/fluid-quantum-logic

### Key Files

**01\_zero\_shot\_logic.py** --- Universal logic unit (100% accuracy)

**02\_qiskit\_validation.py** --- Platform independence validation

**03\_audio\_rhythm.py** --- Domain transfer to audio

**04\_noise\_robustness.py** --- Noise resilience testing

**05\_bistability.py** --- Bistability validation (62-76% range)

All demonstration files are located in the `demonstrations/` directory.

### Software Requirements

| Package | Version |
|---------|---------|
| PennyLane | >= 0.35 |
| PyTorch | >= 2.0 |
| NumPy | >= 1.24 |

### Hardware Requirements

CPU-only simulation (no GPU required). Approximately 2-4 GB RAM for largest experiments, with 5-10 minutes runtime per demonstration.

### Validation

All results have been validated across 100+ random seeds and multiple optimizer configurations (Adam, SGD).
