"""
Demonstration: Domain Transfer (Audio Rhythm Detection)

Demonstrates that the same XOR primitive used for spatial logic
generalizes to temporal data (audio) without retraining.
XOR(t, t-1) functions as a perfect beat/change detector.
"""

import pennylane as qml
from pennylane import numpy as np

# Initialize device once for performance
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def xor_circuit(a, b):
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    # The Exact XOR Primitive from Logic Unit
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])
    return qml.expval(qml.PauliZ(2))

def xor_beat_detector(prev_t, curr_t):
    """
    Pure XOR primitive applied to temporal data.
    Input: Amplitude at t-1, Amplitude at t
    Output: 1 if Change (Beat), 0 if Sustain/Silence
    """
    result = xor_circuit(prev_t, curr_t)
    prob = (1 - result) / 2
    return 1 if prob > 0.5 else 0

def test_rhythm_patterns():
    print("\nTesting Rhythm Sequences (8-step patterns):")
    print("-" * 80)
    # Formatting header to match the paper table structure
    print(f"{'Pattern':<15} {'Sequence':<10} {'Beats':<8} {'Interpretation'}")
    print("-" * 80)

    # Order, Labels, and Descriptions aligned perfectly with Paper Table 3.5
    patterns = {
        'Steady Beat': ([np.pi, 0, np.pi, 0, np.pi, 0, np.pi, 0], 7, "High (regular changes)"),
        'Syncopation': ([np.pi, np.pi, 0, 0, np.pi, 0, np.pi, np.pi], 4, "Moderate (irregular)"),
        'Continuous':  ([np.pi]*8,                                   0, "None (no changes)"),
        'Silence':     ([0]*8,                                       0, "None (no changes)"),
        'Pulse':       ([np.pi] + [0]*7,                             1, "Single onset")
    }

    all_correct = True
    for name, (seq, expected_beats, desc) in patterns.items():
        detected_beats = 0
        # t goes from 1 to 7 (comparing t with t-1)
        for t in range(1, len(seq)):
            detected_beats += xor_beat_detector(seq[t-1], seq[t])

        # Visualizer (Matches the 'Sequence' column visually)
        vis = "".join(['X' if x > 0 else '.' for x in seq])

        # Check correctness
        match = "✓" if detected_beats == expected_beats else "✗"

        print(f"{name:<15} {vis:<10} {detected_beats:<8} {desc} {match}")

        if detected_beats != expected_beats:
            all_correct = False

    return all_correct

def main():
    print("="*80)
    print("DOMAIN TRANSFER: AUDIO RHYTHM DETECTION")
    print("="*80)

    # 1. Primitive Test
    print("\nPrimitive Validation (XOR t, t-1):")
    transitions = [
        (0, 0, 0, "Silence->Silence"),
        (0, np.pi, 1, "Silence->Sound"),
        (np.pi, 0, 1, "Sound->Silence"),
        (np.pi, np.pi, 0, "Sound->Sound")
    ]

    passed_prim = True
    for prev, curr, exp, label in transitions:
        res = xor_beat_detector(prev, curr)
        mark = "✓" if res == exp else "✗"
        print(f"{label:<16} | Out: {res} | Exp: {exp} | {mark}")
        if res != exp: passed_prim = False

    # 2. Sequence Test
    passed_seq = test_rhythm_patterns()

    print("\n" + "="*80)
    if passed_prim and passed_seq:
        print("✅ SUCCESS: Quantum primitive generalizes to temporal domain.")
        print("   Zero retraining required. Same circuit, different modality.")
        return True
    else:
        print("❌ FAILURE: Domain transfer failed.")
        return False

if __name__ == "__main__":
    success = main()
    # Assertion for automated testing
    assert success, "Audio domain transfer validation failed"
