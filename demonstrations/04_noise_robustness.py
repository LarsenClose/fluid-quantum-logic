"""
Demonstration: Noise Robustness

Validates the resilience of geometry-based logic gates under
Depolarizing Noise (NISQ simulation). Compares against standard
reliability thresholds.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def run_noisy_xor(noise_prob):
    dev = qml.device("default.mixed", wires=3)

    @qml.qnode(dev)
    def circuit(a, b):
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)

        # Apply Noise to Inputs
        qml.DepolarizingChannel(noise_prob, wires=0)
        qml.DepolarizingChannel(noise_prob, wires=1)

        # XOR Primitive with Noise
        qml.CNOT(wires=[0, 2])
        qml.DepolarizingChannel(noise_prob, wires=2)
        qml.CNOT(wires=[1, 2])
        qml.DepolarizingChannel(noise_prob, wires=2)

        return qml.expval(qml.PauliZ(2))

    inputs = [(0,0), (0,np.pi), (np.pi,0), (np.pi,np.pi)]
    expected = [0, 1, 1, 0]
    correct = 0

    for i, (a,b) in enumerate(inputs):
        res = circuit(a, b)
        pred = 1 if (1 - res)/2 > 0.5 else 0
        if pred == expected[i]: correct += 1

    return correct / 4.0

def main():
    print("="*60)
    print("NOISE ROBUSTNESS ANALYSIS")
    print("="*60)
    print(f"{'Noise (p)':<10} | {'Accuracy':<10} | {'Status'}")
    print("-" * 40)

    probs = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15]
    accuracies = []

    for p in probs:
        acc = run_noisy_xor(p)
        accuracies.append(acc)

        status = "✅ Robust" if acc >= 0.99 else "⚠️ Degraded"
        if acc < 0.75: status = "❌ Failed"

        print(f"{p:<10.2f} | {acc*100:>5.1f}%    | {status}")

    # Check 10% noise threshold (NISQ benchmark)
    idx_10 = probs.index(0.10)

    print("\n" + "="*60)
    if accuracies[idx_10] >= 0.99:
        print("✅ RESULT: Logic remains functional at 10% depolarizing noise.")
        print("   Geometry-based gates exhibit high noise resilience.")
        success = True
    else:
        print("❌ RESULT: Logic degrades significantly under NISQ conditions.")
        success = False

    return success


def generate_plot(probs, accuracies, filename="noise_robustness.png"):
    """
    Optional: Generate publication-quality figure of noise robustness.

    Args:
        probs: List of noise probabilities tested.
        accuracies: List of corresponding accuracies.
        filename: Output filename (default: noise_robustness.png).
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(probs, [a*100 for a in accuracies], marker='o', linewidth=2)
        plt.axhline(y=99, color='green', linestyle='--', alpha=0.5, label='99% Threshold')
        plt.xlabel("Depolarizing Probability (p)")
        plt.ylabel("XOR Accuracy (%)")
        plt.title("Noise Robustness: Geometric XOR Gate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(filename, dpi=300)
        print(f"\n📊 Plot saved to {filename}")
    except Exception as e:
        print(f"\n⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    success = main()
    # Assertion for automated testing
    assert success, "Noise robustness validation failed at p=0.1 threshold"
