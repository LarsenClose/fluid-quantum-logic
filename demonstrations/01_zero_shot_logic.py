import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

"""
Demonstration: Universal Logic Unit (Zero-Shot)

Implements a 6-qubit quantum logic unit capable of AND, OR, and XOR operations
without any parameter updates. Uses De Morgan's Law (A OR B ≡ NOT(NOT A AND NOT B))
to implement OR using the AND topology, proving universality via circuit geometry.
"""

NUM_WIRES = 6
DEV = qml.device("default.qubit", wires=NUM_WIRES)

class UniversalLogicUnit(nn.Module):
    """
    Universal Logic Unit.

    All gates use inherent geometric primitives:
    - AND: Toffoli primitive
    - OR:  Toffoli primitive + Input/Output Inversion (De Morgan)
    - XOR: CNOT parity primitive
    """

    def __init__(self):
        super().__init__()
        # Ancilla parameters are discrete logic choices, not trained weights
        self.w_ancilla = nn.Parameter(torch.zeros(3), requires_grad=False)

    def forward(self, a, b, gate_select):
        return self.q_forward(a, b, gate_select)

    @qml.qnode(DEV, interface="torch")
    def q_forward(a, b, gate_select):
        # 1. Ancilla Control (Gate Selection)
        qml.RY(gate_select[0], wires=3)  # OR
        qml.RY(gate_select[1], wires=4)  # AND
        qml.RY(gate_select[2], wires=5)  # XOR

        # 2. Input Encoding
        qml.RX(a, wires=0)
        qml.RX(b, wires=1)

        # 3. Geometric Primitives
        def or_primitive():
            # De Morgan Implementation: Invert inputs -> AND -> Invert output
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.Toffoli(wires=[0, 1, 2])
            qml.PauliX(wires=0); qml.PauliX(wires=1) # Reset inputs
            qml.PauliX(wires=2) # Invert output

        def and_primitive():
            qml.Toffoli(wires=[0, 1, 2])

        def xor_primitive():
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 2])

        # 4. Controlled Execution
        qml.ctrl(or_primitive, control=3)()
        qml.ctrl(and_primitive, control=4)()
        qml.ctrl(xor_primitive, control=5)()

        return qml.expval(qml.PauliZ(2))

def test_gate(model, gate_name, context, expected_outputs):
    print(f"\nTesting {gate_name} Gate:")
    print(f"{'Input':<8} | {'Prob(1)':<10} | {'Exp':<5} | {'Status'}")
    print("-" * 40)

    inputs = [(0,0), (0,1), (1,0), (1,1)]
    correct = 0
    context_tensor = torch.tensor(context, dtype=torch.float32)

    for i, (bit_a, bit_b) in enumerate(inputs):
        # Logic to Angle: 0 -> 0, 1 -> π
        angle_a = torch.tensor(np.pi if bit_a else 0.0)
        angle_b = torch.tensor(np.pi if bit_b else 0.0)

        with torch.no_grad():
            result = model(angle_a, angle_b, context_tensor)
            # Map [-1, 1] to [0, 1]
            prob_one = (1 - result.item()) / 2
            prediction = 1 if prob_one > 0.5 else 0

            is_correct = (prediction == expected_outputs[i])
            if is_correct: correct += 1

            mark = "✓" if is_correct else "✗"
            print(f"{bit_a},{bit_b:<6} | {prob_one:.3f}      | {expected_outputs[i]:<5} | {mark}")

    accuracy = (correct / 4) * 100
    print(f"Accuracy: {accuracy:.0f}%")
    return accuracy

def main():
    print("="*60)
    print("FLUID QUANTUM LOGIC: ZERO-SHOT DEMONSTRATION")
    print("="*60)

    model = UniversalLogicUnit()

    # Gate Definitions: [θ_OR, θ_AND, θ_XOR]
    gates = {
        "AND": ([0.0, np.pi, 0.0], [0, 0, 0, 1]),
        "OR":  ([np.pi, 0.0, 0.0], [0, 1, 1, 1]),
        "XOR": ([0.0, 0.0, np.pi], [0, 1, 1, 0]),
    }

    results = {}
    for name, (angles, expected) in gates.items():
        results[name] = test_gate(model, name, angles, expected)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("-" * 60)
    all_passed = True
    for name, acc in results.items():
        print(f"{name:<5}: {acc:.0f}% accuracy")
        if acc < 100: all_passed = False

    if all_passed:
        print("\n✅ SUCCESS: All logic gates achieved 100% accuracy.")
        print("   No gradient updates were performed.")
        return True
    else:
        print("\n❌ FAILURE: Some gates failed to generalize.")
        return False

if __name__ == "__main__":
    success = main()
    # Assertion for automated testing
    assert success, "Zero-shot logic validation failed"
