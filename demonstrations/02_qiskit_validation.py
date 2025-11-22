"""
Demonstration: Cross-Platform Validation (Qiskit)

Validates that the Zero-Shot Logic Primitives function identically on
IBM's Qiskit stack (Aer Simulator), confirming results are not specific
to PennyLane's state-vector simulation.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np

def qiskit_xor(a, b):
    qc = QuantumCircuit(3, 1)
    # Encoding
    if a: qc.x(0)
    if b: qc.x(1)
    # Primitive
    qc.cx(0, 2)
    qc.cx(1, 2)
    # Measure
    qc.measure(2, 0)
    return qc

def qiskit_and(a, b):
    qc = QuantumCircuit(3, 1)
    if a: qc.x(0)
    if b: qc.x(1)
    # Primitive (Toffoli)
    qc.ccx(0, 1, 2)
    qc.measure(2, 0)
    return qc

def qiskit_or(a, b):
    qc = QuantumCircuit(3, 1)
    if a: qc.x(0)
    if b: qc.x(1)
    # Primitive (De Morgan + Toffoli)
    qc.x(0); qc.x(1) # Invert Inputs
    qc.ccx(0, 1, 2)  # AND
    qc.x(0); qc.x(1) # Reset Inputs
    qc.x(2)          # Invert Output
    qc.measure(2, 0)
    return qc

def run_test(gate_name, circuit_func, expected_table):
    simulator = Aer.get_backend('aer_simulator')
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    correct = 0

    print(f"\nValidating {gate_name} on Qiskit Aer:")
    print("-" * 40)

    for i, (a, b) in enumerate(inputs):
        qc = circuit_func(a, b)
        transpiled = transpile(qc, simulator)
        result = simulator.run(transpiled, shots=1024).result()
        counts = result.get_counts()

        # Get most frequent measurement
        measured = int(max(counts, key=counts.get))
        expected = expected_table[i]

        is_correct = (measured == expected)
        if is_correct: correct += 1

        mark = "✓" if is_correct else "✗"
        print(f"In: {a},{b} | Out: {measured} | Exp: {expected} | {mark}")

    return (correct / 4) * 100

def main():
    print("="*60)
    print("PLATFORM INDEPENDENCE VALIDATION")
    print("="*60)

    tests = {
        "AND": (qiskit_and, [0, 0, 0, 1]),
        "OR":  (qiskit_or,  [0, 1, 1, 1]),
        "XOR": (qiskit_xor, [0, 1, 1, 0])
    }

    results = {}
    for name, (func, table) in tests.items():
        results[name] = run_test(name, func, table)

    print("\n" + "="*60)
    print("QISKIT VALIDATION SUMMARY")
    print("-" * 60)
    for name, acc in results.items():
        print(f"{name}: {acc:.0f}%")

    all_passed = all(acc == 100 for acc in results.values())
    if all_passed:
        print("\n✅ CONFIRMED: Logic primitives are platform-independent.")
        return True
    else:
        print("\n❌ FAILURE: Platform-specific discrepancies detected.")
        return False

if __name__ == "__main__":
    success = main()
    # Assertion for automated testing
    assert success, "Qiskit platform validation failed"
