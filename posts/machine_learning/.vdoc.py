# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| output: false

# Import necessary libraries
import qiskit
import numpy as np
from numpy import pi
```
#
#
#
#
#
#
#
#
#
#
#
#
#
A = np.array([[1, -1/3], [-1/3, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"eigenvalues:\n{np.round(eigenvalues,2)}")
print(f"eigenvectors:\n{np.round(eigenvectors,2)}")

# Verify the eigenvalue-eigenvector relationship
assert np.allclose(np.dot(A, eigenvectors.T[0]), eigenvalues[0] * eigenvectors.T[0])
assert np.allclose(np.dot(A, eigenvectors.T[1]), eigenvalues[1] * eigenvectors.T[1])
#
#
#
### The eigenvalue and eigenvectors given by paper
A = np.array([[1, -1/3], [-1/3, 1]])
b = np.array([0,1])
eigenvalues = [2/3, 4/3]
eigenvectors = np.array([[-1/(2**0.5), -1/(2**0.5)],[-1/(2**0.5), 1/(2**0.5)]])

assert np.allclose(np.dot(A, eigenvectors.T[0]), eigenvalues[0] * eigenvectors.T[0])
assert np.allclose(np.dot(A, eigenvectors.T[1]), eigenvalues[1] * eigenvectors.T[1])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
N_b = 2 # number of variables (b.shape[0])
n_b = np.log2(N_b) # number of qubits in b-register
# n_b = 1 # N_b = 2^(n_b), n_b = np.log2(N_b)
n_c = 1 # number of qubits in c-register
N_c = 2 # N_c = 2^(n_c)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import Aer

qr_b = QuantumRegister(n_b, name='b')  # b-register
qr_c = QuantumRegister(n_c, name='c')  # c-register
qr_a = QuantumRegister(1, name='a')    # a-register
qc = QuantumCircuit(qr_c, qr_b, qr_a)  # complete circuit
# Initialize the quantum circuitz
print("Initial quantum circuit:")
# print(qc.draw(output='text'))
qc.draw(output='mpl')
# latex_source = qc.draw(output='latex_source')
# print(latex_source)
#
#
#
#
#
#
#

# normalize b to make sure the aggregate possibility is 1
norm_b = b / np.linalg.norm(b)
qc.initialize(norm_b, qr_b)  # Apply to b-register


#----------Print states--------------------------------
# Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')

# Transpile the circuit for the simulator
transpiled_circuit = transpile(qc, simulator)

# Run the transpiled circuit
result = simulator.run(transpiled_circuit).result()

# Extract the statevector
statevector = result.get_statevector(qc)
print("Statevector after initialization:", statevector)

# Optional: Visualize the circuit (if needed)
qc.draw(output='mpl')

#----------Print states--------------------------------

#
#
#
#
def state_prep(mat_A, vec_b, n_c=1):
  N_b = vec_b.shape[0]
  n_b = np.round(np.log2(N_b))
  qr_b = QuantumRegister(n_b, name='b')  # b-register
  qr_c = QuantumRegister(n_c, name='c')  # c-register
  qr_a = QuantumRegister(1, name='a')    # a-register
  qc = QuantumCircuit(qr_c, qr_b, qr_a)  # complete circuit

  norm_b = vec_b / np.linalg.norm(vec_b)
  qc.initialize(norm_b, qr_b)
  return qc

mat_A = np.array([[1, -1/3], [-1/3, 1]])
vec_b = np.array([0,1])

psi_1 = state_prep(mat_A, vec_b, n_c=1)
psi_1.draw(output='mpl')
#
#
#
#
#
#
