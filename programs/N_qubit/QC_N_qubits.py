#defining a circuit with more parameters

#IMPLEMENTING QUANTUM CIRCUIT, THE SAME AS QISKIT ONE
#the numebr of qubits is defied as a parameter of the circuit

import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *

import itertools

#creating a list of all possible outputs of a quantum circuit, used for the expectation value
def create_QC_OUTPUTS(n_qubits):
    measurements = list (itertools.product([1,0], repeat = n_qubits))
    return [''.join([str(bit) for bit in measurement]) for measurement in measurements]
#itertools does the Cartesian product of passed vector




# n_qubit trainable parameter3
#Hadamard gate (superposition) + Rotation_Y (trainable parameter) + Measure_Z. Returning the expectation value over n_shots
class QuantumCircuit:
    """
    This class provides an interface to interact with our Quantum Circuit
    """

    def __init__(self, n_qubits, backend, shots):

        #-----Circuit definition
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.parameters = qiskit.circuit.ParameterVector('parameters', n_qubits)

        all_qubits = [i for i in range (n_qubits)]#qubits vector

        self._circuit.h(all_qubits)#over all the qubits
        self._circuit.barrier()
        for k in range (n_qubits):
            self._circuit.ry(self.parameters[k], k)
        self._circuit.measure_all()
        #-----

        self.backend = backend
        self.shots = shots
        self.QC_OUTPUTS =  create_QC_OUTPUTS(n_qubits)

    def expectation_Z_parameters(self,counts, shots, n_qubits):
        expects = np.zeros(n_qubits)
        i = 0
        for key in counts.keys():
             percentage = counts[key]/shots
             if (i < 3):
                 expects[i] += percentage
                 i += 1
             else:
                 i = 0
        return expects

    def expectation_Z_counts(self, counts, shots, n_qubits):
        expects = np.zeros(len(self.QC_OUTPUTS))

        for k in range(len(self.QC_OUTPUTS)):
            key = self.QC_OUTPUTS[k]
            perc = counts.get(key, 0) / shots
            expects[k] = perc
        return expects


    def run(self, thetas):
        #acting on a simulator
        thetas = thetas.squeeze()
#        print(thetas)
        p_circuit = self._circuit.bind_parameters({self.parameters[k]: thetas[k].item() for k in range(self.n_qubits)})
        job_sim = qiskit.execute(p_circuit,
                                 self.backend,
                                 shots=self.shots)

        result = job_sim.result()
        counts = result.get_counts(p_circuit)
#        print('C: ', counts)


#        expectation_parameters = self.expectation_Z_parameters(counts, self.shots, self.n_qubits)
        expectation_counts = self.expectation_Z_counts(counts, self.shots, self.n_qubits)
        return expectation_counts

#implementing forward and backward function to be used in the quantum layer

import torch
from torch.autograd import Function

#extending autograd functions for a quantum layer(forward and backward)

class HybridFunction(Function):
    """Hybrid quantum-classical function definition"""

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """Forward pass computation"""

        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        # context variable (it may take multiple values and return them related to the context). Used to keep track for backpropagation

        expectation_z = ctx.quantum_circuit.run(input)  # evaluating model with trainable parameter
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)  # saves a given tensor for a future call to backward (trainable parameter and the result obtained)
        #input = parametri che passo(3), result = risultati del calcolo(2**n_qubit)

        return result

    @staticmethod
    def backward(ctx, grad_output): #grad_output os previous gradient

        """Backward computation"""

        input, expectation = ctx.saved_tensors #evaluated in forward
        input = torch.reshape(input, (-1,))
        gradients = torch.Tensor()

        #iterating to evaluate gradient
        for k in range(len(input)):
            #shifting parameters
            shift_right, shift_left = input.detach().clone(), input.detach().clone()
            shift_right[k] += ctx.shift
            shift_left[k] -= ctx.shift

            # evaluating model after shift
            expectation_right = ctx.quantum_circuit.run(shift_right)
            expectation_left = ctx.quantum_circuit.run(shift_left)

            #evaluating gradient with finite difference formula
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])

            gradients = torch.cat((gradients, gradient.float()))

        result = gradients.float() * grad_output.float()

        return (result).T, None, None
