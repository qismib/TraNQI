File and folder description

CLASSICAL: 
This folder contains two python files with the definition of a classical Neural Network for MNIST data recognition, binary (0-1) and 0-9 handwritten digits.
The file Training_Testing.py contains the training loop and the validation function definition, to be imported in the previous files.

QUANTUM_NN.py: 
Qiskit program with just a few changes. Binary recognition using an hybrid neural network. Quantum circuit made by single qibit and single trainable parameter. 

N_Qubit: 
Binary recognition using an arbitrary number of qubits. Gor each gate the gates are implemented as before, with an Hadamard gate, RY(trainable parameter) and a measurement. The circuit returns the expectation values for every possible combination of qubits (ex for N_qubits=2: 00, 01, 10, 11). 

This folder contains three python files: N_qubit.py that is the main file, QC_N_qubits.py that contains the definition of the quantum circuit and the extended forward and backward methods of Pytorch Function needed for the learning process. Training_Testing.py contains as in 'Classical' the training loop and the validation function definition. Result_N_qubit.txt contains the results of the training loop (with the update of the loss function), the accuracy test and the time used to complete the training process. 

N_trainable_parameters: 
Binary recognition using a 3D rotation, U gate and three trainable parameters. The circuit returns the expectation value of 0-1. 

