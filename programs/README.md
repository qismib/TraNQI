File and folder description

CLASSICAL: 
This folder contains two python files with the definition of a classical Neural Network for MNIST data recognition, binary (0-1) and 0-9 handwritten digits.
The file Training_Testing.py contains the training loop and the validation function definition, to be imported in the previous files.

QUANTUM_NN.py: 
Qiskit program with just a few changes. Binary recognition using an hybrid neural network. Quantum circuit made by single qibit and single trainable parameter. 

N_QUBITS: 
Binary recognition using a quantum circuit made of different qubits with independent parametrized r-y rotation. The number of circuit can be chosen arbitrarly. Run both on qasm_simulator(accuracy of 99.7%) and on ibmq_athens(99.6%). Accuracy_vs_std contains the accuracy percentage obtained with different values of standard deviation of validation set of data, both for not-noisy training data and noisy training data (std_dev = 0.2). 
Also used for not-binary recognition (02- 04): the loss function decreases but there is not convergence: accuracy of 95.5% and 87.7% respectively.

U3: 
Binary recognition using a 3D rotation, U gate and three trainable parameters. The circuit returns the expectation value of |0>-|1>. Model trained both on simulator and on ibmq_athens (accuracy 99.9%) without big differences (U3_compare). Good recognition on noisy data (accuracy_vs_std) both with not-noisy training data and with noisy training data (std_dev = 0.3)

ENTANGLEMENT:
Binary recognition with entangled states circuit (3 qubits, CNOT gate between couples of qubits 01, 12 and parametrized R_z for each of them). Accuracy obtained: 99.8% on qasm simulator and 99.7% on ibmq_santiago. Good recognition on noisy data with noisy training data (std_dev = 0.1)

QAOA: 
Circuit inspired by QAOA optimization algorithm, depending on two parameters. Accuracy values of 99.4% and of 98.9% on qasm simulator and ibmq_santiago respectively. 


It has been necessary to run the last two circuits (wich involved entangled states) on quantum computer with higher Quantum Volume (>=32) to obtain a correct convergence of the loss function and an acceptable accuracy on validation data.

