File and folder description

CLASSICAL: 
This folder contains two python files with the definition of a classical Neural Network for MNIST data recognition, binary (0-1) and 0-9 handwritten digits.
The file Training_Testing.py contains the training loop and the validation function definition, to be imported in the previous files.

QUANTUM_NN.py: 
Qiskit program with just a few changes. Binary recognition using an hybrid neural network. Quantum circuit made by single qibit and single trainable parameter. 

N_Qubit: 
Binary recognition using an arbitrary number of qubits. The gates are implemented as shown in figure: 

        ┌───┐ ░ ┌───────────────────┐ ░ ┌─┐
   q_0: ┤ H ├─░─┤ RY(parameters[0]) ├─░─┤M├──────
        ├───┤ ░ ├───────────────────┤ ░ └╥┘┌─┐
   q_1: ┤ H ├─░─┤ RY(parameters[1]) ├─░──╫─┤M├───
        ├───┤ ░ ├───────────────────┤ ░  ║ └╥┘┌─┐
   q_2: ┤ H ├─░─┤ RY(parameters[2]) ├─░──╫──╫─┤M├
        └───┘ ░ └───────────────────┘ ░  ║  ║ └╥┘
meas: 3/═════════════════════════════════╩══╩══╩═

This folder contains three python files: N_qubit.py that is the main file, QC_N_qubits.py that contains the definition of the quantum circuit and the extended forward and backward methods of Pytorch Function needed for the learning process. Training_Testing.py contains as in 'Classical' the training loop and the validation function definition. Result_N_qubit.txt contains the results of the training loop (with the update of the loss function), the accuracy test and the time used to complete the training process. 

N_trainable_parameters: 
Binary recognition using a 3D rotation, U gate: 

        ┌───┐ ░ ┌──────────────────────────────────────────────┐ ░  ░ ┌─┐
   q_0: ┤ H ├─░─┤ U(parameters[0],parameters[1],parameters[2]) ├─░──░─┤M├
        └───┘ ░ └──────────────────────────────────────────────┘ ░  ░ └╥┘
meas: 1/═══════════════════════════════════════════════════════════════╩═

