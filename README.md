# TraNQI
# Training a quantum-classical neural network with qiskit and pytorch. 

 In this project a particular implementation of hybrid neural networks is studied. The aim is to perform image recognition, MNIST digit, in a Supervised Machine Learning context. 
 
 A hidden layer of a classical neural network is implemented using a parametrized quantum circuit. The outputs from the previous layer are collected and used as inputs for the circuit. The measurement statistics of the quantum circuit can then be collected and used as inputs for the following layer. In the end, the outputs of the neural network are numerically compared with expected results through a loss function, defined as in the classical case. The parameters in the quantum circuit and the ones in the classical layers are optimized through the backpropagation algorithm, where the optimization process is implemeted in order to be compatible with the quantum circuit via the parameter shift rule. The aim is to minimize the loss function and to find a model being able to predict correctly the features of our input data.
 
 ![NN_quantum](https://user-images.githubusercontent.com/83702763/126067398-f88b0be6-1315-4173-9389-d19b07997e05.png)

 
 Several experiments have been carried out starting from the previous model, changing the circuit features and the number of its qubits. Moreover, gaussian errors with different standard deviation values have been added to validation set of data, in order to test our model for the better, and also to training data, to make it stonger. Specifically, three-dimension rotation, U3 gate, has been used for a single qubit circuit, which led to 99.9\% accuracy both on simulator and on actual quantum devices. Furthermore interesting results have been obtained for noisy validation data in this experiment with an accuracy higher than 99\% up to 0.7 deviation standard value. Then, more-qubits circuit was used, where parametrized rotations act on each qubit, which gave an accuracy of 99\% for not-noisy validation data and interesting results for noisy validation data when learning was made with noisy data too. In the end, other two circuits were tested: entangled qubits circuit and a circuit inspired by a hybrid optimization algorithm, QAOA, which still makes use of entanglement. In these cases, the resultant accuracy value was of 99\% and of 98\% respectively on the quantum devices.  

 
 ![Noisy_data](https://user-images.githubusercontent.com/83702763/126067488-3a443f03-bf1d-4352-a743-e90c48856aaf.png)
 
 Programs have been implemented in Python coding language through Qiskit and Pytorch, Python libraries used for Quantum Computing and Machine Learning respectively. Then, the programs have been tested both on simulators and on actual quantum computers on the online platform IBM Quantum Experience, that gives the opportunity to run programs on the IBM quantum computers. 
