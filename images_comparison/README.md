**Noisy_data** contains examples of the images (MNIST digit) used for recognition. Some of them have been changed with gaussian noise at different deviation standrad values.

**Accuracy** : comparison between accuracy percentages for different standard deviation values in the validation set of data when the training is made with not noisy training data.

**Accuracy_noise**:  comparison when the training is made with noisy training data (std deviation: Entanglement=0.1, N_qubits=0.2, U3=0.3)

**Compare_sim**: loss function convergence on qasm simulator for different experiments.

**Compare_QC**: loss function convergence on real quantum computers for different experiments.

**U3_sim_vs_QC**: loss function convergence for U3 experiment for training made on qasm_siulator and on ibmq_athens. The decrease is quite the same.
