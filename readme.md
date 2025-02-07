# Neural Network from Scratch - Titanic Dataset

After completing the "Improve Deep Neural Networks" course from DeepLearning.AI, this project serves as a practical implementation of key deep learning concepts. The goal is to build a neural network from scratch using NumPy and apply it to the Titanic dataset for binary classification (survival prediction).

## Key Concepts Applied

### 1. Data Preprocessing
- Loaded the Titanic dataset and handled missing values.
- Encoded categorical variables to numerical representations.
- Standardized features to improve model convergence.

### 2. Neural Network Implementation
- **Weight Initialization:** Used techniques like Xavier/He initialization.
- **Activation Functions:** Implemented ReLU and sigmoid activation functions.
- **Forward Propagation:** Computed activations through multiple layers.
- **Backward Propagation:** Derived gradients using the chain rule.
- **Optimization:** Used Mini-Batch Gradient Descent with Adam optimizer.
- **Regularization:** Applied L2 regularization and dropout to prevent overfitting.

### 3. Training and Evaluation
- Split the dataset into training and test sets.
- Trained the neural network using backpropagation and optimization techniques.
- Evaluated model performance using accuracy and loss metrics.

## Technologies Used
- Python
- NumPy
- Pandas (for data preprocessing)
- scikit-learn