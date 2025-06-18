<H3>S.SAJETHA</H3>
<H3>212223100049</H3>
<H3>EX. NO.5</H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>

```py
import numpy as np
import matplotlib.pyplot as plt

def rbf(x, c, g=1): return np.exp(-g * np.linalg.norm(x - c) ** 2)

def transform_and_train(X, y, c1, c2):
    phi = np.array([[rbf(x, c1), rbf(x, c2), 1] for x in X])
    w = np.linalg.pinv(phi).dot(y)

    # Plot
    plt.figure(figsize=(13,5))
    plt.subplot(1,2,1)
    plt.scatter(*X[y==0].T, label="Class 0")
    plt.scatter(*X[y==1].T, label="Class 1")
    plt.title("XOR: Linearly Inseparable")
    plt.xlabel("X1"); plt.ylabel("X2"); plt.legend()

    plt.subplot(1,2,2)
    hidden = phi[:, :2]
    plt.scatter(*hidden[y==0].T, label="Class 0")
    plt.scatter(*hidden[y==1].T, label="Class 1")
    plt.plot([0, 1], [1, 0], "k--")
    plt.title("RBF Transformed: Linearly Separable")
    plt.xlabel("Φ1"); plt.ylabel("Φ2"); plt.legend()
    plt.show()

    return w

def predict(x, w, c1, c2):
    return round(w.dot([rbf(x, c1), rbf(x, c2), 1]))

# Data & centers
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
c1, c2 = np.array([0,1]), np.array([1,0])

# Train
weights = transform_and_train(X, y, c1, c2)

# Test
for point in X:
    print(f"Input: {point}, Predicted: {predict(point, weights, c1, c2)}")

```

<H3>OUTPUT:</H3>

![image](https://github.com/user-attachments/assets/1fd1a358-b00c-45fb-b0ca-66685531ff99)


<H3>Result:</H3>
Thus, a Radial Basis Function Neural Network is implemented to classify XOR data.
