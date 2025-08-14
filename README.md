# MNIST Digit Classification from Scratch with Interactive Drawing Demo

This project implements a **fully connected neural network from scratch** (no TensorFlow or PyTorch) to classify handwritten digits from the **MNIST dataset**.  
Additionally, it features an interactive **Tkinter-based GUI** that allows you to draw digits and see real-time predictions from the trained network.

## Features

- Custom-built neural network implementation using only **NumPy**.
- Training and testing on the MNIST dataset (CSV format).
- Manual implementation of:
  - Forward propagation
  - Backpropagation
  - Weight and bias updates
  - ReLU and Softmax activations
- Saving and loading trained weights and biases.
- **Drawing demo** where you can sketch a digit and the model predicts it instantly.

## How It Works

1. **Data Loading**  
   - Reads MNIST CSV files containing pixel values and labels.
   - Normalizes input values to the range [0, 1].

2. **Neural Network Architecture**  
   - Input layer: 784 neurons (28×28 pixels).
   - Two hidden layers: 100 neurons each, ReLU activation.
   - Output layer: 10 neurons, Softmax activation.

3. **Training Process**  
   - Loops through training samples, performing forward and backward passes.
   - Updates weights and biases using gradient descent.
   - Evaluates accuracy on the test set after each epoch.

4. **Interactive Demo**  
   - Built with Tkinter.
   - Allows drawing with a smooth Gaussian brush.
   - Converts your sketch into a 28×28 grayscale image.
   - Predicts and displays the recognized digit in the window title.

## Installation

```bash
pip install numpy pillow opencv-python matplotlib
```

## Usage

### Train the model
1. Download the MNIST dataset in CSV format.
2. Place `mnist_train.csv` and `mnist_test.csv` inside an `mnist/` folder.
3. Run:
```bash
python main.py
```
The trained weights will be saved inside the `best_weights/` directory.

### Run the drawing demo
After training (or using pre-trained weights):
```bash
python main.py
```
Uncomment the following lines at the bottom of the script:
```python
root = tk.Tk()
app = DrawApp(root, predict)
root.mainloop()
```

## Example Results

You can include screenshots or GIFs here, for example:
```
results/demo_prediction.gif
```
![Example](results/demo_prediction.gif)

