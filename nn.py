import csv
import numpy as np
import matplotlib
import copy
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import cv2

matplotlib.use('TkAgg')

def save_data(weights, biases, file_name):
    array_weights = np.array(weights, dtype=object)
    np.save(file_name + '_weights', array_weights)
    array_biases = np.array(biases, dtype=object)
    np.save(file_name + '_biases', array_biases)


def load_data(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        label = []
        images = []
        for row in reader:
            label.append(int(row[0]))
            images.append(list(map(int, row[1:])))
    return label, np.array(images, dtype=np.uint8)


def create_weights_and_biases(len_input, len_output, num_hidden_layers, len_hidden_layers, epsilon=1):
    weights = []
    biases = []

    weights.append(np.random.uniform(-epsilon, epsilon, (len_input, len_hidden_layers[0])))
    biases.append(np.zeros(len_hidden_layers[0]))

    for i in range(1, num_hidden_layers):
        weights.append(np.random.uniform(-epsilon, epsilon, (len_hidden_layers[i - 1], len_hidden_layers[i])))
        biases.append(np.zeros(len_hidden_layers[i]))

    weights.append(np.random.uniform(-epsilon, epsilon, (len_hidden_layers[-1], len_output)))
    biases.append(np.zeros(len_output))

    return weights, biases

def Relu(x):
    return np.maximum(0, x)


def dev_ReLU(x):
    return x > 0


def Softmax(x):
    x = x.astype(float)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict(input, weights, biases):
    a = []
    z = []

    input = input.reshape(1,784) / 255.0

    a.append(input)
    z.append(np.dot(input, weights[0])+biases[0])

    for i in range(1,len(weights)):
        a.append(Relu(z[i-1]))
        z.append(np.dot(a[i], weights[i])+biases[i])

    a.append(Softmax(z[-1]))
    return z, a

def backpropagation(weights, a, z, label):

    correct_output = np.zeros(10)
    correct_output[label] = 1

    dweights = []
    dbiases = []

    delta = (a[-1] - correct_output)
    dweights.append(np.dot(a[-2].T, delta))
    dbiases.append(np.sum(delta))

    for i in range(len(weights)-2,-1,-1):
        delta = np.dot(delta, weights[i+1].T)*dev_ReLU(z[i])
        dweights.append(np.dot(a[i].T, delta))
        dbiases.append(np.sum(delta))

    return dweights, dbiases

def update_weights_and_biases(weights, biases, dweights, dbiases, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * dweights[len(weights) - i - 1]
    for i in range(len(biases)):
        biases[i] -= learning_rate * dbiases[len(biases) - i - 1]

    return weights, biases


def train(iterations=200):
    train_labels, train_images = load_data('mnist/mnist_train.csv')
    test_labels, test_images = load_data('mnist/mnist_test.csv')

    len_input = 784
    len_output = 10
    num_hidden_layers = 2
    len_hidden_layers = [100, 100]
    learning_rate = 1e-3

    weights, biases = create_weights_and_biases(len_input, len_output, num_hidden_layers, len_hidden_layers)
    accuracy = 0
    for i in range(iterations):
        for j in range(len(train_images)):
            z, a = predict(train_images[j], weights, biases)
            dweights, dbiases = backpropagation(weights, a, z, train_labels[j])
            weights, biases = update_weights_and_biases(weights, biases, dweights, dbiases, learning_rate)

        correct = 0
        for j in range(len(test_images)):
            z, a = predict(test_images[j], weights, biases)
            if np.argmax(a[-1]) == test_labels[j]:
                correct += 1
        accuracy = correct/len(test_images)
        print(f'Iteration {i+1}: {accuracy}')

    save_data(weights, biases, 'best_weights/nn_' + str(accuracy))


def gaussian_2d(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    gaussian = np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
    return gaussian

loaded_weights = np.load("best_weights/nn_0.9523_weights.npy", allow_pickle=True)
loaded_biases = np.load("best_weights/nn_0.9523_biases.npy", allow_pickle=True)


best_weights = [np.array(w) for w in loaded_weights]
best_biases = [np.array(b) for b in loaded_biases]


class DrawApp:
    def __init__(self, root, predict_fn):
        self.root = root
        self.root.title("Draw a Number")

        self.canvas_size = 280
        self.image_size = 28

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)  # Black background
        self.Tk_image = ImageTk.PhotoImage(self.image)
        self.draw = ImageDraw.Draw(self.image)

        self.predict_fn = predict_fn

        self.blob_size = (100,100)
        self.gaussian = gaussian_2d(self.blob_size, 5)
        self.gaussian = (self.gaussian * 128).astype(np.uint8)
        self.blob = Image.new("RGBA", self.blob_size, (255, 255, 255, 0))
        self.blob.putalpha(Image.fromarray(self.gaussian))


    def paint(self, event):

        x, y = event.x - self.blob_size[0]//2, event.y - self.blob_size[1]//2
        self.image.paste(self.blob, (x , y), mask=self.blob)
        self.Tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.Tk_image, anchor=tk.NW)
        # self.canvas.create_image(0, 0, image=ImageTk.PhotoImage(self.image), anchor=tk.NW)
        # self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white")
        #self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img = self.image.resize((self.image_size, self.image_size))

        input = np.array((np.array(img)))
        input = input.reshape(784,)

        z,a = self.predict_fn(input,best_weights,best_biases)
        predicted_digit = np.argmax(a[-1])

        self.root.title(f"Predicted Digit: {predicted_digit}")


#train()

# root = tk.Tk()
# app = DrawApp(root, predict)
# root.mainloop()
#
test_labels, test_images = load_data('mnist/mnist_test.csv')

for i in range(len(test_images)):
    z, a = predict(test_images[i], best_weights, best_biases)
    predicted_digit = np.argmax(a[-1])
    if predicted_digit != test_labels[i]:
        print(f"Predicted: {predicted_digit}, Actual: {test_labels[i]}")
        # img = test_images[i].reshape(28, 28)
        # img = cv2.resize(img, (280, 280))
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

correct = 0
for i in range(len(test_images)):
    z, a = predict(test_images[i], best_weights, best_biases)
    predicted_digit = np.argmax(a[-1])
    if predicted_digit == test_labels[i]:
        correct += 1

print(f"Accuracy on test data: {correct/len(test_images)}")

