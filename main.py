from pyexpat import model
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model   # type: ignore

#load model
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
        
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
        
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights for input-to-hidden and hidden-to-output layers
        self.weights_hidden_input = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        # Compute hidden layer activation
        self.hidden_input = np.dot(x, self.weights_hidden_input)
        self.hidden_output = sigmoid(self.hidden_input)
        # Compute output layer activation
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = sigmoid(self.final_input)
        return self.final_output
    
    #load model weights from trainig
def load_model_weights(nn, filename="custom_nn_weights.npz"):
        data = np.load(filename)
        nn.weights_hidden_input = data["weights_hidden_input"]
        nn.weights_hidden_output = data["weights_hidden_output"]
        print(f"Model weights loaded from {filename}")

input_size = 28 * 28
hidden_size = 64      # Note to self should match what was used during training.
output_size = 10

nn = NeuralNetwork(input_size, hidden_size, output_size)
load_model_weights(nn, "custom_nn_weights.npz")


# Set up the drawing board dimensions
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

# Create the main window
root = tk.Tk()
root.title("Draw a Digit")

# Create a canvas widget
canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
canvas.pack()

image1 = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
draw = ImageDraw.Draw(image1)

# Function to handle drawing on the canvas
def paint(event):
    # Define a brush size
    brush_size = 4
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)

    canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    draw.ellipse([x1, y1, x2, y2], fill='black')

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_WIDTH, CANVAS_HEIGHT], fill='white')

# Function to process the drawing and predict the digit
def predict_digit():
    # 28x28 because that is the input for the mnist
    img = image1.resize((28, 28))
    # Invert image colors: MNIST digits are white on black background
    img = ImageOps.invert(img)
    # Convert image to a numpy array and normalize pixel values
    img_arr = np.array(img) / 255.0
    # Reshape to match the model's expected input shape (batch, height, width, channels)
    img_arr = img_arr.reshape(1, 28, 28, 1)
    
    # Make prediction with the model
    prediction = model.predict(img_arr)
    digit = np.argmax(prediction)
    print("Predicted digit:", digit)

# Bind mouse drag event to the paint function
canvas.bind("<B1-Motion>", paint)

# Create buttons to clear the canvas and trigger prediction
button_frame = tk.Frame(root)
button_frame.pack()

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
clear_button.pack(side="left", padx=10, pady=10)

predict_button = tk.Button(button_frame, text="Predict", command=predict_digit)
predict_button.pack(side="left", padx=10, pady=10)

# Run the Tkinter main loop
root.mainloop()