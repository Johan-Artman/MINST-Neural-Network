import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn

# Create the same CNN model as in gpu_test.py
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(5 * 28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = x.view(-1, 5 * 28 * 28)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model and load weights
model = SimpleCNN().to(device)
try:
    model.load_state_dict(torch.load('gpu_trained_model.pth'))
    model.eval()  # Set to evaluation mode
    print("Successfully loaded trained model")
except FileNotFoundError:
    print("No trained model found. Please run gpu_test.py first to train the model.")
    exit(1)

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
    # Convert to PyTorch tensor and reshape
    img_tensor = torch.FloatTensor(img_arr).reshape(1, 1, 28, 28).to(device)
    
    # Make prediction with the model
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = output.cpu().numpy()[0]
        digit = np.argmax(probabilities)
        confidence = probabilities[digit] * 100
        print(f"Predicted digit: {digit} (confidence: {confidence:.2f}%)")

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