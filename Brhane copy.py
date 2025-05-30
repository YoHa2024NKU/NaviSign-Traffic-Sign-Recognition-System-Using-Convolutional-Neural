import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
import pickle

# Load the trained model
try:
    with open("model_trained.p", "rb") as pickle_in:
        model = pickle.load(pickle_in)
except FileNotFoundError:
    print("Error: Trained model 'model_trained.p' not found. Please run Main.py to train and save the model.")
    exit()

# Class labels
def getClassName(classNo):
    classes = {
        0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h', 8: 'Speed Limit 120 km/h',
        9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
        14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
        23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
        26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
        29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
        32: 'End of all speed and passing limits', 33: 'Turn right ahead',
        34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
        37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
        40: 'Roundabout mandatory', 41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return classes.get(classNo, "Unknown")

# Preprocessing function
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# Classify an image
def classify_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32))
    image = preprocessing(image)
    image = image.reshape(1, 32, 32, 1)
    predictions = model.predict(image)
    classIndex = np.argmax(predictions, axis=1)[0]
    probabilityValue = np.amax(predictions)
    return classIndex, probabilityValue

# GUI Functions
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        classIndex, probabilityValue = classify_image(file_path)
        label.configure(text=f"Class: {getClassName(classIndex)}\nProbability: {probabilityValue:.2f}")
    except Exception as e:
        label.configure(text="Error: Unable to classify image.")
        print(f"Error: {e}")

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classification')
top.configure(background='#CDCDCD')

heading = Label(top, text="Traffic Sign Classification", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)

sign_image = Label(top)
sign_image.pack(side=BOTTOM, expand=True)

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label.pack(side=BOTTOM, expand=True)

top.mainloop()