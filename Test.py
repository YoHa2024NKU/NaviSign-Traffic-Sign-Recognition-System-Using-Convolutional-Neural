import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
import pickle
import os

# Initialize text-to-speech engine (if available)
try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    audio_enabled = True
except ImportError:
    print("pyttsx3 not installed. Audio feedback disabled.")
    audio_enabled = False

# Load the trained model
with open("model_trained.p", "rb") as pickle_in:
    model = pickle.load(pickle_in)

# Class labels from test.py
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

# Preprocessing function from main.py
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)
video_running = False
cap = None
threshold = 0.75

def classify_frame(frame):
    img = cv2.resize(frame, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img, verbose=0)
    classIndex = np.argmax(predictions, axis=1)[0]
    probabilityValue = np.amax(predictions)
    if probabilityValue >= threshold:
        return getClassName(classIndex), probabilityValue
    return "Low confidence", probabilityValue

def update_video_feed():
    global video_running, cap
    if not video_running:
        return
    ret, frame = cap.read()
    if ret:
        sign, confidence = classify_frame(frame)
        label_text = f"{sign} ({confidence:.2f})" if confidence >= threshold else "Low confidence"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        sign_image.configure(image=imgtk)
        sign_image.image = imgtk
        label.configure(foreground='#011638', text=label_text)
    top.after(33, update_video_feed)

def toggle_video():
    global video_running, cap
    if not video_running:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video_running = True
            video_button.configure(text="Stop Video")
            update_video_feed()
        else:
            label.configure(text="Error: No camera found")
            cap.release()
    else:
        video_running = False
        video_button.configure(text="Start Video")
        if cap:
            cap.release()
        sign_image.configure(image='')
        label.configure(text='')

def classify(file_path):
    image = cv2.imread(file_path)
    sign, confidence = classify_frame(image)
    label_text = f"{sign} ({confidence:.2f})" if confidence >= threshold else "Low confidence"
    label.configure(foreground='#011638', text=label_text)
    if audio_enabled and confidence >= threshold:
        try:
            engine.say(sign)
            engine.runAndWait()
        except Exception as e:
            print(f"Audio feedback failed: {str(e)}")

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

# Add buttons
upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=10)

video_button = Button(top, text="Start Video", command=toggle_video, padx=10, pady=5)
video_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
video_button.pack(side=BOTTOM, pady=10)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

def on_closing():
    global video_running, cap
    if video_running and cap:
        cap.release()
    top.destroy()

top.protocol("WM_DELETE_WINDOW", on_closing)
top.mainloop()