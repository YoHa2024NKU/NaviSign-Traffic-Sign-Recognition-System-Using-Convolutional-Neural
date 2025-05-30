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
top.geometry('1200x600')
top.title('Traffic Sign Classification')
top.configure(background='#ECF0F1')

# Add heading
heading = Label(top, text="Real Time Traffic Sign Recognition", pady=10, font=('Arial', 28, 'bold'), bg='#ECF0F1', fg='#2C3E50')
heading.place(x=400, y=10)

# Video and Image Frames
main_frame = Frame(top, bg='#ECF0F1')
main_frame.place(x=50, y=100)

# Video Section (Left)
video_frame = Frame(main_frame, bg='#ECF0F1', width=500, height=400)
video_frame.grid(row=0, column=0, padx=20, pady=10)  # Added padding for separation
label_video = Label(video_frame, text="Low confidence", bg='#ECF0F1', font=('Arial', 16, 'bold'), fg='#2C3E50', wraplength=250)
label_video.pack(pady=10)
sign_video = Label(video_frame, width=500, height=300)
sign_video.pack()

# Image Section (Right)
image_frame = Frame(main_frame, bg='#ECF0F1', width=500, height=400)
image_frame.grid(row=0, column=1, padx=20, pady=10)  # Added padding for separation
label_image = Label(image_frame, text="Speed Limit 50 km/h\n(1.00)", bg='#ECF0F1', font=('Arial', 16, 'bold'), fg='#2C3E50', wraplength=250)
label_image.pack(pady=10)
sign_image = Label(image_frame, width=500, height=300)
sign_image.pack()

# Classify button (created once)
current_file_path = None
def classify():
    global current_file_path
    if current_file_path:
        image = cv2.imread(current_file_path)
        sign, confidence = classify_frame(image)
        label_text = f"{sign} ({confidence:.2f})" if confidence >= threshold else "Low confidence"
        label_image.configure(text=label_text)
        if audio_enabled and confidence >= threshold:
            try:
                engine.say(sign)
                engine.runAndWait()
            except Exception as e:
                print(f"Audio feedback failed: {str(e)}")

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((500, 400), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        sign_video.configure(image=imgtk)
        sign_video.image = imgtk
        label_video.configure(text=label_text)

        # Add voice feedback when a sign is recognized with high confidence
        if confidence >= threshold and audio_enabled:
            try:
                engine.say(sign)
                engine.runAndWait()
            except Exception as e:
                print(f"Audio feedback failed: {str(e)}")

    top.after(33, update_video_feed)

def toggle_video():
    global video_running, cap
    if not video_running:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video_running = True
            start_button.configure(state='disabled')
            stop_button.configure(state='normal')
            update_video_feed()
        else:
            label_video.configure(text="Error: No camera found")
            cap.release()
    else:
        video_running = False
        start_button.configure(state='normal')
        stop_button.configure(state='disabled')
        if cap:
            cap.release()
        sign_video.configure(image='')
        label_video.configure(text='')

def upload_image():
    global current_file_path
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded = uploaded.resize((500, 400), Image.Resampling.LANCZOS)
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label_image.configure(text='')
        current_file_path = file_path
        classify_button.configure(state='normal')
    except:
        pass

# Buttons Section (Bottom)
button_frame = Frame(top, bg='#ECF0F1')
button_frame.place(x=50, y=550)  # Move the button frame to the left

start_button = Button(button_frame, text="Start Video", command=toggle_video, padx=15, pady=5, bg='#4682B4', fg='white', font=('Arial', 14, 'bold'))
stop_button = Button(button_frame, text="Stop Video", command=toggle_video, padx=15, pady=5, bg='#DC143C', fg='white', font=('Arial', 14, 'bold'))
upload_button = Button(button_frame, text="Upload an image", command=upload_image, padx=15, pady=5, bg='#32CD32', fg='white', font=('Arial', 14, 'bold'))
classify_button = Button(button_frame, text="Classify Image", command=classify, padx=15, pady=5, bg='#1E90FF', fg='white', font=('Arial', 14, 'bold'))
close_button = Button(button_frame, text="Close Image", command=top.destroy, padx=15, pady=5, bg='#FF6347', fg='white', font=('Arial', 14, 'bold'))  # Renamed button

# Align buttons to the left with spacing
start_button.pack(side=LEFT, padx=10, pady=5)
stop_button.pack(side=LEFT, padx=10, pady=5)
upload_button.pack(side=LEFT, padx=40, pady=5)  # Increased horizontal padding for separation
classify_button.pack(side=LEFT, padx=10, pady=5)
close_button.pack(side=LEFT, padx=10, pady=5)

# Finalize GUI
def on_closing():
    global video_running, cap
    if video_running and cap:
        cap.release()
    top.destroy()

top.protocol("WM_DELETE_WINDOW", on_closing)
top.mainloop()