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

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('NaviSign: AI-Powered Real-Time Traffic Sign Recognition')
top.configure(background='#1A2530')  # Updated background to navy blue

# Heading
heading = Label(top, text="NaviSign: AI-Powered Real-Time Traffic Sign Recognition", pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#1A2530', foreground='#E67E22')  # Kept orange heading
heading.pack()

# Image display frame with gold border
image_frame = Frame(top, bg='#1A2530', highlightbackground='#E9C46A', highlightthickness=2)
image_frame.pack(pady=20)
sign_image = Label(image_frame, bg='#1A2530')
sign_image.pack()

# Prediction label
label = Label(top, text="No prediction yet", background='#1A2530', foreground='#F1FAEE', font=('arial', 15, 'bold'))
label.pack()

# Confidence label
confidence_label = Label(top, text="0%", background='#1A2530', foreground='#83C5BE', font=('arial', 12, 'bold'))
confidence_label.pack(pady=5)

# Button frame
button_frame = Frame(top, bg='#1A2530')
button_frame.pack(pady=20)

# Buttons with teal background
upload = Button(button_frame, text="Upload Image", command=lambda: upload_image(), padx=20, pady=10)
upload.configure(background='#2A9D8F', foreground='white', font=('arial', 12, 'bold'), borderwidth=0, activebackground='#264653')
upload.pack(side=LEFT, padx=10)

classify_b = Button(button_frame, text="Classify Image", command=lambda: classify(None), padx=20, pady=10)
classify_b.configure(background='#2A9D8F', foreground='white', font=('arial', 12, 'bold'), borderwidth=0, activebackground='#264653')
classify_b.pack(side=LEFT, padx=10)
classify_b.pack_forget()  # Hidden initially

video_button = Button(button_frame, text="Start Video", command=lambda: toggle_video(), padx=20, pady=10)
video_button.configure(background='#2A9D8F', foreground='white', font=('arial', 12, 'bold'), borderwidth=0, activebackground='#264653')
video_button.pack(side=LEFT, padx=10)

reset_button = Button(button_frame, text="Reset", command=lambda: reset(), padx=20, pady=10)
reset_button.configure(background='#606C88', foreground='white', font=('arial', 12, 'bold'), borderwidth=0)
reset_button.pack(side=LEFT, padx=10)

# Status label
status_label = Label(top, text="✔ Model loaded", background='#1A2530', foreground='#6BBF47', font=('arial', 10))
status_label.pack(side=BOTTOM, pady=10)

video_running = False
cap = None
threshold = 0.75
current_file_path = None

def reset():
    global current_file_path
    sign_image.configure(image='')
    sign_image.image = None
    label.configure(text="No prediction yet")
    confidence_label.configure(text="0%")
    classify_b.pack_forget()
    current_file_path = None

def classify(file_path):
    if not file_path and not current_file_path:
        return
    image = cv2.imread(file_path or current_file_path)
    sign, confidence = classify_frame(image)
    label_text = sign if confidence >= threshold else "Low confidence"
    label.configure(text=label_text)
    confidence_label.configure(text=f"{confidence*100:.0f}%")
    if audio_enabled and confidence >= threshold:
        try:
            engine.say(sign)
            engine.runAndWait()
        except Exception as e:
            print(f"Audio feedback failed: {str(e)}")

def show_classify_button(file_path):
    global current_file_path
    current_file_path = file_path
    classify_b.pack(side=LEFT, padx=10)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded = uploaded.resize((400, 300), Image.Resampling.LANCZOS)
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text="No prediction yet")
        confidence_label.configure(text="0%")
        show_classify_button(file_path)
    except:
        pass

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
        label_text = sign if confidence >= threshold else "Low confidence"
        confidence_label.configure(text=f"{confidence*100:.0f}%")
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        sign_image.configure(image=imgtk)
        sign_image.image = imgtk
        label.configure(text=label_text)
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
            classify_b.pack_forget()
            update_video_feed()
        else:
            label.configure(text="Error: No camera found")
            cap.release()
    else:
        video_running = False
        video_button.configure(text="Start Video")
        if cap:
            cap.release()
            cap = None
        sign_image.configure(image='')
        sign_image.image = None
        label.configure(text="No prediction yet")
        confidence_label.configure(text="0%")

def on_closing():
    global video_running, cap
    if video_running and cap:
        cap.release()
    top.destroy()

top.protocol("WM_DELETE_WINDOW", on_closing)
top.mainloop()