# Traffic Sign Recognition System Using Convolutional Neural Networks 
   Real-Time Traffic Sign Recognition

This project focuses on recognizing traffic signs in real-time using a convolutional neural network (CNN) model. The system is designed to classify traffic signs from images or video streams, providing an essential tool for autonomous driving and driver assistance systems.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Real-time traffic sign recognition from images and video webcam streams.
- High accuracy in classifying various traffic signs.
- User-friendly GUI for easy interaction.
- Audio feedback for classified traffic signs.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-time-traffic-sign-recognition.git
   ```

2. Navigate to the project directory:
   ```bash
   cd real-time-traffic-sign-recognition
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main application:
   ```bash
   python main.py  # For Training the model
   python test.py  # For Running and classifying the images or videos
   ```

2. Use the GUI to upload images or start a video stream for real-time traffic sign recognition.

## Dataset

The project utilizes the German Traffic Sign Recognition Benchmark (GTSRB) dataset, a widely used benchmark for traffic sign recognition tasks. The dataset is publicly available at https://benchmark.ini.rub.de/gtsrb_dataset.html



## Model Training

The model is trained using a convolutional neural network (CNN) with the following parameters:

- **Batch Size:** 50
- **Steps per Epoch:** 2000
- **Epochs:** 30
- **Image Dimensions:** (32, 32, 3)

### Training Process

1. **Data Preprocessing:**
   - Convert images to grayscale.
   - Equalize histogram to standardize lighting.
   - Normalize pixel values between 0 and 1.

2. **Data Augmentation:**
   - Random shifts, zooms, shears, and rotations to increase dataset variability.

3. **Model Architecture:**
   - Multiple convolutional layers with ReLU activation.
   - Max pooling layers for downsampling.
   - Dropout layers to prevent overfitting.
   - Dense layers for classification.

### Training Results

#### Loss

![Training Loss](Figure_1_loss.png)

#### Accuracy

![Training Accuracy](Figure_2_accuracy.png)

## Results

The model achieves high accuracy in classifying traffic signs. Here are some sample results:

![Result 1](result.jpg)

![Result 2](result2.jpg)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
