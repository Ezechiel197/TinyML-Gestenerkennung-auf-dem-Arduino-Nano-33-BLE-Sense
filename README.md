# TinyML Gesture Recognition on the Arduino Nano 33 BLE Sense

A real-time motion classification system (**Punch** vs. **Lift**) built with TensorFlow / Keras and TensorFlow Lite for Microcontrollers (TFLM) on an **Arduino Nano 33 BLE Sense**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Hardware & Prerequisites](#hardware--prerequisites)
- [How It Works & Signal Processing](#how-it-works--signal-processing)
- [Model Architecture & Quantization](#model-architecture--quantization)
- [Project Structure](#project-structure)
- [Author](#author)
- [Picture](#picture)
- [Installation & Setup](#installation--setup)
- [LED Status Indicators & Output](#led-status-indicators--output)
- [Configuration Parameters](#configuration-parameters)

---

## Project Overview

This project demonstrates a complete Edge AI workflow—from data collection and neural network training in Python to resource-constrained inference directly on a microcontroller without cloud connectivity.

The system uses the built-in 6-axis IMU (accelerometer and gyroscope) of the Arduino Nano 33 BLE Sense to distinguish between two arm movements:
- **Punch** (forward thrust motion)
- **Lift** (upward lifting motion)

---

## Hardware & Prerequisites

### Hardware
- **Microcontroller:** Arduino Nano 33 BLE Sense (nRF52840, ARM Cortex-M4F @ 64 MHz, 256 KB SRAM, 1 MB Flash)
- **Sensors:** On-board 9-axis IMU **LSM9DS1** (using 3-axis accelerometer & 3-axis gyroscope)
- **Connection:** Micro-USB data cable

### Software & Libraries
- **Python (Training & Conversion):**
  - `tensorflow` / `keras`
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`
- **Arduino IDE / PlatformIO (Deployment):**
  - Board Support Package: *Arduino Mbed OS Nano Boards*
  - `Arduino_LSM9DS1` (reading IMU sensor data)
  - `Chirale_TensorFlowLite` (TFLite Micro runtime for Arduino)

---

## How It Works & Signal Processing

To ensure reliable classification on the microcontroller, the firmware applies the exact same preprocessing pipeline used during training:

1. **Motion Trigger (Threshold Detection):**
   At rest, the sum of absolute acceleration magnitudes ($\vert{}a_x\vert{} + \vert{}a_y\vert{} + \vert{}a_z\vert{}$) is around $1.0\,g$ due to gravity. Once this sum exceeds `kTriggerAccelSum = 1.3f`, a gesture is triggered and window capture begins.
2. **Windowing:**
   Starting at the trigger point, **119 consecutive samples** are captured across all **6 channels** ($a_x, a_y, a_z, g_x, g_y, g_z$), forming an input shape of `119 x 6 = 714` values.
3. **Z-Score Normalization:**
   Each reading is standardized per channel using the training mean (`kMean`) and standard deviation (`kStd`):
   $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
4. **Inference & Buffer Flush:**
   After classification, the result is displayed for `1500 ms`. Afterward, the internal IMU FIFO buffer is drained so data accumulated during the display delay does not cause false triggers.

---

## Model Architecture & Quantization

- **Dataset:** 201 recorded gestures (2,010 windows total of `119 x 6` samples from `data/punch.csv` and `data/lift.csv`).
- **Network Structure:** Neural network (Keras `Sequential` model with `Flatten`, `Dense`, and `ReLU` layers, plus a binary output for $P(\text{lift})$). *(Note: Add exact layer counts/unit sizes from your notebook here if needed.)*
- **Quantization & Deployment:**
  - Converted to the **TensorFlow Lite FlatBuffer format** and quantized to **8-bit integer (`int8`)**.
  - Model size in Flash (`model.h`): **17,616 bytes (~17.2 KB)**.
  - Allocated Tensor Arena in RAM (`kTensorArenaSize`): **16 KB** (`16 * 1024` bytes).

---

## Project Structure

```text
├── data/
│   ├── punch.csv               # Raw data for "Punch" gestures (6 axes)
│   └── lift.csv                # Raw data for "Lift" gestures (6 axes)
├── training/
│   └── gesture_training.ipynb  # Jupyter Notebook: preprocessing, PCA, training & TFLite export
├── arduino/
│   ├── gesture_classifier.ino  # Arduino sketch for real-time inference & LED feedback
│   └── model.h                 # Quantized TFLite model as a C byte array
└── README.md
```

---

## Author
Ezechiel Tonkeme

---

## Picture
<img width="505" height="593" alt="image" src="https://github.com/user-attachments/assets/a2f7eff4-d802-4a0b-a656-5191c3ef4d16" />
