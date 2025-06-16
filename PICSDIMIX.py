import os
from kivy.config import Config

os.environ['KIVY_GL_BACKEND'] = 'sdl2'
os.environ['KIVY_CLOCK'] = 'interrupt'
os.environ['KIVY_VIDEO'] = 'ffpyplayer'
os.environ['KIVY_WINDOW'] = 'sdl2'  # Force SDL2 window backend
os.environ['KIVY_BCM_DISPMANX_LAYER'] = '0'  # For Raspberry Pi compatibility
os.environ['KIVY_X11_NO_MITSHM'] = '1'  # Suppress XWayland warnings
os.environ['SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR'] = '0'  # Fix window focus
Config.set('kivy', 'keyboard_mode', 'system')  # Critical for virtual keyboard
Config.set('kivy', 'keyboard_layout', 'numeric.json')  # Force numeric layout
Config.set('graphics', 'fullscreen', '1')  # Use '1' instead of 'auto'
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'multisamples', '0')
Config.set('kivy', 'kivy_clock', 'interrupt')

import logging
logging.getLogger("colormath").setLevel(logging.ERROR)
import paho.mqtt.client as mqtt
import json
import threading
import platform
import time
import serial
import serial.tools.list_ports
import subprocess
import csv
import sys
import numpy as np
import pandas as pd
import joblib
import math
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import load_model,Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping  
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
from colormath.color_objects import LCHabColor
from math import sqrt
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.animation import Animation
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window



MQTT_BROKER = "localhost"
MQTT_TOPIC_INPUT = "sensor/rgb"
MQTT_TOPIC_OUTPUT = "test/colorant_output"
MQTT_TOPIC_FEEDBACK = "test/mixed_rgb_feedback"

model = None
scaler_rgb = None
scaler_output = None

previous_detection_state = None  
previous_arduino_state = None  
arduino = None  
baud_rate = 38400  
detect_arduino_event = None  
screen_manager = ScreenManager()
accepting_messages = True
global_data = None
predicted_colorant = None
latest_rgb = None
awaiting_feedback = False


dataset_path = r"c:\Users\Ranillo\Desktop\files\Thesis\TrainingAccuracy\ID.csv"
CORRECTED_PATH = r"c:\Users\Ranillo\Desktop\files\Thesis\TrainingAccuracy\IDcorrected.csv"
backup_data_path = r"c:\Users\Ranillo\Desktop\files\Thesis\TrainingAccuracy\IDcorrected_backup.csv"
model_path =  r"c:\Users\Ranillo\Desktop\files\Thesis\TrainingAccuracy\ID.keras"

#dataset_path = r"/home/picsdimix/tftgui/ID.csv"
#CORRECTED_PATH = r"/home/picsdimix/tftgui/IDcorrected.csv"
#backup_data_path = r"/home/picsdimix/tftgui/IDcorrected_backup.csv"
#model_path =  r"/home/picsdimix/tftgui/ID.keras"

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss instead of just training loss
    patience=50,         # Stop if no improvement for 20 epochs
    restore_best_weights=True  # Restore model weights from best epoch
)

def show_virtual_keyboard(mode="full"):
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.Popen(r"C:\Windows\System32\osk.exe")
        elif system == "Linux":
            if mode == "numeric":
                subprocess.Popen(["matchbox-keyboard", "--keypad"])
            else:
                subprocess.Popen(["matchbox-keyboard"])
    except Exception as e:
        print(f"Could not launch virtual keyboard: {e}")

def hide_virtual_keyboard():
    system = platform.system()
    if system == "Windows":
        subprocess.Popen("taskkill /IM osk.exe /F", shell=True)
    elif system == "Linux":
        subprocess.Popen(["pkill", "matchbox-keyboard"])
def show_warning_popup(message):
    popup = Popup(
        title="Input Error",
        content=Label(text=message),
        size_hint=(None, None),
        size=(400, 200),
        auto_dismiss=True
    )
    popup.open()

# Detect the Arduino's serial port
def find_arduino():
    """Detects the Arduino's serial port without spamming messages."""
    global previous_detection_state  
    ports = list(serial.tools.list_ports.comports())

    if not ports:  
        if previous_detection_state is None or previous_detection_state is True:
            print("No Arduino detected.")
        previous_detection_state = False  
        return None

    for port in ports:
        if any(x in port.description for x in ["Arduino", "CH340", "USB-SERIAL", "USB Serial Device", "CP210x"]) or \
           any(x in port.device for x in ["/dev/ttyUSB", "/dev/ttyACM"]):
            
            if previous_detection_state is not True:
                print(f"Found Arduino on {port.device}")
            previous_detection_state = True  
            return port.device  

    return None  

def connect_arduino(gui_instance=None):
    """Attempts to connect to the detected Arduino and update the GUI."""
    global arduino  
    detected_port = find_arduino()

    if detected_port:
        try:
            # Attempt to connect to the Arduino
            arduino = serial.Serial(detected_port, baud_rate, timeout=1)
            print(f"Connected to Arduino on {detected_port}")

            # Update GUI status if applicable
            if gui_instance and hasattr(gui_instance, 'config_screen'): 
                Clock.schedule_once(lambda dt: gui_instance.config_screen.update_arduino_status(f"Arduino Connected: {detected_port}"), 0)

            return True  # Connection successful

        except serial.SerialException as e:
            print(f"ERROR: Port {detected_port} found but cannot be opened. {e}")

            # Update GUI status if applicable
            if gui_instance and hasattr(gui_instance, 'config_screen'):
                Clock.schedule_once(lambda dt: gui_instance.config_screen.update_arduino_status("Port found but cannot be opened."), 0)

            arduino = None  # Ensure arduino is set to None on failure

    else:
        # If no port is detected, set arduino to None
        arduino = None  
        if gui_instance and hasattr(gui_instance, 'config_screen'):
            Clock.schedule_once(lambda dt: gui_instance.config_screen.update_arduino_status("No Arduino detected."), 0)

    return False  # Connection failed 

def listen_to_arduino(gui_instance):
    """Listens for Arduino messages without blocking the GUI."""
    global arduino

    def check_serial(dt):
        """Called periodically by Kivy's Clock to check Arduino."""
        try:
            # Check if Arduino is connected
            if arduino and arduino.is_open:
                # Read a line (non-blocking due to timeout)
                line = arduino.readline().decode().strip()
                
                if line:
                    print(f"Arduino says: {line}")
                    
                    # Handle "DONE" signal
                    if line == "DONE":
                        main_screen = gui_instance.sm.get_screen('main')
                        Clock.schedule_once(
                            lambda dt: main_screen.handle_operation_complete(), 
                            0
                        )
            else:
                # Attempt reconnection if disconnected
                print("Arduino disconnected. Reconnecting...")
                connect_arduino(gui_instance)

        except serial.SerialException as e:
            print(f"Serial error: {e}")
            connect_arduino(gui_instance)  # Reconnect on error
        except Exception as e:
            print(f"Unexpected error: {e}")

    # Start checking every 100ms (adjust interval as needed)
    Clock.schedule_interval(check_serial, 0.1)
        
# GUI Status Update Only (No Reconnection Here)
def check_arduino_connection(gui_instance, dt=None):
    """Checks Arduino connection status and updates the GUI."""
    global arduino, previous_arduino_state

    is_connected = arduino is not None and arduino.is_open

    if is_connected != previous_arduino_state:
        if is_connected:
            print("Arduino connected successfully!")
        else:
            print("Arduino not detected.")

        if gui_instance and hasattr(gui_instance, 'config_screen'):
            Clock.schedule_once(lambda dt: gui_instance.config_screen.update_arduino_status("Connected" if is_connected else "Disconnected"), 0)

    previous_arduino_state = is_connected  

# Thread-Based Reconnection Handling
def monitor_arduino_connection():
    """Continuously checks and reconnects to the Arduino if disconnected."""
    global arduino

    while True:
        is_connected = arduino is not None and arduino.is_open

        if not is_connected:
            print("Arduino disconnected. Retrying...")
            time.sleep(3)  # Wait before retrying
            connect_arduino()  # Attempt reconnection

        time.sleep(1)  # Check every second

# Starts the GUI status check loop
def start_gui_arduino_monitoring(gui_instance):
    """Starts checking Arduino status for GUI updates every 1 second."""
    Clock.schedule_interval(lambda dt: check_arduino_connection(gui_instance), 1.0)

# Starts the background monitoring thread
def start_arduino_monitoring(gui_instance):
    """Starts checking the Arduino connection globally every second."""
    Clock.schedule_interval(lambda dt: check_arduino_connection(gui_instance), 1.0)
    
    # Ensure it's called in all screens by tracking screen changes
    if hasattr(gui_instance, 'screen_manager'):
        gui_instance.screen_manager.bind(current=lambda instance, value: check_arduino_connection(gui_instance))

# Closes the serial connection properly
def close_serial():
    """Closes the serial connection to the Arduino."""
    global arduino
    if arduino and arduino.is_open:
        print("Closing Serial Port...")
        arduino.close()
        arduino = None  

# Sends a command to the Arduino
def send_command(command, *values):
    """Sends a command with the correct colorant proportions to Arduino."""
    if arduino and arduino.is_open:
        if values:
            formatted_values = " ".join(map(str, values))
            command = f"{command} {formatted_values}"
        command += "\n"
        try:
            arduino.write(command.encode())
            print(f"Sent command: {command.strip()}")
        except serial.SerialException as e:
            print(f"ERROR: Failed to send command '{command.strip()}'. {e}")
    else:
        print(f"Arduino not connected. Command '{command}' was not sent.")

def check_wifi_connection():
    """Check if WiFi is connected on both Windows & Raspberry Pi."""
    system = platform.system()

    if system == "Windows":
        try:
            result = subprocess.check_output(['netsh', 'wlan', 'show', 'interfaces']).decode('utf-8')
            return "SSID" in result  # If SSID exists, WiFi is connected
        except:
            return False  # No WiFi

    elif system == "Linux":  # Works for Raspberry Pi
        try:
            result = subprocess.check_output(["iwgetid", "-r"]).decode("utf-8").strip()
            return bool(result)  # Connected if SSID is found
        except:
            return False  # No WiFi

    return False  # Default: No connection

def get_ssid():
    """Get the current SSID (WiFi name) for Windows & Raspberry Pi."""
    system = platform.system()

    if system == "Windows":
        try:
            result = subprocess.check_output(['netsh', 'wlan', 'show', 'interfaces']).decode('utf-8')
            for line in result.split("\n"):
                if "SSID" in line and "BSSID" not in line:
                    return line.split(":")[1].strip()  # Extract SSID
        except:
            return "N/A"

    elif system == "Linux":  # Works for Raspberry Pi
        try:
            return subprocess.check_output(["iwgetid", "-r"]).decode("utf-8").strip()
        except:
            return "N/A"

    return "N/A"

def load_model_in_background():
    import threading
    threading.Thread(target=ensure_model_loaded, daemon=True).start()


def compute_color_properties_colormath(r, g, b):
    rgb = sRGBColor(r / 255.0, g / 255.0, b / 255.0, is_upscaled=False)
    lch = convert_color(convert_color(rgb, LabColor), LCHabColor)

    brightness = lch.lch_l / 100
    hue = lch.lch_h if lch.lch_h is not None else 0
    chroma = lch.lch_c / 100

    return brightness, hue, chroma

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def ensure_model_loaded():
    global model, scaler_rgb, scaler_output, global_data

    # Use relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_rgb_path = os.path.join(base_dir, "scaler_rgb_ID.pkl")
    scaler_output_path = os.path.join(base_dir, "scaler_output_ID.pkl")
    model_path = os.path.join(base_dir, "ID.keras")
    dataset_path = os.path.join(base_dir, "ID.csv")

    try:
        if global_data is None:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
            global_data = pd.read_csv(dataset_path)
            # Add progress tracking
            print("Processing dataset features...")
            features = global_data.apply(lambda row: pd.Series(
                compute_color_properties_colormath(row['R'], row['G'], row['B']),
                index=['Brightness', 'Hue', 'Chroma']
            ), axis=1)
            global_data = pd.concat([global_data, features], axis=1)
            global_data = global_data.sample(frac=1, random_state=42).reset_index(drop=True)
            # Improved weighting logic
            color_counts = global_data.groupby(['R', 'G', 'B']).size().reset_index(name='count')
            global_data = global_data.merge(color_counts, on=['R', 'G', 'B'], how='left')
            global_data['weight'] = 1 / global_data['count']
            if 'DeltaE00' in global_data.columns:
                global_data['weight'] *= np.where(global_data['DeltaE00'] > 4, 0.3, 1.0)
            global_data['weight'] = global_data['weight'] / global_data['weight'].sum()

        X = global_data[['R', 'G', 'B', 'Brightness', 'Hue', 'Chroma']].values
        y = global_data[['Red', 'Yellow', 'Blue', 'Black', 'White']].values
        
        if os.path.exists(scaler_rgb_path) and os.path.exists(scaler_output_path):
            scaler_rgb = joblib.load(scaler_rgb_path)
            scaler_output = joblib.load(scaler_output_path)
        else:
            scaler_rgb = MinMaxScaler(feature_range=(0, 1))
            scaler_output = MinMaxScaler(feature_range=(0, 1))
            scaler_rgb.fit(X) 
            scaler_output.fit(y)
            joblib.dump(scaler_rgb, "scaler_rgb_ID.pkl")
            joblib.dump(scaler_output, "scaler_output_ID.pkl")

        X_scaled = scaler_rgb.fit_transform(X)
        y_scaled = scaler_output.fit_transform(y)

        if os.path.exists(model_path) and os.path.exists(scaler_rgb_path) and os.path.exists(scaler_output_path):
            print("Loading existing model and scalers...")
            model = keras.models.load_model(model_path)
        else:
            print("Training new model...")
            # Improved model architecture
            model = Sequential([
                Input(shape=(6,)),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(5, activation='linear')
            ])
            
            model.compile(
            optimizer=Adam(learning_rate=0.0005), 
            loss='mse',  # More robust loss function
            weighted_metrics=['mse']
            )
            
            model.fit(
                X_scaled, y_scaled,
                epochs=500,
                batch_size=16,
                validation_split=0.2,
                sample_weight=global_data['weight'].values,
                callbacks=[early_stopping],
                shuffle=True
            )
        model.save(model_path)  #  TensorFlow format
        print("Model and scalers initialized successfully")
        print("Scaler RGB min_: ", scaler_rgb.data_min_)
        print("Scaler RGB max_: ", scaler_rgb.data_max_)
        print("Scaler Output min_: ", scaler_output.data_min_)
        print("Scaler Output max_: ", scaler_output.data_max_)

        import hashlib

        def get_file_hash(path):
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()

        model_hash = get_file_hash(model_path)
        print(f"Model file hash: {model_hash}")

    except Exception as e:
        print(f"Initialization error: {str(e)}")
        raise RuntimeError("Failed to initialize model and scalers") from e
        
# Retrain with new corrected data
if os.path.exists(CORRECTED_PATH) and os.path.getsize(CORRECTED_PATH) > 0:
    corrected_df = pd.read_csv(CORRECTED_PATH, names=global_data.columns, skiprows=1).dropna().astype(float)
    combined_data = pd.concat([global_data, corrected_df], ignore_index=True)

    # Prepare training data
    X = combined_data[['R', 'G', 'B', 'Brightness', 'Hue', 'Chroma']].values
    y = combined_data[['Red', 'Yellow', 'Blue', 'Black', 'White']].values
    X_scaled = scaler_rgb.fit_transform(X)
    y_scaled = scaler_output.fit_transform(y)
    model.fit(X_scaled, y_scaled, epochs=10, batch_size=8)
    model.save(model_path)
    corrected_df.to_csv(backup_data_path, mode='a', header=not os.path.exists(backup_data_path), index=False)
    open(CORRECTED_PATH, 'w').close()
    print(" Corrected data moved to backup and cleared.")

else:
    print(" No new data. Model not retrained.")


def calculate_delta_e(predicted_rgb, feedback_rgb):
    color1_rgb = sRGBColor(*[min(255, max(0, c)) / 255.0 for c in predicted_rgb], is_upscaled=False)
    color2_rgb = sRGBColor(*[min(255, max(0, c)) / 255.0 for c in feedback_rgb], is_upscaled=False)

    color1_lab = convert_color(color1_rgb, LabColor, target_illuminant='d65')
    color2_lab = convert_color(color2_rgb, LabColor, target_illuminant='d65')

    print(f" Debug Lab Values - Predicted: {color1_lab}, Feedback: {color2_lab}")

    #Just return as float without .item()
    delta_e = delta_e_cie2000(color1_lab, color2_lab) / 2
    return float(delta_e)  # Explicit float to avoid surprises


def scale_colorant_to_range(colorant, min_total=100, max_total=120):

    red, yellow, blue, black, white = colorant
    total_colorant = red + yellow + blue + black
    total = total_colorant + white
    original_white = white  # Store original prediction

    if total < min_total:
        print(f" Scaling up: Total mix is {total:.2f}mL (below {min_total}mL). Adjusting...")
        scale_factor = min_total / total
        red, yellow, blue, black = [c * scale_factor for c in (red, yellow, blue, black)]
        white = min_total - (red + yellow + blue + black)  # Maintain total balance
    elif total > max_total:
        print(f" Scaling down: Total mix is {total:.2f}mL (above {max_total}mL). Adjusting...")
        scale_factor = max_total / total
        red, yellow, blue, black = [c * scale_factor for c in (red, yellow, blue, black)]
        white = max_total - (red + yellow + blue + black)  # Maintain total balance
    if original_white == 250 and total_colorant > 0:
        required_white = total_colorant * 16  # Maintain 1:16 ratio
        print(f"Adjusting white to maintain 1:16 ratio (Required: {required_white:.2f}mL).")
        white = required_white


    # Final total check
    final_total = sum([red, yellow, blue, black, white])
    if final_total < min_total:
        white += min_total - final_total
    elif final_total > max_total:
        scale_factor = max_total / final_total
        red, yellow, blue, black, white = [c * scale_factor for c in (red, yellow, blue, black, white)]

    return [round(red, 2), round(yellow, 2), round(blue, 2), round(black, 2), round(white, 2)]

def find_closest_colorant(r, g, b):
    """
    Find the closest colorant using DeltaE00 first.
    Override alternatives with hue-check for colors where hue accuracy is crucial.
    """
    input_rgb = sRGBColor(r / 255.0, g / 255.0, b / 255.0, is_upscaled=False)
    input_lab = convert_color(input_rgb, LabColor)
    input_brightness, input_hue, input_chroma = compute_color_properties_colormath(r, g, b)

    min_delta_e = float("inf")
    best_match = None
    best_match_rgb = None
    best_match_hue_diff = 999  # Track the hue difference of the best match
    alternative_matches = []

    # Step 1: Find closest DeltaE00 match (always track the closest)
    for _, row in global_data.iterrows():
        dataset_rgb = sRGBColor(row['R'] / 255.0, row['G'] / 255.0, row['B'] / 255.0, is_upscaled=False)
        dataset_lab = convert_color(dataset_rgb, LabColor)
        dataset_brightness, dataset_hue, dataset_chroma = row['Brightness'], row['Hue'], row['Chroma']

        delta_e = delta_e_cie2000(input_lab, dataset_lab) / 2
        hue_diff = abs(input_hue - dataset_hue) if dataset_hue is not None else 999

        # Track lowest DeltaE00 match and store hue difference
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_match = row[['Red', 'Yellow', 'Blue', 'Black', 'White']].values
            best_match_rgb = (row['R'], row['G'], row['B'])
            best_match_hue_diff = hue_diff

        # Store potential alternatives with DeltaE00 between 2 and 3
        if 2 < delta_e <= 3:
            alternative_matches.append({
                "delta_e": delta_e,
                "colorant": row[['Red', 'Yellow', 'Blue', 'Black', 'White']].values,
                "rgb": (row['R'], row['G'], row['B']),
                "brightness_diff": abs(input_brightness - dataset_brightness),
                "hue_diff": hue_diff,
                "chroma_diff": abs(input_chroma - dataset_chroma)
            })
    def scale_colorant(colorant):
        """Scale colorant to 100-120mL range while maintaining proportions"""
        total = sum(colorant)
        if total <= 0:
            return colorant  # Avoid division by zero
            
        # Calculate initial scaling factor to reach 100mL
        target_total = 100
        scale_factor = target_total / total
        
        # Check if scaling exceeds 120mL
        if (total * scale_factor) > 120:
            scale_factor = 120 / total
            
        scaled = [x * scale_factor for x in colorant]
        return [round(x, 2) for x in scaled]
    if best_match is not None:
        best_match = scale_colorant(best_match)
        print(f" Closest DeltaE00 Match (Scaled): RGB({best_match_rgb[0]}, {best_match_rgb[1]}, {best_match_rgb[2]}) → RYBKW: {np.round(best_match, 2)} (DeltaE00: {min_delta_e:.2f})")

    # Step 3: If DeltaE00 ≤ 1.5, return the closest match immediately
    if min_delta_e <= 1.5:
        print(f"Found a close dataset match. Using it directly (DeltaE00: {min_delta_e:.2f})")
        return scale_colorant_to_range(best_match), min_delta_e

    # Step 4: If DeltaE00 is between 2 and 3, check alternatives with stricter thresholds
    if alternative_matches:
        print("Checking alternative matches with stricter brightness, hue, and chroma limits...")

        # Thresholds: Tightened up for hue-sensitive colors
        hue_threshold = 5  # Hue override — ±5° max!
        brightness_threshold = 0.05
        chroma_threshold = 0.1

        # Sort alternatives: prioritize DeltaE00 first, but ensure hue stays locked in
        alternative_matches.sort(
            key=lambda x: (x['delta_e'], x['hue_diff'], x['brightness_diff'], x['chroma_diff'])
        )

        for alt in alternative_matches:
            if (
                alt['hue_diff'] < hue_threshold and
                alt['brightness_diff'] < brightness_threshold and
                alt['chroma_diff'] < chroma_threshold
            ):
                # Hue override — pick the alternative only if hue stays within 5°
                if best_match_hue_diff > hue_threshold and alt['hue_diff'] < hue_threshold:
                    print(
                        f"Hue Override: Switching to alternative match with closer hue!"
                        f" RGB({alt['rgb'][0]}, {alt['rgb'][1]}, {alt['rgb'][2]}) → RYBKW: {np.round(alt['colorant'], 2)} "
                        f"(DeltaE00: {alt['delta_e']:.2f})"
                    )
                    return scale_colorant_to_range(alt['colorant']), alt['delta_e']
                else:
                    print(
                        f"Using alternative match with better brightness/chroma: "
                        f"RGB({alt['rgb'][0]}, {alt['rgb'][1]}, {alt['rgb'][2]}) → RYBKW: {np.round(alt['colorant'], 2)} "
                        f"(DeltaE00: {alt['delta_e']:.2f})"
                    )
                    return scale_colorant_to_range(alt['colorant']), alt['delta_e']

    # Step 5: If no better alternative found, fallback to closest DeltaE00 match
    print(f" No better match found in dataset. Using closest DeltaE00 match (DeltaE00: {min_delta_e:.2f})")
    return best_match, min_delta_e

def predict_colorant(r, g, b):
    global model, scaler_rgb, scaler_output, global_data
    
    print(f"\n Debugging: R={r}, G={g}, B={b}")
    brightness, hue, chroma = compute_color_properties_colormath(r, g, b)
    print(f" Initial Brightness: {brightness:.2f} |  Initial Chroma: {chroma:.2f} |  Initial Hue: {hue:.2f}")
    
    if not isinstance(r, (int, float)) or not isinstance(g, (int, float)) or not isinstance(b, (int, float)):
        print("⚠ ERROR: RGB values must be numbers.")
        return None

    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        print("⚠ ERROR: Invalid RGB values (0-255 required).")
        return None
    #Step 1: Check for closest match in dataset
    dataset_match, dataset_delta_e = find_closest_colorant(r, g, b)
    if dataset_match is None:
        print("ERROR: find_closest_colorant() returned None! Blending will not happen.")
        return None

    if dataset_match is not None and dataset_delta_e <= 1.5:
        red, yellow, blue, black, white = dataset_match
        
        if brightness >= 0.90:
            print(" Adjusting for near-white colors...")
            
            # Store original values
            original_red = red
            original_yellow = yellow
            original_blue = blue
            original_black = black
            original_white = white
            
            # Initialize reductions
            red_reduction = yellow_reduction = blue_reduction = black_reduction = 0
            
            if brightness > 0.95:
                # Cap total colorant to 0.8mL for very bright colors
                max_colorant = 0.8
                total_colorant = red + yellow + blue + black
                if total_colorant > max_colorant:
                    scale_factor = max_colorant / total_colorant
                    red_reduction = red * (1 - scale_factor)
                    yellow_reduction = yellow * (1 - scale_factor)
                    blue_reduction = blue * (1 - scale_factor)
                    black_reduction = black * (1 - scale_factor)
                    
                    red *= scale_factor
                    yellow *= scale_factor
                    blue *= scale_factor
                    black *= scale_factor
            else:
                # Reduce each colorant by exactly 10%
                reduction = 0.1
                red_reduction = red * reduction
                yellow_reduction = yellow * reduction 
                blue_reduction = blue * reduction
                black_reduction = black * reduction
                
                red -= red_reduction
                yellow -= yellow_reduction
                blue -= blue_reduction
                black -= black_reduction
            
            # Calculate total reduction and adjust white
            total_reduction = red_reduction + yellow_reduction + blue_reduction + black_reduction
            white = original_white + total_reduction + (original_white * 0.05)
            
            # Ensure white doesn't exceed 100%
            white = min(white, 100)
            
            adjustment_applied = True
            print(f" After near-white adjustments:")
            print(f"  Red: {original_red:.2f} → {red:.2f}mL")
            print(f"  Yellow: {original_yellow:.2f} → {yellow:.2f}mL") 
            print(f"  Blue: {original_blue:.2f} → {blue:.2f}mL")
            print(f"  Black: {original_black:.2f} → {black:.2f}mL")
            print(f"  White: {original_white:.2f} → {white:.2f}mL")
            print(f"Total {red+yellow+blue+black+white:.2f}mL")
        
        return [red, yellow, blue, black, white]

    # Step 2: Predict colorant with Neural Network
    rgb_scaled = scaler_rgb.transform([[r, g, b, brightness, hue, chroma]])
    predicted_scaled = model.predict(rgb_scaled)
    predicted_colorant = np.maximum(scaler_output.inverse_transform(predicted_scaled)[0], 0)
    # Scale the NN prediction to 100-120mL range
    def scale_to_range(colorant):
        total = sum(colorant)
        if total <= 0:
            return colorant
        
        # First: Scale to 100-120mL
        target_total = 100
        scale_factor = target_total / total
        if (total * scale_factor) > 120:
            scale_factor = 120 / total
        scaled = [x * scale_factor for x in colorant]
        
        # Second: Enforce minimum white (20mL) if prediction was zero
        if scaled[4] < 20.0:  # Index 4 = White
            scaled[4] = 20.0  # Set white to 20mL
            print("⚠ Enforced minimum white: 20mL")
        
        return scaled
    predicted_colorant = scale_to_range(predicted_colorant)
    print(f" Neural Network Prediction (RYBKW - Scaled): {np.round(predicted_colorant, 2)}")

    # Blending: Only happens when dataset match is available
    if dataset_match is not None:
        print(f"Blending NN prediction with dataset match (DeltaE00: {dataset_delta_e:.2f})")
        if dataset_delta_e <= 1.8:
            dataset_weight = 0.85
        elif dataset_delta_e <= 2:
            dataset_weight = 0.75
        elif dataset_delta_e <= 3:
            dataset_weight = 0.5
        else:
            dataset_weight = 0.0  # NN takes over completely

        nn_weight = 1 - dataset_weight
        blended_colorant = [(p * nn_weight + d * dataset_weight) for p, d in zip(predicted_colorant, dataset_match)]
        print(f"Final Blended Colorant Before Adjustments: {np.round(blended_colorant, 2)}")
        red, yellow, blue, black, white = blended_colorant
    else:
        red, yellow, blue, black, white = predicted_colorant  # Use pure NN prediction

    adjustment_applied = False 

    if brightness > 0.90 and not adjustment_applied:
        print(" Adjusting for very bright colors (near white)...")
        scale_factor = max(0.4, 0.2 + chroma * 5) if chroma < 0.1 else max(0.8, 0.75 + chroma * 3)
        red = max(0.1, red * scale_factor)
        yellow = max(0.1, yellow * scale_factor)
        blue = max(0.1, blue * scale_factor)

        # Optional channel adjustments based on hue range:
        if 180 <= hue < 240:
            blue = max(blue, 0.3 * chroma)
        if 30 <= hue < 90:
            yellow = max(yellow, 0.3 * chroma)

        white = max(white, 100 - (red + yellow + blue) * (1 + chroma * 0.2))
        black = 0  
        total_colorant = red + yellow + blue
        if total_colorant > max_colorant:
            scaling_factor = max_colorant / total_colorant
            red *= scaling_factor
            yellow *= scaling_factor
            blue *= scaling_factor

        adjustment_applied = True
        print(f" After near-white adjustments: [R={round(red, 2)}, Y={round(yellow, 2)}, B={round(blue, 2)}, K={round(black, 2)}, W={round(white, 2)}]")

    # --- Hue-Based Adjustments ---
    hue_adjustments = {
        (0, 30): {"red": 0.15, "yellow": 0.05},
        (30, 60): {"red": 0.1, "yellow": 0.18},
        (60, 90): {"red": 0.05, "yellow": 0.25},
        (90, 150): {"yellow": 0.15, "blue": 0.1},
        (150, 180): {"blue": 0.15, "yellow": 0.05},
        (180, 210): {"blue": 0.3},
        (210, 240): {"blue": 0.35, "red": 0.02},
        (240, 270): {"blue": 0.2, "red": 0.05},
        (270, 300): {"blue": 0.15, "red": 0.15},
        (300, 330): {"red": 0.2, "blue": 0.1},
        (330, 360): {"red": 0.2}
    }
    hue_adjustment_done = False
    # Apply hue adjustments only if brightness is not too high and chroma is significant
    if brightness < 0.9 and chroma > 0.005:
        for (h_min, h_max), adj in hue_adjustments.items():
            if h_min <= hue < h_max:
                factor = max(0.5, chroma * 2)
                red += adj.get("red", 0) * factor
                yellow += adj.get("yellow", 0) * factor
                blue += adj.get("blue", 0) * factor
                hue_adjustment_done = True
                print(f" After hue adjustments: [R={round(red, 2)}, Y={round(yellow, 2)}, B={round(blue, 2)}, K={round(black, 2)}, W={round(white, 2)}]")
                break

    # --- Low Chroma Adjustments ---
    if chroma < 0.02 and not adjustment_applied:
        print("Low chroma detected — boosting corrections...")
        red *= 1.2
        yellow *= 1.2
        blue *= 1.1
        # Remove special handling for any specific hue here; apply uniform adjustments
        adjustment_applied = True

    # --- Final Scaling and Balancing ---
    total = red + yellow + blue + black + white
    red, yellow, blue, black, white = [round(c / total * 100, 2) for c in [red, yellow, blue, black, white]]
    print(f"Final Colorant (Total: {round(total, 2)} mL) →  Red: {red:.2f} mL,  Yellow: {yellow:.2f} mL,  Blue: {blue:.2f} mL,  Black: {black:.2f} mL,  White: {white:.2f} mL")
    return [red, yellow, blue, black, white]

def retrain_model(minor_adjustment=False):
    global model, X, y, scaler_rgb, scaler_output

    # Check if the corrected data file exists and is not empty
    if not os.path.exists(CORRECTED_PATH) or os.path.getsize(CORRECTED_PATH) == 0:
        print("No new data. Skipping retraining.")
        return

    # Load the corrected feedback data
    corrected_df = pd.read_csv(CORRECTED_PATH, header=0)  # Read with existing headers
    if corrected_df.empty or len(corrected_df) < 5:
        print("Not enough high-quality feedback data. Skipping retraining to avoid overfitting.")
        return  # Exit if not enough good corrections

    # Ensure all values are numeric and drop any NaN values
    corrected_df = corrected_df.dropna().astype(float)

    # Compute additional features for the corrected data
    features_corrected = corrected_df.apply(lambda row: pd.Series(
        compute_color_properties_colormath(int(row['R']), int(row['G']), int(row['B'])),
        index=['Brightness', 'Hue', 'Chroma']
    ), axis=1)
    corrected_df = pd.concat([corrected_df, features_corrected], axis=1)

    # Load all feedback data, including rejected cases
    all_feedback = pd.read_csv(dataset_path)  # Load full dataset including rejected cases
    full_data = pd.concat([all_feedback, corrected_df], ignore_index=True)  # Merge corrected & rejected
    full_data.drop_duplicates(inplace=True)  # Remove duplicate entries

    # Fit scalers on the combined dataset
    scaler_rgb.partial_fit(full_data[['R', 'G', 'B', 'Brightness', 'Hue', 'Chroma']])  
    scaler_output.partial_fit(full_data[['Red', 'Yellow', 'Blue', 'Black', 'White']])

    # Scale the features and targets
    X_scaled = scaler_rgb.transform(full_data[['R', 'G', 'B', 'Brightness', 'Hue', 'Chroma']])
    y_scaled = scaler_output.transform(full_data[['Red', 'Yellow', 'Blue', 'Black', 'White']])

    # Adjust training parameters based on minor or major adjustment
    if minor_adjustment:
        epochs = 10  # Reduce minor retraining cycles
        learning_rate = 0.0001  # Lower learning rate
        print("Minor adjustment mode: low epochs and learning rate.")
    else:
        epochs = 30  # Full retraining, but limit to prevent overfitting
        learning_rate = 0.0005  # Normalized LR

    # Recompile model with adjusted learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Apply EarlyStopping again
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

    # Save the retrained model
    model.save(model_path)

    # Backup the corrected data
    corrected_df.to_csv(backup_data_path, mode='a', header=not os.path.exists(backup_data_path), index=False)

    # Clear the corrected data file after processing
    open(CORRECTED_PATH, 'w').close()
    print("Model retrained and updated.")

def handle_feedback():
    global latest_rgb, predicted_colorant, awaiting_feedback

    # Check if latest_rgb is None
    if latest_rgb is None:
        print("No RGB data available! Please scan a color before entering feedback.")
        awaiting_feedback = False
        return None  # Return None if no RGB data

    r, g, b = latest_rgb  # Now this will only execute if latest_rgb is not None

    predicted_colorant = predict_colorant(r, g, b)
    
    if predicted_colorant is None:
        print("No prediction available. Please predict a color first!")
        awaiting_feedback = False
        return None  # Return None if no prediction

    # Compute DeltaE00
    delta_e = calculate_delta_e(latest_rgb, predicted_colorant)
    print(f"DeltaE00: {delta_e:.2f}")

    # Handle corrections based on Delta E
    if delta_e > 6:
        print("DeltaE00 too high! Storing in dataset but skipping retraining.")
        save_feedback(latest_rgb, predicted_colorant, corrected=False)
        return delta_e  # Return the delta_e value

    elif delta_e < 1.8:
        print("Excellent match! No adjustment needed.")
        return delta_e  # Return the delta_e value

    # Adjust and retrain based on DeltaE00
    correction_factor = 0.7 if delta_e > 2 else 1 - ((delta_e - 1) / 1)
    adjusted_colorant = [(p * (1 - correction_factor) + f * correction_factor) for p, f in zip(predicted_colorant, predicted_colorant)]
    save_feedback(latest_rgb, adjusted_colorant, corrected=True)
    retrain_model(minor_adjustment=(delta_e < 4))

    return delta_e  # Return the delta_e value

def save_feedback(mixed_rgb, colorant_values, corrected=True):
    dataset_file = r"c:\Users\Ranillo\Desktop\files\Thesis\TrainingAccuracy\ID.csv"
    corrected_file = r"c:\Users\Ranillo\Desktop\files\Thesis\TrainingAccuracy\IDcorrected.csv"

    feedback_data = pd.DataFrame([[*mixed_rgb, *colorant_values]],
                                 columns=['R', 'G', 'B', 'Red', 'Yellow', 'Blue', 'Black', 'White'])

    try:
        # Append feedback to main dataset
        feedback_data.to_csv(dataset_file, mode='a', header=not os.path.exists(dataset_file), index=False)
        print(f"Feedback saved to {dataset_file}!")

        if corrected:
            # Store in corrected dataset only if it was manually corrected
            feedback_data.to_csv(corrected_file, mode='a', header=not os.path.exists(corrected_file), index=False)
            print(f"Corrected feedback saved to {corrected_file}!")

    except Exception as e:
        print(f"Error saving feedback: {e}")


# Custom Button with an Image (Eye Icon)
class ImageButton(ButtonBehavior, Image):
    pass

# Function to handle MQTT connection
def on_connect(client, userdata, flags, rc, properties=None):
    print("on_connect triggered")  # Debugging
    if rc == 0:
        print("Connected to MQTT Broker.")
        client.subscribe([(MQTT_TOPIC_INPUT, 0)])
        
        app = App.get_running_app()
        if hasattr(app, 'config_screen') and app.config_screen:
            print("Updating Config Screen MQTT status...")
            app.config_screen.update_mqtt_status("MQTT Connected")
        else:
            print("config_screen is None, cannot update UI.")
    else:
        print(f"Connection failed with code {rc}")

# Function to handle incoming MQTT messages
def on_message(client, userdata, msg):
    global accepting_messages, latest_rgb  # Include latest_rgb in the global variables

    if not accepting_messages:
        print("Ignoring MQTT message during operation.")
        return

    # Add initialization check
    if model is None or scaler_rgb is None or scaler_output is None:
        print("Model not ready - requeuing message")
        Clock.schedule_once(lambda dt: on_message(client, userdata, msg), 0.5)
        return  

    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)
        print(f"Raw MQTT message received: {payload}")

        app = App.get_running_app()
        if not app:
            print("App instance not available")
            return

        if not hasattr(app, 'sm'):
            print("Screen manager not initialized in app")
            return
            
        screen_manager = app.sm

        if 'main' not in screen_manager.screen_names:
            print("Main screen not registered")
            return

        main_screen = screen_manager.get_screen('main')
        if not main_screen:
            print("Could not retrieve main screen instance")
            return

        if msg.topic == MQTT_TOPIC_INPUT:
            try:
                r = int(data.get("r", 0))
                g = int(data.get("g", 0))
                b = int(data.get("b", 0))
                
                if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                    print("RGB values out of range (0-255)")
                    Clock.schedule_once(lambda dt: main_screen.update_status("Error: RGB values out of range (0-255)"))
                    return
                    
            except (TypeError, ValueError) as e:
                print(f"Invalid RGB values received: {e}")
                Clock.schedule_once(lambda dt: main_screen.update_status("Error: Invalid RGB values"))
                return

            print(f"Processing valid RGB: R={r}, G={g}, B={b}")

            # Set the latest_rgb variable
            latest_rgb = (r, g, b)  # Assign RGB values to latest_rgb

            try:
                predicted_colorant = predict_colorant(r, g, b)
                if predicted_colorant is None:
                    print("No prediction available.")
                    return
                
                red, yellow, blue, black, white = predicted_colorant
                
                main_screen.red = red
                main_screen.yellow = yellow
                main_screen.blue = blue 
                main_screen.black = black
                main_screen.white = white

                print(f"Predicted Colorants: Red={red:.2f}, Yellow={yellow:.2f}, "
                      f"Blue={blue:.2f}, Black={black:.2f}, White={white:.2f}")

                Clock.schedule_once(lambda dt: (
                    main_screen.update_rgb(r, g, b),
                    main_screen.update_colorant(red, yellow, blue, black, white),
                    main_screen.show_operation_button(),
                    main_screen.update_status("Status: Value Received")
                ))

            except Exception as e:
                print(f"Color prediction error: {e}")
                Clock.schedule_once(lambda dt: main_screen.update_status(f"Error: {str(e)}"))

    except json.JSONDecodeError:
        print("ERROR: Invalid JSON payload")
        app = App.get_running_app()
        if app and hasattr(app, 'sm'):
            screen_manager = app.sm
            if 'main' in screen_manager.screen_names:
                main_screen = screen_manager.get_screen('main')
                Clock.schedule_once(lambda dt: main_screen.update_status("Error: Invalid data format"))
        else:
            print("Cannot update status - UI components unavailable")
            
    except Exception as e:
        print(f"Critical MQTT handling error: {e}")

def on_disconnect(client, userdata, rc, properties=None, *args):
    print(f"Disconnected with code {rc}")
    if hasattr(App.get_running_app(), 'config_screen'):
        App.get_running_app().config_screen.update_mqtt_status("Disconnected")

def initialize_mqtt():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.initialized = True
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=5, max_delay=60)
    
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start()
        print("MQTT client initialized with auto-reconnect")
        return client  # Return the client instance
    except Exception as e:
        print(f"Initial connection failed: {e}")
        Clock.schedule_once(lambda dt: initialize_mqtt(), 5)
        return None
    
# Start MQTT
initialize_mqtt()

class MainScreen(Screen):
    def __init__(self, mqtt_client, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.mqtt_client = mqtt_client

        # Create the layout
        self.layout = FloatLayout()

        # Ensure fullscreen mode dynamically
        Window.fullscreen = True  # Force fullscreen mode

        # Bind screen size updates for dynamic resizing
        self.bind(size=self.adjust_widgets)

        # Video sources
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.video_sources = {
         "on": os.path.join(base_dir, "Load.mp4"),
         "off": os.path.join(base_dir, "Intro.mp4")
        }

        # Start with the 'off' state (Standby Video)
        self.current_video = "off"

        # Fullscreen background video (Initially hidden)
        self.video_player = Video(
            source=self.video_sources[self.current_video],  # Start with Standby.mp4
            state='play',  # Auto-play on start
            allow_stretch=True, 
            keep_ratio=False,
            size_hint=(1, 1),  # Fullscreen
            pos=(0, 0),
            opacity=0,  # Initially hidden
            options={'eos': 'loop'}  # Loop video when it ends
        )
        self.layout.add_widget(self.video_player)

        # Widgets Container (Ensures buttons are on top)
        self.widgets_container = FloatLayout(opacity=0, size_hint=(1, 1))

        # CONFIG Button (Lower-Right Corner)
        self.config_button = Button(
            text="CONFIG",
            font_size=16,
            size_hint=(None, None),
            size=(120, 40),
            pos_hint={'right': 1, 'bottom': 1},  # Lower-right corner
            background_color=(0, 0, 1, 1)
        )
        self.config_button.bind(on_press=self.go_to_config)
        self.widgets_container.add_widget(self.config_button)

        # RGB Label
        self.rgb_label = Label(
            text="Waiting for RGB values...", 
            font_size=22,
            color=(0, 0, 0, 1),
            size_hint=(None, None),  
            size=(300, 50),  
            pos_hint={'x': 0.35, 'y': 0.25},
            halign="left" 
        )
        self.widgets_container.add_widget(self.rgb_label)

        # Colorant Label 
        self.colorant_label = Label(
            text="Predicted Colorants: None", 
            font_size=22,
            color=(0, 0, 0, 1),
            size_hint=(None, None),
            size=(300, 50),
            pos_hint={'x': 0.35, 'y': 0.2},  
            halign="left"
        )
        self.widgets_container.add_widget(self.colorant_label)

        app = App.get_running_app()
        config_screen = app.config_screen if hasattr(app, 'config_screen') else None

        # Status
        self.status_label = Label(
            text="Status: Standby", 
            font_size=22,
            color=(0, 0, 0, 1),
            size_hint=(None, None),
            size=(300, 50),
            pos_hint={'x': 0.35, 'y': 0.15},
            halign="left"
        )	
        self.widgets_container.add_widget(self.status_label)

        # Proceed Button
        self.proceed_button = Button(
            text="Proceed",
            font_size=16,
            size_hint=(None, None),
            size=(120, 40),
            pos_hint={'center_x': 0.5, 'y': 0.08},
            background_color=(0, 1, 0, 1),
            opacity=0,
            disabled=True
        )
        self.proceed_button.bind(on_press=self.proceed_operation)
        self.widgets_container.add_widget(self.proceed_button)

        # Add widgets container on top of the video
        self.layout.add_widget(self.widgets_container)

        self.add_widget(self.layout)

        # Resize video when window resizes
        Window.bind(size=self.adjust_video_size)

        # Start fade-in animations
        Clock.schedule_once(self.fade_in_video, 5)

        global accepting_messages
        accepting_messages = True

    def adjust_video_size(self, instance, value):
        """Ensures the video stays fullscreen when the window resizes."""
        self.video_player.size = Window.size  # Match the window size

        self.video_player.pos = (0, 0)  # Keep at top-left

    def fade_in_video(self, dt):
        """Fades in the background video after a delay."""
        anim = Animation(opacity=1, duration=2)  # Smooth fade-in over 2 sec
        anim.bind(on_complete=self.fade_in_widgets)  # After video fades in, fade widgets
        anim.start(self.video_player)

    def fade_in_widgets(self, animation, widget):
        """Fades in all widgets after the video has fully appeared."""
        anim = Animation(opacity=1, duration=1)  # Widgets fade in over 1 sec
        anim.start(self.widgets_container)

    def show_widgets(self, dt):
        anim = Animation(opacity=1, duration=1)
        anim.start(self.widgets_container)

    def go_to_config(self, instance):
        """Navigate to Config Screen."""
        self.manager.current = 'config'

    def update_rgb(self, r, g, b):
        self.rgb_label.text = f"Received RGB: R={r}, G={g}, B={b}"

    def update_colorant(self, red, yellow, blue, black, white):
        self.red, self.yellow, self.blue, self.black, self.white = red, yellow, blue, black, white
        self.colorant_label.text = (
    f"Predicted: "
    f"Red={red:.2f}, Yellow={yellow:.2f}, Blue={blue:.2f}, "
    f"Black={black:.2f}, White={white:.2f}")

    def show_operation_button(self):	
        if hasattr(self, 'proceed_button'):  
            print("Showing Proceed Button!")  # Debugging output
            self.proceed_button.opacity = 1
            self.proceed_button.disabled = False
        else:
            print("Error: Proceed button not found!")

    def update_status(self, status_text):
        self.status_label.text = status_text

    def proceed_operation(self, instance):
        """Handles the operation when the 'Proceed' button is pressed."""
        self.current_video = "on"  # Switch to ongoing video
        self.video_player.source = self.video_sources[self.current_video]
        self.video_player.state = 'play'  # Play new video
        global accepting_messages

        # Restart video playback after changing the source
        Clock.schedule_once(lambda dt: setattr(self.video_player, 'state', 'play'), 0.5)


        self.update_status("Status: Ongoing Operation")
        self.proceed_button.opacity = 0
        self.proceed_button.disabled = True  # Disable until operation completes

        # Stop MQTT messages during operation
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.accepting_messages = False
            print("MQTT message handling paused.")

        if arduino and arduino.is_open:
            send_command('ON')  # Send START command
            print("Sent 'ON' command to Arduino")

            # Send the stored colorant values
            rounded_values = [
                round(self.red, 2),
                round(self.yellow, 2),
                round(self.blue, 2),
                round(self.black, 2),
                round(self.white, 2)
            ]
            send_command("COLORANT", *rounded_values)
            print(f"Sent COLORANT values: {', '.join(f'{v:.2f}' for v in rounded_values)}")

        else:											
            print("Arduino not connected. Could not send colorant values.")

        accepting_messages = False  # Pause MQTT during operation
        self.proceed_button.disabled = True

    def adjust_widgets(self, *args):
        """Ensure video covers full screen dynamically."""
        self.video_player.size = Window.size
        self.video_player.pos = (0, 0)

    def handle_operation_complete(self):
        """Called when Arduino sends DONE signal"""
        # Reset video first
        self.current_video = "off"
        self.video_player.source = self.video_sources[self.current_video]
        self.video_player.state = 'play'

        # Ensure any existing popup is dismissed
        if hasattr(self, 'feedback_popup') and self.feedback_popup:
            self._dismiss_popup()

        # Reset UI state
        self.proceed_button.disabled = False
        self.proceed_button.opacity = 0
        self.status_label.text = "Operation Complete"
        
        # Delay popup after video transition completes
        Clock.schedule_once(lambda dt: self.show_feedback_popup(), 1.0)  # Increased delay

        # Don't clear values yet - wait for feedback decision
        self.update_status("Operation Complete")

        # Defer MQTT enable to after feedback
        global accepting_messages
        accepting_messages = False  # Keep disabled until feedback handled

    def show_feedback_popup(self):
        """Show feedback popup with state checks"""
        if hasattr(self, 'feedback_popup') and self.feedback_popup:
            return  # Prevent duplicates

        # Cleanup previous overlay completely
        self._cleanup_overlay()

        # Create new overlay
        self.overlay = FloatLayout(size=self.size)
        with self.overlay.canvas.before:
            Color(0, 0, 0, 0.5)
            self.overlay_rect = Rectangle(size=self.size, pos=self.pos)
        
        self.overlay.bind(size=self.update_overlay_rect, pos=self.update_overlay_rect)
        self.add_widget(self.overlay)

        # Create popup with persistent reference
        self.feedback_popup = Popup(
            title="Feedback",
            content=self._create_popup_content(),
            size_hint=(0.8, 0.5),
            auto_dismiss=False
        )
        self.feedback_popup.bind(on_dismiss=self._on_popup_dismissed)
        self.feedback_popup.open()

    def _create_popup_content(self):
        """Create popup UI components"""
        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        message_label = Label(text="Would you like to give feedback?", 
                            size_hint_y=None, 
                            height=50)
        
        yes_button = Button(text="Yes", size_hint_y=None, height=50)
        yes_button.bind(on_press=self._handle_yes)
        
        no_button = Button(text="No", size_hint_y=None, height=50)
        no_button.bind(on_press=self._handle_no)

        popup_layout.add_widget(message_label)
        popup_layout.add_widget(yes_button)
        popup_layout.add_widget(no_button)
        
        return popup_layout

    def _handle_yes(self, instance):
        """Handle Yes button with sequenced cleanup"""
        self._dismiss_popup()
        self.show_feedback_input(instance)

    def _handle_no(self, instance):
        """Handle No button with full cleanup"""
        self._dismiss_popup()
        self._finalize_cleanup()

    def _dismiss_popup(self):
        """Safely dismiss popup"""
        if self.feedback_popup:
            self.feedback_popup.dismiss()
            
    def show_feedback_input(self, instance):
        """Show the RGB input prompt for user feedback."""
        self.feedback_input_popup = Popup(
            title="Enter RGB Values",
            content=self._create_rgb_input_content(),
            size_hint=(0.8, 0.5),
            auto_dismiss=False
        )
        self.feedback_input_popup.bind(on_dismiss=self._on_feedback_input_dismissed)
        self.feedback_input_popup.open()

    def _create_rgb_input_content(self):
        """Create the content for the RGB input popup with numeric keyboard support."""
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Numeric-only input fields with keyboard binding
        self.feedback_r = TextInput(
            hint_text="R (0-255)",
            multiline=False,
            input_filter='int',  # Restrict to integers
            write_tab=False,     # Disable tab switching
        )
        self.feedback_g = TextInput(
            hint_text="G (0-255)",
            multiline=False,
            input_filter='int',
            write_tab=False
        )
        self.feedback_b = TextInput(
            hint_text="B (0-255)",
            multiline=False,
            input_filter='int',
            write_tab=False
        )

        # Bind focus events for keyboard control
        for input_field in [self.feedback_r, self.feedback_g, self.feedback_b]:
            input_field.bind(focus=self.handle_focus)

        layout.add_widget(self.feedback_r)
        layout.add_widget(self.feedback_g)
        layout.add_widget(self.feedback_b)

        # Buttons
        submit_button = Button(text="Submit")
        submit_button.bind(on_press=self.submit_feedback)
        layout.add_widget(submit_button)

        cancel_button = Button(text="Cancel")
        cancel_button.bind(on_press=self.cancel_feedback)
        layout.add_widget(cancel_button)

        return layout

    def _on_popup_dismissed(self, instance):
        """Handle popup dismissal completion"""
        self._finalize_cleanup()
        self.feedback_popup = None  
        self._enable_mqtt()

    def _finalize_cleanup(self):
        """Complete cleanup operations"""
        self._cleanup_overlay()
        self.clear_screen_values()
        self.update_status("Status: Ready")

    def _cleanup_overlay(self):
        """Remove overlay completely"""
        if hasattr(self, 'overlay'):
            try:
                self.remove_widget(self.overlay)
            except ValueError:
                pass
            del self.overlay

    def _enable_mqtt(self):
        """Re-enable MQTT after cleanup"""
        global accepting_messages
        accepting_messages = True
        
    def update_overlay_rect(self, instance, value):
        """Update the overlay rectangle size and position."""
        self.overlay_rect.pos = self.overlay.pos
        self.overlay_rect.size = self.overlay.size

    def remove_overlay(self, instance):
        """Remove the overlay when the popup is dismissed."""
        if hasattr(self, 'overlay') and self.overlay in self.children:
            self.remove_widget(self.overlay)

    def dismiss_popup(self, instance):
        """Dismiss the feedback popup."""
        if self.feedback_popup:
            print("Dismissing feedback popup...")
            self.feedback_popup.dismiss()
        else:
            print("Feedback popup is not initialized.")

    def dismiss_initial_popup(self, instance):
        """Dismiss the initial feedback popup and clean up completely"""
        # Dismiss popup
        if self.feedback_popup:
            print("Dismissing initial feedback popup...")
            self.feedback_popup.dismiss()
            self.feedback_popup = None  # Clear reference
        
        # Remove overlay if it exists
        if hasattr(self, 'overlay') and self.overlay in self.children:
            self.remove_widget(self.overlay)

        # Update status directly instead of using do_update()
        self.update_status("Status: Ready")
        Clock.schedule_once(lambda dt: self.clear_screen_values(), 0)

    def submit_feedback(self, instance):
        try:
            r = int(self.feedback_r.text.strip())
            g = int(self.feedback_g.text.strip())
            b = int(self.feedback_b.text.strip())

            if not all(0 <= x <= 255 for x in (r, g, b)):
                show_warning_popup("RGB values must be between 0-255")
                return

            # Schedule cleanup on main thread
            Clock.schedule_once(lambda dt: (
                self.feedback_input_popup.dismiss(),
                self.remove_overlay(None),
                self.clear_screen_values()
            ), 0)

            # Start feedback handling thread
            threading.Thread(
                target=self.handle_feedback_thread,
                args=(r, g, b),
                daemon=True
            ).start()

        except ValueError:
            self.update_status("Invalid input: Numbers only (0-255)")

    def show_delta_e_result_popup(self, delta_e):
        """Show Delta E result popup."""
        if hasattr(self, 'delta_e_popup') and self.delta_e_popup:
            return  # Prevent duplicates

        # Create the message based on Delta E value
        threshold = 2.0  # Define your threshold for passing
        if delta_e <= threshold:
            message = f"Pass: Delta E is {delta_e:.2f}, which is within the acceptable range."
        else:
            message = f"Fail: Delta E is {delta_e:.2f}, which exceeds the acceptable range."

            # Create the popup
            self.delta_e_popup = Popup(
                title="Delta E Result",
                content=Label(text=message),
                size_hint=(0.8, 0.5),
                auto_dismiss=False
            )

            # Add an OK button to dismiss the popup and go back to standby
            ok_button = Button(text="OK")
            ok_button.bind(on_press=self.dismiss_delta_e_popup)
            self.delta_e_popup.add_widget(ok_button)

            self.delta_e_popup.open()

    def dismiss_delta_e_popup(self, instance):
        """Dismiss the Delta E result popup and reset to standby."""
        if self.delta_e_popup:
            self.delta_e_popup.dismiss()
            self.delta_e_popup = None  # Clear reference
            self.reset_to_standby()  # Reset to standby state
            
    def _on_feedback_input_dismissed(self, instance):
        """Handle the dismissal of the feedback input popup."""
        self.feedback_input_popup = None  # Clear the reference to the popup
        # You can add any additional cleanup or state resetting here if needed

    def cancel_feedback(self, instance):
        """Cancel the feedback process."""
        Clock.schedule_once(lambda dt: (
            self.feedback_input_popup.dismiss(),
            self.remove_overlay(None),
            self.clear_screen_values(),
            self.update_status("Feedback process canceled.")
        ), 0)

    def remove_overlay(self, instance):
        """Safely remove overlay if it exists"""
        if hasattr(self, 'overlay'):
            try:
                if self.overlay in self.children:
                    self.remove_widget(self.overlay)
            except ReferenceError:
                pass
            finally:
                del self.overlay

    def reset_to_standby(self):
        """Reset the screen to standby state."""
        self.current_video = "off"  # Reset to standby video
        self.video_player.source = self.video_sources[self.current_video]
        self.video_player.state = 'play'  # Play the standby video
        self.update_status("Status: Standby")  # Update status label

    def skip_feedback(self, overlay):
        print("Feedback skipped.")
        self.layout.remove_widget(overlay)
        self.clear_screen_values()  # Clear the screen values
        self.reset_to_standby()  # Reset to standby state
        self.mqtt_client.accepting_messages = True

    def clear_screen_values(self):
        """Reset the screen values to their initial state."""
        self.rgb_label.text = "Waiting for RGB values..."  # Reset RGB label
        self.colorant_label.text = "Predicted Colorants: None"  # Reset colorant label
        self.status_label.text = "Status: Ready"  # Reset status label
        
    def handle_feedback_thread(self, r, g, b):
        try:
            global mixed_rgb, awaiting_feedback
            mixed_rgb = [r, g, b]
            awaiting_feedback = True

            # Call the original handle_feedback function
            delta_e = handle_feedback()  # Modify handle_feedback to return delta_e

            # Show the Delta E result popup
            if delta_e is not None:
                Clock.schedule_once(lambda dt: self.show_delta_e_result_popup(delta_e), 0)

        except Exception as e:
            print(f"Feedback error: {str(e)}")
        finally:
            Clock.schedule_once(lambda dt: (
                setattr(self.mqtt_client, 'accepting_messages', True),
                self.remove_overlay(None)
            ), 0)
    

    def handle_focus(self, instance, value):
        if value:  # When input box is tapped
            show_virtual_keyboard("numeric")
        else:      # When moving to another field
            hide_virtual_keyboard()

    def on_enter(self):
        """Resume video playback when returning to MainScreen."""
        print("Resuming video playback...")

        # Make sure video_player is available and has a valid source
        if not hasattr(self, 'video_player'):
            print("Video player not initialized yet.")
            return

        if not self.video_player.source:
            print("No video source found. Loading standby video...")
            standby = self.video_sources.get("off")
            if standby and os.path.exists(standby):
                self.video_player.source = standby
            else:
                print("No valid video file found. Skipping playback.")
                return

        print(f"Now playing: {self.video_player.source}")
        self.video_player.state = 'play'

    def on_leave(self):
        """Pause video playback when leaving MainScreen."""
        print("Pausing video playback...")
        if self.video_player.state == 'play':
            self.video_player.state = 'pause'


    def on_stop(self):
        """Ensure video pauses properly when the app is closing."""
        print("Pausing video playback instead of stopping...")
        self.video_player.state = 'pause'  #Prevents complete stop


class WiFiScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)

        # Label for status messages
        self.status_label = Label(text="Wi-Fi Status", size_hint_y=None, height=40, font_size='14sp', halign="center", valign="middle")
        self.status_label.bind(size=self.status_label.setter('text_size'))
        layout.add_widget(self.status_label)

        # Scrollable area for Wi-Fi list
        self.scroll_view = ScrollView(size_hint=(1, None), size=(Window.width, 200))
        self.networks_box = BoxLayout(orientation="vertical", size_hint_y=None, width=Window.width)
        self.networks_box.bind(minimum_height=self.networks_box.setter("height"))
        self.scroll_view.add_widget(self.networks_box)
        layout.add_widget(self.scroll_view)

        # Connect area
        self.connect_layout = BoxLayout(size_hint_y=None, height=40)

        self.ssid_input = TextInput(hint_text="SSID", multiline=False, font_size='12sp')
        self.connect_layout.add_widget(self.ssid_input)

        # Password input
        self.password_input = TextInput(hint_text="Password", multiline=False, password=True, font_size='12sp')
        self.password_input.bind(focus=lambda instance, value: show_virtual_keyboard("full") if value else hide_virtual_keyboard())
        self.connect_layout.add_widget(self.password_input)
        
        # Create a container for the button and image
        password_container = RelativeLayout(size_hint=(None, None), size=(50, 40))

        # The Button (acts as the clickable area)
        self.show_password_button = Button(
            background_normal="",  # Remove default button background
            background_color=(0.8, 0.8, 0.8, 1),  # Set gray background
            size_hint=(None, None),
            size=(50, 40),
            pos=(0, 0)
        )

        # Eye Icon Image (Properly Positioned)
        self.eye_icon = Image(
            source="eye_closed.png",
            size_hint=(None, None),
            size=(30, 30),
            pos=(10, 5)  # Position inside the button
        )

        # Add elements in correct order
        password_container.add_widget(self.show_password_button)
        password_container.add_widget(self.eye_icon)  # Make sure image is on top

        self.show_password_button.bind(on_press=self.toggle_password_visibility)
        self.connect_layout.add_widget(password_container)

        # Connect button
        self.connect_button = Button(text="Connect", size_hint_x=0.3, font_size='12sp')
        self.connect_button.bind(on_press=self.connect_to_wifi)
        self.connect_layout.add_widget(self.connect_button)
        layout.add_widget(self.connect_layout)

        # Buttons
        self.scan_button = Button(text="Scan for Wi-Fi", size_hint=(1, None), height=40, font_size='14sp')
        self.scan_button.bind(on_press=self.scan_wifi)
        layout.add_widget(self.scan_button)

        ok_button = Button(text="OK", size_hint=(1, None), height=40, font_size='14sp')
        ok_button.bind(on_press=self.go_back)
        layout.add_widget(ok_button)

        self.add_widget(layout)


    def scan_wifi(self, instance):
        """Scan for nearby Wi-Fi networks on both Windows & Raspberry Pi."""
        try:
            self.status_label.text = "Scanning for Wi-Fi networks..."
            self.networks_box.clear_widgets()
            system = platform.system()

            if system == "Windows":
                # Use netsh on Windows
                result = subprocess.run(["netsh", "wlan", "show", "network", "mode=bssid"], capture_output=True, text=True, shell=True)
                output = result.stdout
                networks = []
                for line in output.split("\n"):
                    line = line.strip()
                    if line.startswith("SSID") and ":" in line:
                        ssid = line.split(":", 1)[1].strip()
                        if ssid:
                            networks.append(ssid)

            elif system == "Linux":
                # Use nmcli on Raspberry Pi
                result = subprocess.run(["nmcli", "-t", "-f", "SSID", "dev", "wifi"], capture_output=True, text=True, shell=True)
                output = result.stdout.strip()
                networks = output.split("\n") if output else []

            else:
                networks = []
                self.status_label.text = "Unsupported OS"
                return

            if networks:
                self.status_label.text = "Networks found."
                for network in networks:
                    button = Button(text=network, size_hint_y=None, height=30, font_size='12sp')
                    button.bind(on_press=self.fill_ssid)
                    self.networks_box.add_widget(button)
            else:
                self.status_label.text = "No networks found."

        except Exception as e:
            self.status_label.text = f"Error: {e}"
    def fill_ssid(self, instance):
        """Fill the SSID input with the selected network."""
        self.ssid_input.text = instance.text

    def toggle_password_visibility(self, instance):
        """Toggle password visibility and update the eye icon."""
        if self.password_input.password:
            self.password_input.password = False  # Show password
            self.eye_icon.source = "eye_open.png"  # Open eye icon
        
        else:
            self.password_input.password = True  # Hide password
            self.eye_icon.source = "eye_closed.png"  # Closed eye icon

    def connect_to_wifi(self, instance):
        """Connect to a Wi-Fi network on both Windows & Raspberry Pi."""
        ssid = self.ssid_input.text.strip()
        password = self.password_input.text.strip()

        if not ssid:
            self.status_label.text = "Please enter an SSID."
            return

        try:
            self.status_label.text = f"Connecting to {ssid}..."
            system = platform.system()

            if system == "Windows":
                command = ["netsh", "wlan", "connect", f"name={ssid}"]
                if password:
                    profile_content = f"""<?xml version=\"1.0\"?>
<WLANProfile xmlns=\"http://www.microsoft.com/networking/WLAN/profile/v1\">
    <name>{ssid}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>manual</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{password}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>
"""
                profile_path = f"{ssid}.xml"
                with open(profile_path, "w") as profile_file:
                    profile_file.write(profile_content)
                subprocess.run(["netsh", "wlan", "add", "profile", f"filename={profile_path}"], shell=True)
                os.remove(profile_path)

            elif system == "Linux":
                # Use nmcli on Raspberry Pi
                command = ["nmcli", "dev", "wifi", "connect", ssid]
                if password:
                    command += ["password", password]

            else:
                self.status_label.text = "Unsupported OS"
                return

            result = subprocess.run(command, capture_output=True, text=True, shell=True)

            if result.returncode == 0:
                self.status_label.text = f"Connected to {ssid}."
            else:
                self.status_label.text = f"Failed to connect: {result.stderr.strip()}"

        except Exception as e:
            self.status_label.text = f"Error: {e}"

    def go_back(self, instance):
        """Navigate back to the main screen."""
        self.manager.current = 'config'

class ConfigScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        # Background Image
        self.background = Image(
            source='PaintBG.jpg',
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
        )
        layout.add_widget(self.background)

        # WiFi Status Label
        self.wifi_label = Label(
            text="Checking WiFi...",
            font_size=20,
            color=(0, 0, 0, 1),
            size_hint=(None, None),  
            size=(150, 40),  
            pos_hint={'x': 0.01, 'y': 0.95},
            halign="left" 
        )
        layout.add_widget(self.wifi_label)

        # WiFi Icon
        self.wifi_image = Image(
            source='WIFIRED.png',
            size_hint=(None, None),
            size=(40, 40),
            pos_hint={'x': 0.17, 'top': 0.96}
        )
        layout.add_widget(self.wifi_image)

        # SSID Label
        self.ssid_label = Label(
            text="SSID: N/A",
            font_size=20,
            color=(0, 0, 0, 1),
            size_hint=(None, None),  
            size=(150, 40),  
            pos_hint={'x': 0.005, 'y': 0.90},
            halign="left" 
        )
        layout.add_widget(self.ssid_label)

        # Change Network Button
        self.change_network_button = Button(
            text="Change Network", 
            font_size=14,
            size_hint=(None, None),  
            size=(150, 40),  
            pos_hint={'x': 0.005, 'y': 0.85},
            halign="left" 
        )
        self.change_network_button.bind(on_press=self.change_network)
        layout.add_widget(self.change_network_button)

        # Arduino Status Label
        self.arduino_status = Label(
            text="Searching for Arduino...",
            font_size=18,
            color=(0, 0, 0, 1),
            size_hint=(None, None),
            size=(250, 40),
            pos_hint={'right': 1, 'top': 1}
        )
        layout.add_widget(self.arduino_status)

        # MQTT Status Label
        self.mqtt_status = Label(	
            text="Checking MQTT...",
            font_size=18,
            color=(0, 0, 0, 1),
            size_hint=(None, None),
            size=(250, 40),
            pos_hint={'right': 1, 'top': 0.94}  # Slightly below Arduino status
        )
        layout.add_widget(self.mqtt_status)
        
        # Inside ConfigScreen class
        self.dispense_button = Button(
            text="Dispense Paint",
            font_size=16,
            size_hint=(None, None),
            size=(150, 40),
            pos_hint={'center_x': 0.5, 'y': 0.5},  # Position it in the center
            background_color=(0, 1, 0, 1)  # Green color
        )
        self.dispense_button.bind(on_press=self.send_dispense_command)
        layout.add_widget(self.dispense_button)

        # Back Button
        back_button = Button(
            text="Back",
            font_size=16,
            size_hint=(None, None),
            size=(120, 40),
            pos_hint={'center_x': 0.5, 'bottom': 1},
            background_color=(1, 0, 0, 1)
        )
        back_button.bind(on_press=self.go_back)
        layout.add_widget(back_button)

        self.add_widget(layout)

        # Variables to store scheduled events
        self.wifi_event = None
        self.arduino_event = None

    def update_mqtt_status(self, text):
        """Safely updates the MQTT status label."""
        if hasattr(self, 'mqtt_status') and self.mqtt_status:
            Clock.schedule_once(lambda dt: setattr(self.mqtt_status, 'text', text), 0)


    def on_enter(self):
        """Update the connection status immediately when entering the Config Panel."""
        print("Entering Config Panel. Updating connection statuses...")

        # Immediately check WiFi and update UI
        if hasattr(self, 'check_wifi_status'):
            self.check_wifi_status(0)

        if hasattr(self, 'check_arduino_status'):
            self.check_arduino_status(0)  # Correct function

        # Start periodic checks
        if self.wifi_event is None:
            self.wifi_event = Clock.schedule_interval(self.check_wifi_status, 5.0)
        if self.arduino_event is None:
            self.arduino_event = Clock.schedule_interval(self.check_arduino_status, 5.0)

    def on_leave(self):
        """Stop updates when leaving the screen."""
        self.stop_monitoring()

    def change_network(self, instance):
        """Go to WiFi settings screen."""
        self.manager.current = 'wifi'

    def go_back(self, instance):
        """Return to the main screen."""
        self.manager.current = 'main'
        
    def send_dispense_command(self, instance):
        amount_ml = 5  # Example amount in ml for dispensing

        # Send command to Arduino
        send_command("DISPENSE", amount_ml)
        print(f"Dispense command sent to Arduino: {amount_ml} ml")

    def update_arduino_status(self, text):
        """Safely updates the Arduino status label without crashing."""
        if hasattr(self, 'arduino_status') and self.arduino_status:
            Clock.schedule_once(lambda dt: setattr(self.arduino_status, 'text', text), 0)
        else:
            print(f"Warning: Tried to update Arduino status to '{text}', but `arduino_status` does not exist!")

    def update_wifi_ui(self, status_text, image_path, ssid):
        """Update WiFi status in the UI."""
        self.wifi_label.text = status_text
        self.wifi_image.source = image_path
        self.ssid_label.text = f"SSID: {ssid}"

    def check_wifi_status(self, dt):
        """Check WiFi status every few seconds."""
        is_connected = check_wifi_connection()
        ssid = get_ssid() if is_connected else "N/A"
        self.update_wifi_ui(f"WiFi: {'Connected' if is_connected else 'Not Connected'}", 
                            'WIFIGREEN.png' if is_connected else 'WIFIRED.png', ssid)  # Update UI immediately

    def check_arduino_status(self, dt):
        """Check Arduino status every few seconds."""
        detected_port = find_arduino()
        status_text = f"Connected to {detected_port}" if detected_port else "Searching for Arduino..."
        self.update_arduino_status(status_text)

    def start_monitoring(self):
        """Ensure WiFi & Arduino monitoring starts only if not already scheduled."""
        if self.wifi_event is None and not Clock.get_events(self.check_wifi_status):
            self.wifi_event = Clock.schedule_interval(self.check_wifi_status, 5.0)

        if self.arduino_event is None and not Clock.get_events(self.check_arduino_status):
            self.arduino_event = Clock.schedule_interval(self.check_arduino_status, 5.0)

    def stop_monitoring(self):
        """Stop checking WiFi & Arduino status when leaving the screen."""
        if self.wifi_event:
            Clock.unschedule(self.wifi_event)
            self.wifi_event = None

        if self.arduino_event:
            Clock.unschedule(self.arduino_event)
            self.arduino_event = None

class TitleBar(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (1, None)
        self.height = 20
        self.orientation = 'horizontal'
        self.padding = [0, 0, 10, 0]
        self.spacing = 5
        self.pos_hint = {'top': 1}

        self.add_buttons()

    def add_buttons(self):
        self.add_widget(BoxLayout())  # Push buttons to the right

        minimize_btn = Button(text='_', size_hint=(None, 1), width=40)
        exit_btn = Button(text='X', size_hint=(None, 1), width=40)

        minimize_btn.bind(on_release=self.minimize_window)
        exit_btn.bind(on_release=self.exit_app)

        self.add_widget(minimize_btn)
        self.add_widget(exit_btn)

    def minimize_window(self, *args):
        Window.minimize()

    def exit_app(self, *args):
        app = App.get_running_app()
        if hasattr(app, 'sm'):
            main_screen = app.sm.get_screen('main')
            if hasattr(main_screen, 'mqtt_client'):
                main_screen.mqtt_client.stop_monitoring()
        app.stop()
        sys.exit(0)

class SwapTest(App):
    def build(self):
        self.title = "Smart Color Mixer"
        
        # Create screen manager and screens
        self.sm = ScreenManager()
        
        # Initialize the MQTT client first
        self.mqtt_client = initialize_mqtt()  # Get the client instance
        if not self.mqtt_client:
            print("Failed to initialize MQTT client.")
            return  # Exit the function if initialization fails

        # Add MainScreen with mqtt_client
        self.sm.add_widget(MainScreen(mqtt_client=self.mqtt_client, name='main'))
        
        # Add ConfigScreen and WiFiScreen
        self.config_screen = ConfigScreen(name='config')  # Initialize config_screen here
        self.sm.add_widget(self.config_screen)  # Add it to the ScreenManager
        self.sm.add_widget(WiFiScreen(name='wifi'))
        
        # Build layout hierarchy
        root_layout = BoxLayout(orientation='vertical')
        root_layout.add_widget(TitleBar())
        root_layout.add_widget(self.sm)
        
        return root_layout

    def on_start(self):
        """Runs when the app starts and initializes all connections asynchronously."""
        print("App Started. Searching for WiFi, MQTT, and Arduino...")

        # Initialize MQTT first
        self.mqtt_client = initialize_mqtt()  # Get the client instance
        if not self.mqtt_client:
            print("Failed to initialize MQTT client.")
            return  # Exit the function if initialization fails

        # Immediately check WiFi status
        if hasattr(self, 'config_screen'):
            self.config_screen.check_wifi_status(0)

        # Start Arduino Monitoring
        start_arduino_monitoring(self.root)

        # Load model immediately
        ensure_model_loaded()

        # Delay other initializations
        Clock.schedule_once(self.initialize_connections, 0.5)
        Clock.schedule_once(lambda dt: load_model_in_background(), 1.0)

        # Start listening to Arduino in a separate thread
        threading.Thread(target=listen_to_arduino, args=(self,), daemon=True).start()

    def initialize_connections(self, dt):
        """Finalize all connection initializations after GUI loads"""
        print("Finalizing connection initializations...")
        
        # 1. Verify MQTT connection
        if hasattr(self, 'mqtt_client') and self.mqtt_client:
            if not self.mqtt_client.is_connected():
                print("MQTT not connected, attempting to reconnect...")
                self.mqtt_client.reconnect()
        else:
            print("Warning: MQTT client not properly initialized")

        # 2. Verify Arduino connection
        main_screen = self.sm.get_screen('main')
        if not (arduino and arduino.is_open):
            print("Arduino not connected, attempting to reconnect...")
            connect_arduino(main_screen)

        # 3. Update status displays
        if hasattr(self, 'config_screen'):
            self.config_screen.check_wifi_status(0)
            self.config_screen.check_arduino_status(0)

        print("All connections initialized")

    def on_pause(self):
        """This runs when the app loses focus (e.g., when switching to another app)."""
        print("App Paused. Stopping background updates...")
        self.pause_updates()
        return True  # Allows the app to be paused without being killed

    def on_resume(self):
        print("App Resumed. Restarting background updates...")

        if hasattr(self, 'sm'):
            main_screen = self.sm.get_screen('main')
            if hasattr(main_screen, 'bg_video'):
                main_screen.bg_video.state = 'play'
                print("Resumed video playback.")

        self.resume_updates()

    def pause_updates(self):
        """Pause all periodic updates to prevent UI freeze when app is not in focus."""
        if hasattr(self, 'config_screen') and self.config_screen:
            self.config_screen.stop_monitoring()

    def resume_updates(self):
        """Resume periodic updates when app is back in focus."""
        if hasattr(self, 'config_screen') and self.config_screen:
            self.config_screen.start_monitoring()

    def on_stop(self):
        """Ensures all background processes stop properly when closing the app."""
        print("Stopping all background processes...")

        # Close Serial Connection (if open)
        global arduino
        if 'arduino' in globals() and arduino and arduino.is_open:
            print("Closing Serial Port...")
            arduino.close()

        # Correct MQTT Disconnection
        if hasattr(self, 'mqtt_client') and self.mqtt_client is not None:
            try:
                if hasattr(self.mqtt_client, 'is_connected') and self.mqtt_client.is_connected():
                    print("Disconnecting MQTT...")
                    self.mqtt_client.disconnect()
                    self.mqtt_client.loop_stop()
            except Exception as e:
                print(f"Error stopping MQTT: {e}")

        print("All processes stopped. Exiting safely...")

    def on_request_close(self, *args):
        print("App is closing. Cleaning up...")

        # Stop MQTT safely (corrected)
        if hasattr(self, 'mqtt_client'):
            try:
                if self.mqtt_client.is_connected():
                    print("Disconnecting MQTT...")
                    self.mqtt_client.disconnect()
                    self.mqtt_client.loop_stop()
            except Exception as e:
                print(f"Error stopping MQTT: {e}")

        # Stop video playback (if present)
        try:
            main_screen = self.sm.get_screen('main')
            if hasattr(main_screen, 'bg_video'):
                main_screen.bg_video.state = 'stop'
                print("Video playback stopped.")
        except Exception as e:
            print(f"Error stopping video: {e}")

        # Close Arduino connection if open
        if 'arduino' in globals():
            try:
                if arduino and arduino.is_open:
                    arduino.close()
                    print("Arduino connection closed.")
            except Exception as e:
                print(f"Error closing Arduino: {e}")

        self.stop()
        sys.exit(0)
        return True

if __name__ == '__main__':
    gui_instance = SwapTest()  # Initialize the GUI instance

    # Start GUI first, then handle Arduino after it's loaded
    Clock.schedule_once(lambda dt: connect_arduino(gui_instance), 0.5)  

    gui_instance.run()  # Run the GUI
