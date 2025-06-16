#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Adafruit_TCS34725.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#include <Preferences.h>
#define BUTTON_PIN 32
#define LED_CONTROL_PIN 4

void applyColorCorrection(float &r, float &g, float &b) {
    float luminance = (r + g + b) / 3.0 / 255.0;
    float matrix[3][3] = {
        {1.18, -0.32,  0.10},
        {-0.25, 1.42, -0.20},
        {0.05, -0.15,  0.85}
    };
    
    float correctedR = matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b;
    float correctedG = matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b;
    float correctedB = matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b;
    
    // Clamp to 0–1
    r = constrain(correctedR, 0.0f, 1.0f);
    g = constrain(correctedG, 0.0f, 1.0f);
    b = constrain(correctedB, 0.0f, 1.0f);
}



const char* ssid = "AEIOU";
const char* password = "passnghotspot";
const char* mqttServer = "192.168.0.174";
const int mqttPort = 1883;
const char* mqttTopic = "sensor/rgb";

WiFiClient espClient;
PubSubClient mqttClient(espClient);



Preferences preferences;
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_50MS, TCS34725_GAIN_1X);//dito calibrate
TFT_eSPI tft = TFT_eSPI();

unsigned long buttonPressTime = 0;
bool isCalibrating = false;
bool whiteCalibrated = false, blackCalibrated = false;
uint16_t whiteR, whiteG, whiteB;
uint16_t blackR, blackG, blackB;
uint16_t prevR = 0, prevG = 0, prevB = 0;

void getStableColor(uint16_t &r, uint16_t &g, uint16_t &b) {
    const int samples = 10;  // Take 10 samples for stability
    uint16_t rArr[samples], gArr[samples], bArr[samples], cArr[samples];

    // Collect multiple samples
    for (int i = 0; i < samples; i++) {
        tcs.getRawData(&rArr[i], &gArr[i], &bArr[i], &cArr[i]);
        Serial.printf("Sample %d - Raw: R=%d, G=%d, B=%d, C=%d\n", i, rArr[i], gArr[i], bArr[i], cArr[i]);
        delay(20);  // Small delay to allow stable readings
    }

    // Sort arrays for median filtering
    std::sort(rArr, rArr + samples);
    std::sort(gArr, gArr + samples);
    std::sort(bArr, bArr + samples);
    std::sort(cArr, cArr + samples);

    r = rArr[samples/2];  // NOT average
    g = gArr[samples/2];
    b = bArr[samples/2];
    uint16_t c = cArr[samples / 2];

    // Debugging before normalization
    Serial.printf("Median Raw: R=%d, G=%d, B=%d, C=%d\n", r, g, b, c);

    // Debugging after normalization
    Serial.printf("Normalized RGB: R=%d, G=%d, B=%d\n", r, g, b);
}


void connectToWiFi() {
    WiFi.begin(ssid, password);
    tft.fillScreen(TFT_BLUE);
    tft.setCursor(10, 10);
    tft.setTextColor(TFT_WHITE);
    tft.setTextSize(2);
    tft.print("Connecting to WiFi...");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    tft.fillScreen(TFT_GREEN);
    tft.setCursor(10, 10);
    tft.print("WiFi Connected!");
    delay(2000);
}

void connectToMQTT() {
    while (!mqttClient.connected()) {
        Serial.print("Connecting to MQTT...");
        if (mqttClient.connect("ESP32Client")) {
            Serial.println("Connected!");
            return;  // Exit function on success
        } else {
            Serial.print("Failed, rc=");
            Serial.print(mqttClient.state());
            Serial.println(" Retrying in 3s...");
            delay(3000);
        }
    }
}



void saveCalibration() {
    preferences.begin("calibration", false);
    preferences.putBool("calibrated", true);
    preferences.putInt("whiteR", whiteR);
    preferences.putInt("whiteG", whiteG);
    preferences.putInt("whiteB", whiteB);
    preferences.putInt("blackR", blackR);
    preferences.putInt("blackG", blackG);
    preferences.putInt("blackB", blackB);
    preferences.putBool("whiteCalibrated", true);
    preferences.putBool("blackCalibrated", true);
    preferences.end();
    Serial.println("Calibration Saved!");
}

void loadCalibration() {
    preferences.begin("calibration", true);
    whiteR = preferences.getInt("whiteR", 255);
    whiteG = preferences.getInt("whiteG", 255);
    whiteB = preferences.getInt("whiteB", 255);
    blackR = preferences.getInt("blackR", 0);
    blackG = preferences.getInt("blackG", 0);
    blackB = preferences.getInt("blackB", 0);
    whiteCalibrated = preferences.getBool("whiteCalibrated", false);
    blackCalibrated = preferences.getBool("blackCalibrated", false);
    preferences.end();
    Serial.println("Loaded Calibration:");
    Serial.printf("White RGB: %d, %d, %d\n", whiteR, whiteG, whiteB);
    Serial.printf("Black RGB: %d, %d, %d\n", blackR, blackG, blackB);
    Serial.printf("White Calibrated: %s\n", whiteCalibrated ? "YES" : "NO");
    Serial.printf("Black Calibrated: %s\n", blackCalibrated ? "YES" : "NO");
    
}



void setup() {
    Serial.begin(115200);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_CONTROL_PIN, OUTPUT);
    analogWrite(LED_CONTROL_PIN, 0);  // Ensure LED starts off

    tft.init();
    tft.setRotation(0);

    int retries = 5;
    while (!tcs.begin() && retries > 0) {
        Serial.println("TCS34725 not found, retrying...");
        delay(1000);
        retries--;
    }

    if (retries == 0) {
        Serial.println("TCS34725 sensor failed! Restarting ESP...");
        ESP.restart();  // Restart ESP32
    }

    
    connectToWiFi();
    mqttClient.setServer(mqttServer, mqttPort);
    connectToMQTT();
    loadCalibration();
}



void loop() {
    if (!mqttClient.connected()) connectToMQTT();
    mqttClient.loop();

    checkButtonPress();
}

void checkButtonPress() {
    static bool lastButtonState = HIGH;
    bool currentButtonState = digitalRead(BUTTON_PIN);

    if (currentButtonState == LOW && lastButtonState == HIGH) {
        buttonPressTime = millis();
    }

    if (currentButtonState == HIGH && lastButtonState == LOW) {
        unsigned long pressDuration = millis() - buttonPressTime;

        if (pressDuration > 2000) {
            enterCalibrationMode();
        } else {
            captureAndSendRGB();
        }
    }

    lastButtonState = currentButtonState;
}

void enterCalibrationMode() {
    isCalibrating = true;
    analogWrite(LED_CONTROL_PIN, 100);  // Turn on LED for calibration
    tft.fillScreen(TFT_YELLOW);
    tft.setRotation(0);
    tft.setTextColor(TFT_BLACK, TFT_YELLOW);
    tft.setTextSize(1);
    tft.setCursor(10, 10);
    tft.println("Calibration Mode");

    // White Calibration
    tft.setCursor(10, 30);
    tft.println("Place WHITE and press button");
    while (digitalRead(BUTTON_PIN) == HIGH);
    delay(50);
    while (digitalRead(BUTTON_PIN) == LOW);

    Serial.println("Capturing White...");
    uint16_t c;
    getStableColor(whiteR, whiteG, whiteB);
    Serial.printf("White RGB: %d, %d, %d\n", whiteR, whiteG, whiteB);
    whiteCalibrated = true; // Mark white as calibrated

    tft.fillScreen(TFT_GREEN);
    tft.setCursor(10, 10);
    tft.println("White Captured!");
    delay(2000);

    // Black Calibration
    tft.fillScreen(TFT_YELLOW);
    tft.setCursor(10, 10);
    tft.println("Place BLACK and press button");
    while (digitalRead(BUTTON_PIN) == HIGH);
    delay(50);
    while (digitalRead(BUTTON_PIN) == LOW);

    Serial.println("Capturing Black...");
    getStableColor(blackR, blackG, blackB); // Median-filtered samples
    Serial.printf("Black RGB: %d, %d, %d\n", blackR, blackG, blackB);
    blackCalibrated = true; // Mark black as calibrated

    tft.fillScreen(TFT_GREEN);
    tft.setCursor(10, 10);
    tft.println("Black Captured!");
    delay(2000);

    saveCalibration();  // Save both white and black calibration
    isCalibrating = false;

    tft.fillScreen(TFT_BLUE);
    tft.setCursor(10, 10);
    tft.println("Calibration Done!");
    delay(2000);
}

#define GAMMA 2.2
float applyGamma(float normalizedValue) { // Input should be 0.0-1.0
    return pow(normalizedValue, 1.0/GAMMA) * 255.0; // Apply gamma to normalized value
}

void captureAndSendRGB() {
    uint16_t r, g, b;  
    
    // Turn on LED FIRST for stable readings
    analogWrite(LED_CONTROL_PIN, 100);  
    delay(300);  // Let LED stabilize
    
    Serial.println("🟡 Waiting for sensor stabilization...");
    getStableColor(r, g, b);  
    Serial.printf("RAW_SENSOR_DATA: R=%d, G=%d, B=%d\n", r, g, b); 

    float r_norm = constrain((float)(r - blackR) / (whiteR - blackR), 0.0f, 1.0f);
    float g_norm = constrain((float)(g - blackG) / (whiteG - blackG), 0.0f, 1.0f);
    float b_norm = constrain((float)(b - blackB) / (whiteB - blackB), 0.0f, 1.0f);

    applyColorCorrection(r_norm, g_norm, b_norm); // Now works with floats!

    // Apply gamma correction
    uint8_t r_scaled = applyGamma(r_norm);
    uint8_t g_scaled = applyGamma(g_norm);
    uint8_t b_scaled = applyGamma(b_norm);

     //Publish data
     String rgbData = "{\"r\":" + String(r_scaled) + ",\"g\":" + String(g_scaled) + ",\"b\":" + String(b_scaled) + "}";
     mqttClient.publish(mqttTopic, rgbData.c_str());

    // Display color
    tft.fillScreen(TFT_BLACK);
    tft.fillRect(10, 20, 100, 100, tft.color565(r_scaled, g_scaled, b_scaled));
    tft.setCursor(10, 130);
    tft.setTextSize(1);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.printf("R:%3d G:%3d B:%3d", r_scaled, g_scaled, b_scaled);

    // Turn off LED
    analogWrite(LED_CONTROL_PIN, 0);

    // Store for stability check
    prevR = r_scaled;
    prevG = g_scaled;
    prevB = b_scaled;
}

