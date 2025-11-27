// === Includes ===
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <driver/i2s.h>
#include <math.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <SoundMeter-Sources_inferencing.h>

// === Edge Impulse ===
float eiBuffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
uint32_t eiIndex = 0;
String eiLabel = "---";
float eiConf = 0.0f;

// === WiFi / MQTT ===
const char* WIFI_SSID = "";
const char* WIFI_PASS = "";
const char* MQTT_SERVER = "57.129.12.32";
const uint16_t MQTT_PORT = 1883;
const char* MQTT_TOPIC = "soundmeter/values";
const unsigned long PUBLISH_INTERVAL_MS = 1000UL;

// === I2S & OLED ===
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024
Adafruit_SSD1306 display(128, 64, &Wire);

// === Button ===
#define BUTTON_PIN 13
bool isSleeping = false;
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// === RMS reference / Calibration ===
const double RMS_REF_94dB = pow(10.0, -26.0 / 20.0);
const double CALIB_OFFSET = 0.0;
const double NOISE_FLOOR_DB = 29.0;

// === A-weighting filter ===
const int ORDER = 6;
const double b_coeff[ORDER+1] = {
  0.169994948147430, 0.280415310498794, -1.120574766348363,
  0.131562559965936, 0.974153561246036, -0.282740857326553,
 -0.152810756202003
};
const double a_coeff[ORDER+1] = {
  1.00000000000000000, -2.12979364760736134, 0.42996125885751674,
  1.62132698199721426, -0.96669962900852902, 0.00121015844426781,
  0.04400300696788968
};
double xhist[ORDER+1], yhist[ORDER];

// === Smoothing ===
const double dt = (double)BUFFER_SIZE / (double)SAMPLE_RATE;
const double tau_fast = 0.125, tau_slow = 1.0;
const double alpha_fast = exp(-dt / tau_fast);
const double alpha_slow = exp(-dt / tau_slow);
double rms_fast_raw = 0.0, rms_slow_raw = 0.0;
double rms_fast_A   = 0.0, rms_slow_A   = 0.0;

// === MQTT ===
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
unsigned long lastPublishMs = 0;

// === Helper ===
void filter_init() {
  for (int i = 0; i <= ORDER; ++i) xhist[i] = 0.0;
  for (int i = 0; i < ORDER; ++i) yhist[i] = 0.0;
}
float aweight_df1_process(float x) {
  for (int i = ORDER; i > 0; --i) xhist[i] = xhist[i-1];
  xhist[0] = x;
  double yn = 0.0;
  for (int k = 0; k <= ORDER; ++k) yn += b_coeff[k] * xhist[k];
  for (int k = 1; k <= ORDER; ++k) yn -= a_coeff[k] * (k-1 < ORDER ? yhist[k-1] : 0.0);
  for (int i = ORDER-1; i > 0; --i) yhist[i] = yhist[i-1];
  yhist[0] = yn;
  return (float)yn;
}

// === I2S setup ===
void i2s_install() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_pin_config_t pin_config = {
    .bck_io_num = 5, .ws_io_num = 4,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = 15
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

// === Button / Sleep ===
void enterSleepMode() {
  isSleeping = true;
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(20, 20);
  display.println("SLEEP MODE");
  display.setCursor(10, 35);
  display.println("Press to wake");
  display.display();
  i2s_stop(I2S_PORT);
}
void exitSleepMode() {
  isSleeping = false;
  i2s_start(I2S_PORT);
  display.clearDisplay();
  display.setCursor(25, 25);
  display.println("Waking...");
  display.display();
  delay(400);
}
void checkButton() {
  int reading = digitalRead(BUTTON_PIN);
  if (reading == LOW && lastButtonState == HIGH) {
    if (millis() - lastDebounceTime > debounceDelay) {
      isSleeping ? exitSleepMode() : enterSleepMode();
      lastDebounceTime = millis();
    }
  }
  lastButtonState = reading;
}

// === MQTT connect ===
void mqttConnect() {
  if (mqttClient.connected()) return;
  Serial.print("Connecting to MQTT...");
  if (mqttClient.connect("SoundMeterClient")) Serial.println("OK");
  else {
    Serial.printf("failed (%d)\n", mqttClient.state());
    delay(2000);
  }
}

// === setup ===
void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) while (1);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 10);
  display.println("SoundMeter + AI");
  display.display();

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println("\nWiFi connected");
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  i2s_install(); filter_init();
}

// === loop ===
void loop() {
  checkButton();
  if (isSleeping) { delay(100); return; }

  if (!mqttClient.connected()) mqttConnect();
  mqttClient.loop();

  static int32_t buffer[BUFFER_SIZE];
  size_t bytes_read;
  i2s_read(I2S_PORT, buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);
  int samples = bytes_read / sizeof(int32_t);

  double sum_raw = 0.0, sum_A = 0.0, mean = 0.0;
  for (int i = 0; i < samples; ++i) {
    float s = (float)(buffer[i] >> 8) / (float)(1 << 23);
    mean += s;
  }
  mean /= samples;

  for (int i = 0; i < samples; ++i) {
    float s = (float)(buffer[i] >> 8) / (float)(1 << 23);
    s -= mean;
    sum_raw += s * s;
    float sA = aweight_df1_process(s);
    sum_A += sA * sA;
    eiBuffer[eiIndex++] = s;
    if (eiIndex >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
      eiIndex = 0;
      signal_t sig; ei_impulse_result_t res;
      numpy::signal_from_buffer(eiBuffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &sig);
      if (run_classifier(&sig, &res, false) == EI_IMPULSE_OK) {
        float best = 0; String lbl = "---";
        for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
          if (res.classification[i].value > best) {
            best = res.classification[i].value;
            lbl = res.classification[i].label;
          }
        }
        eiLabel = lbl; eiConf = best;
      }
    }
  }

  double rms_raw = sqrt(sum_raw / samples);
  double rms_A = sqrt(sum_A / samples);
  rms_fast_raw = alpha_fast * rms_fast_raw + (1 - alpha_fast) * rms_raw;
  rms_slow_raw = alpha_slow * rms_slow_raw + (1 - alpha_slow) * rms_raw;
  rms_fast_A   = alpha_fast * rms_fast_A + (1 - alpha_fast) * rms_A;
  rms_slow_A   = alpha_slow * rms_slow_A + (1 - alpha_slow) * rms_A;

  auto to_dB = [](double r){ return (r>1e-12)?20*log10(r/RMS_REF_94dB)+94.0+CALIB_OFFSET:-100.0; };
  double dB_fast=to_dB(rms_fast_raw), dB_slow=to_dB(rms_slow_raw),
         dBA_fast=to_dB(rms_fast_A), dBA_slow=to_dB(rms_slow_A);

  if (dB_fast<NOISE_FLOOR_DB) dB_fast=NOISE_FLOOR_DB;
  if (dBA_fast<NOISE_FLOOR_DB) dBA_fast=NOISE_FLOOR_DB;

  // === OLED compact layout ===
  // === OLED COMPACT DISPLAY FIX ===
display.clearDisplay();

// Τίτλος
display.setTextSize(1);
display.setCursor(0, 0);
display.print("SoundMeter + AI");

// FAST/SLOW τιμές (πιο μικρή γραμματοσειρά)
display.setTextSize(1);
display.setCursor(0, 12);
display.printf("FAST: %.1f dB  A:%.1f", dB_fast, dBA_fast);
display.setCursor(0, 24);
display.printf("SLOW: %.1f dB  A:%.1f", dB_slow, dBA_slow);

// Πηγή ήχου (AI)
display.setCursor(0, 38);
display.printf("Src: %s", eiLabel.c_str());

// Εμπιστοσύνη (confidence)
display.setCursor(0, 50);
display.printf("Conf: %.0f%%", eiConf * 100);

display.display();


  // === MQTT publish ===
if (mqttClient.connected()) {
  // === Publish JSON με όλα τα δεδομένα ===
  char payload[512];
  snprintf(payload, sizeof(payload),
           "{"
           "\"dB_fast\":%.2f,"
           "\"dBA_fast\":%.2f,"
           "\"dB_slow\":%.2f,"
           "\"dBA_slow\":%.2f,"
           "\"source\":\"%s\","
           "\"confidence\":%.2f"
           "}",
           dB_fast, dBA_fast, dB_slow, dBA_slow,
           eiLabel.c_str(), eiConf);

  bool ok = mqttClient.publish(MQTT_TOPIC, payload, true);  // retain=true για να μένει η τελευταία
  if (!ok) {
    Serial.println("MQTT publish failed");
  } else {
    Serial.print("Published: ");
    Serial.println(payload);
  }
  mqttClient.loop(); 
}
}
