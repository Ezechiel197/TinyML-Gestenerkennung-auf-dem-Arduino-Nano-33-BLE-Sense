// Gesture classifier (punch / lift) on the Arduino Nano 33 BLE Sense.
// A motion spike starts a 119-sample IMU window, which is normalised with the
// notebook's mean/std, classified by the TFLite model and shown on the RGB LED:
// punch -> red, lift -> green, uncertain -> blue.

#include <Arduino_LSM9DS1.h>
#include <Chirale_TensorFlowLite.h>
#include "model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kSamples = 119;
constexpr int kChannels = 6;  // ax, ay, az, gx, gy, gz

// Must match the notebook run that produced model.h (cell "Normalisierung").
const float kMean[kChannels] = {-0.026f, -0.025f, 0.978f, 1.760f, 0.888f, 1.212f};
const float kStd[kChannels]  = {0.177f, 0.174f, 0.234f, 27.099f, 26.389f, 28.157f};
// |ax|+|ay|+|az| in g; at rest this is ~1, so a clearly larger sum means the gesture started.
// Must equal the recorder's --threshold, or board windows start later than the training windows.
constexpr float kTriggerAccelSum = 1.3f;
// Model output is P(lift); anything closer to 0.5 than this margin counts as uncertain.
constexpr float kDecisionMargin = 0.15f;
constexpr unsigned long kResultShowMs = 1500;

constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

enum class Gesture { None, Punch, Lift, Uncertain };

// The RGB LED on the Nano 33 BLE is active-low.
void showGesture(Gesture g) {
  digitalWrite(LEDR, g == Gesture::Punch ? LOW : HIGH);
  digitalWrite(LEDG, g == Gesture::Lift ? LOW : HIGH);
  digitalWrite(LEDB, g == Gesture::Uncertain ? LOW : HIGH);
}

void fail(const char* msg) {
  Serial.println(msg);
  while (true) {
    showGesture(Gesture::Punch);
    delay(100);
    showGesture(Gesture::None);
    delay(100);
  }
}

// The model may take float or int8 tensors depending on how it was converted.
void setInput(int i, float v) {
  if (input->type == kTfLiteInt8) {
    long q = lroundf(v / input->params.scale) + input->params.zero_point;
    input->data.int8[i] = (int8_t)constrain(q, -128L, 127L);
  } else {
    input->data.f[i] = v;
  }
}

float getOutput() {
  if (output->type == kTfLiteInt8) {
    return (output->data.int8[0] - output->params.zero_point) * output->params.scale;
  }
  return output->data.f[0];
}

void readImu(float s[kChannels]) {
  while (!(IMU.accelerationAvailable() && IMU.gyroscopeAvailable())) {}
  IMU.readAcceleration(s[0], s[1], s[2]);
  IMU.readGyroscope(s[3], s[4], s[5]);
}

void setup() {
  Serial.begin(9600);
  // Wait briefly for a Serial Monitor, but also run standalone.
  while (!Serial && millis() < 3000) {}

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  showGesture(Gesture::None);

  if (!IMU.begin()) fail("IMU init failed");

  const tflite::Model* tfModel = tflite::GetModel(model);
  if (tfModel->version() != TFLITE_SCHEMA_VERSION) fail("Model schema version mismatch");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterpreter(tfModel, resolver, tensorArena, kTensorArenaSize);
  interpreter = &staticInterpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) fail("AllocateTensors failed");

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->bytes / (input->type == kTfLiteInt8 ? 1 : sizeof(float)) != kSamples * kChannels) {
    fail("Unexpected input tensor size");
  }

  Serial.print("Ready, arena used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" bytes");
}

void loop() {
  float s[kChannels];

  readImu(s);
  if (fabsf(s[0]) + fabsf(s[1]) + fabsf(s[2]) < kTriggerAccelSum) return;

  // The triggering sample is the first one of the window.
  digitalWrite(LED_BUILTIN, HIGH);
  for (int t = 0; t < kSamples; t++) {
    if (t > 0) readImu(s);
    for (int c = 0; c < kChannels; c++) {
      setInput(t * kChannels + c, (s[c] - kMean[c]) / kStd[c]);
    }
  }
  digitalWrite(LED_BUILTIN, LOW);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  float pLift = getOutput();
  Gesture g = pLift >= 0.5f + kDecisionMargin   ? Gesture::Lift
              : pLift <= 0.5f - kDecisionMargin ? Gesture::Punch
                                                : Gesture::Uncertain;

  Serial.print("P(lift) = ");
  Serial.print(pLift, 3);
  Serial.print(" -> ");
  Serial.println(g == Gesture::Lift ? "lift" : g == Gesture::Punch ? "punch" : "uncertain");

  showGesture(g);
  delay(kResultShowMs);
  showGesture(Gesture::None);

  // Drop samples buffered during the pause so the next trigger sees fresh motion.
  while (IMU.accelerationAvailable()) readImu(s);
}
