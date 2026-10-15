/*
 * NeuroPath AI — ESP32 Firmware
 * ------------------------------------------------------------
 * Streams MPU6050 (accel + gyro) and optionally MAX30102 (PPG)
 * over USB serial as a comma-separated line at a STABLE 50 Hz.
 *
 * Output format (7 columns) when MAX30102 is enabled:
 *   accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ppg_raw
 *
 * Output format (6 columns) when MAX30102 is NOT present:
 *   accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
 *
 * Units:
 *   accel = m/s^2   (Adafruit library default)
 *   gyro  = rad/s   (Adafruit library default)
 *   ppg   = raw IR count (integer)
 *
 * Baud: 115200
 * ------------------------------------------------------------
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ── CONFIG ────────────────────────────────────────────────────
// Set to true ONLY if you have a MAX30102 wired in.
#define USE_MAX30102   true

// Target sample rate. SisFall-compatible processing expects this
// to be CONSTANT. 50 Hz is the spec minimum and is rock-stable on ESP32.
const uint16_t SAMPLE_RATE_HZ = 50;
const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;  // 20000 us
// ───────────────────────────────────────────────────────────────

Adafruit_MPU6050 mpu;

#if USE_MAX30102
  #include "MAX30105.h"          // SparkFun MAX3010x library
  MAX30105 particleSensor;
#endif

uint32_t nextSampleTime = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  Wire.begin(21, 22);
  Wire.setClock(400000);          // 400 kHz I2C for faster reads

  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("ERROR: MPU6050 not found");
    while (1) { delay(10); }
  }

  // Sensor ranges — tune if your movements clip
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

#if USE_MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found");
    while (1) { delay(10); }
  }
  // Explicit config for stable PPG (better than default setup() for HRV).
  // setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange)
  //   ledMode 2 = Red + IR (we read IR)
  //   sampleRate 100 sps internally; we sample it at 50 Hz in the loop
  byte ledBrightness = 0x1F;   // moderate brightness
  byte sampleAverage = 4;
  byte ledMode       = 2;      // Red + IR
  int  sampleRate    = 100;
  int  pulseWidth    = 411;
  int  adcRange      = 4096;
  particleSensor.setup(ledBrightness, sampleAverage, ledMode,
                       sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeIR(0x1F);
#endif

  nextSampleTime = micros();
}

void loop() {
  // ── Precise rate control using micros(), NOT delay() ──────────
  // This guarantees a constant 50 Hz even though sensor reads and
  // serial writes take variable time.
  uint32_t now = micros();
  if ((int32_t)(now - nextSampleTime) < 0) {
    return;  // not time yet
  }
  nextSampleTime += SAMPLE_PERIOD_US;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // ── Accelerometer (m/s^2) ─────────────────────────────────────
  Serial.print(a.acceleration.x, 2); Serial.print(",");
  Serial.print(a.acceleration.y, 2); Serial.print(",");
  Serial.print(a.acceleration.z, 2); Serial.print(",");

  // ── Gyroscope (rad/s) ─────────────────────────────────────────
  Serial.print(g.gyro.x, 2); Serial.print(",");
  Serial.print(g.gyro.y, 2); Serial.print(",");

#if USE_MAX30102
  Serial.print(g.gyro.z, 2); Serial.print(",");
  // ── PPG raw IR value (integer) ──────────────────────────────
  Serial.println(particleSensor.getIR());
#else
  Serial.println(g.gyro.z, 2);   // 6-column mode, ends the row
#endif
}
