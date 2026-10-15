/*
 * NeuroPath AI — ESP32 Firmware v2
 * Sensors : MPU6050 (accel + gyro)
 * MAX30102 : removed (short-circuited) — PPG is synthesised at 72 bpm
 *
 * Serial output (115200 baud, 50 Hz):
 *   accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,ppg_raw
 *   units: m/s²  m/s²  m/s²  rad/s  rad/s  rad/s  (int)
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

#define SDA_PIN   21
#define SCL_PIN   22
#define BAUD      115200
#define RATE_HZ   50              // must match backend SAMPLE_RATE_HZ
#define PERIOD_MS (1000 / RATE_HZ)

// Synthetic PPG: 72 bpm = 1.2 Hz sine wave, amplitude 300, offset 2000
#define PPG_OFFSET   2000.0f
#define PPG_AMP       300.0f
#define PPG_FREQ        1.2f      // Hz  (~72 bpm)

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(BAUD);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);          // 100 kHz — reliable on breadboard

  while (!mpu.begin(0x68, &Wire)) {
    Serial.println("ERR:MPU6050");
    delay(1000);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("READY");
}

void loop() {
  static unsigned long next_ms = 0;
  unsigned long now = millis();

  if (now < next_ms) return;       // busy-wait free pacing
  next_ms = now + PERIOD_MS;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Synthesise PPG (replaces missing MAX30102)
  float t   = now / 1000.0f;
  float ppg = PPG_OFFSET + PPG_AMP * sinf(2.0f * PI * PPG_FREQ * t);

  // CSV line — backend parser splits on ',' and expects exactly 7 fields
  Serial.print(a.acceleration.x, 4); Serial.print(',');
  Serial.print(a.acceleration.y, 4); Serial.print(',');
  Serial.print(a.acceleration.z, 4); Serial.print(',');
  Serial.print(g.gyro.x,         4); Serial.print(',');
  Serial.print(g.gyro.y,         4); Serial.print(',');
  Serial.print(g.gyro.z,         4); Serial.print(',');
  Serial.println((int)ppg);
}
