#include <OneWire.h>
#include <DallasTemperature.h>
#include <DHT.h>

// Pin assignments
#define PIN_MQ135 A0
#define PIN_MQ3   A1
#define PIN_DS18B20 2
#define PIN_DHT22 3
#define DHTTYPE DHT22

OneWire oneWire(PIN_DS18B20);
DallasTemperature ds18b20(&oneWire);
DHT dht(PIN_DHT22, DHTTYPE);

void setup() {
  Serial.begin(9600);
  ds18b20.begin();
  dht.begin();

  // Print CSV Header
  Serial.println("time_ms,mq135,mq3,ds18b20_temp_c,dht_temp_c,humidity_pct");
}

void loop() {
  unsigned long timeMs = millis();

  // Read gas sensors (Analog)
  int mq135Val = analogRead(PIN_MQ135);
  int mq3Val   = analogRead(PIN_MQ3);

  // Read temperature probe (Digital 1-Wire)
  ds18b20.requestTemperatures();
  float dsTemp = ds18b20.getTempCByIndex(0);

  // Read ambient temp and humidity (Digital)
  float dhtTemp  = dht.readTemperature();
  float dhtHumid = dht.readHumidity();

  // CSV output: time_ms,mq135,mq3,ds18b20,dht_temp,humidity
  Serial.print(timeMs);
  Serial.print(",");
  Serial.print(mq135Val);
  Serial.print(",");
  Serial.print(mq3Val);
  Serial.print(",");
  
  if (dsTemp == DEVICE_DISCONNECTED_C) {
    Serial.print("ERR");
  } else {
    Serial.print(dsTemp, 2);
  }
  Serial.print(",");

  if (isnan(dhtTemp)) {
    Serial.print("ERR");
  } else {
    Serial.print(dhtTemp, 2);
  }
  Serial.print(",");

  if (isnan(dhtHumid)) {
    Serial.println("ERR");
  } else {
    Serial.println(dhtHumid, 2);
  }

  delay(2000); // 2-second sampling interval
}