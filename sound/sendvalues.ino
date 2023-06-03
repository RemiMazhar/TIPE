int n = 0;
unsigned long t0;
unsigned long t1;
int val;

void setup() {
  Serial.begin(115200);
  t0 = micros();
  t1 = micros();
}


void loop() {
    val = analogRead(0);

    // pour envoyer au programme python (~6000Hz):
    Serial.write(lowByte(val));
    Serial.write(highByte(val));    

    // pour envoyer au serial plotter (~2000Hz):
    //Serial.println(val);

    // pour mesurer le temps d'exécution de trucs:
    /*
    n++;
    if (n == 1000)
    {
      n = 0;
      t1 = micros();
      Serial.println();
      Serial.println(1e9 / (t1 - t0));
      delay(3000);
      t0 = micros();
    }
    */
}
