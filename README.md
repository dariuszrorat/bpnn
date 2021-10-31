# Arduino backpropagation neural network library

C++ port of
http://arctrix.com/nas/python/bpnn.py

### Example

XOR problem

```
/*
 * Neural Network XOR problem
 */
#include <bpnn.h>

float ** patterns;
float ** targets;

float* input = new float[2];
float* output;
int y;

BPNN nn = BPNN(2, 4, 1);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  randomSeed(analogRead(3));

  patterns = new float* [4];
  targets = new float* [4];

  for (int i = 0; i < 4; i++)
  {
      patterns[i] = new float[2];
      targets[i] = new float[1];
  }

  patterns[0][0] = 0;
  patterns[0][1] = 0;
  targets[0][0] = 0;

  patterns[1][0] = 0;
  patterns[1][1] = 1;
  targets[1][0] = 1;

  patterns[2][0] = 1;
  patterns[2][1] = 0;
  targets[2][0] = 1;

  patterns[3][0] = 1;
  patterns[3][1] = 1;
  targets[3][0] = 0;

  Serial.println("NETWORK TRAINING...");
  nn.train(patterns, targets, 4, 1000, 0.001, 0.5, 0.1);

  float* out;
  Serial.println("NETWORK TRAINING ENDED.");
  out = nn.compute(patterns[0]);
  Serial.println(out[0]);
  out = nn.compute(patterns[1]);
  Serial.println(out[0]);
  out = nn.compute(patterns[2]);
  Serial.println(out[0]);
  out = nn.compute(patterns[3]);
  Serial.println(out[0]);

  Serial.println("SAVE WEIGHTS.");
  nn.save();
  pinMode(13, OUTPUT);

}

void loop() {

    input[0] = analogRead(A0) / 1023.0;
    input[1] = analogRead(A1) / 1023.0;
    output = nn.compute(input);

    y = output[0] > 0.8 ? HIGH : LOW;
    digitalWrite(13, y);
}
```