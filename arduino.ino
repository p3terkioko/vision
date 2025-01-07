#include <SoftwareSerial.h>

// Define Bluetooth RX and TX pins
#define BT_RX 2
#define BT_TX 3

SoftwareSerial Bluetooth(BT_RX, BT_TX); // Create a software serial object

// Define pins for front motors (IC1)
const int leftFrontMotorPin1 = 9;
const int leftFrontMotorPin2 = 10;
const int rightFrontMotorPin1 = 11;
const int rightFrontMotorPin2 = 12;

// Define pins for rear motors (IC2)
const int leftRearMotorPin1 = 5;
const int leftRearMotorPin2 = 6;
const int rightRearMotorPin1 = 7;
const int rightRearMotorPin2 = 8;

void setup() {
  // Set motor control pins as outputs
  pinMode(leftFrontMotorPin1, OUTPUT);
  pinMode(leftFrontMotorPin2, OUTPUT);
  pinMode(rightFrontMotorPin1, OUTPUT);
  pinMode(rightFrontMotorPin2, OUTPUT);

  pinMode(leftRearMotorPin1, OUTPUT);
  pinMode(leftRearMotorPin2, OUTPUT);
  pinMode(rightRearMotorPin1, OUTPUT);
  pinMode(rightRearMotorPin2, OUTPUT);

  Serial.begin(9600);         // USB serial communication (debugging)
  Bluetooth.begin(9600);      // Bluetooth communication
  Serial.println("Ready to receive Bluetooth commands");
}

void loop() {
  if (Bluetooth.available()) {
    char command = Bluetooth.read(); // Read one character from Bluetooth
    Serial.print("Command received: ");
    Serial.println(command); // Debugging output

    switch (command) {
      case 'F': moveForward(); break;   // Forward
      case 'B': moveBackward(); break;  // Backward
      case 'L': turnLeft(); break;      // Left
      case 'R': turnRight(); break;     // Right
      case 'S': stopMotors(); break;    // Stop
      default: stopMotors(); break;     // Stop on unknown command
    }
  }
}

// Function to stop all motors
void stopMotors() {
  digitalWrite(leftFrontMotorPin1, LOW);
  digitalWrite(leftFrontMotorPin2, LOW);
  digitalWrite(rightFrontMotorPin1, LOW);
  digitalWrite(rightFrontMotorPin2, LOW);

  digitalWrite(leftRearMotorPin1, LOW);
  digitalWrite(leftRearMotorPin2, LOW);
  digitalWrite(rightRearMotorPin1, LOW);
  digitalWrite(rightRearMotorPin2, LOW);
}

// Function for forward motion
void moveForward() {
  digitalWrite(leftFrontMotorPin1, HIGH);
  digitalWrite(leftFrontMotorPin2, LOW);
  digitalWrite(rightFrontMotorPin1, HIGH);
  digitalWrite(rightFrontMotorPin2, LOW);

  digitalWrite(leftRearMotorPin1, HIGH);
  digitalWrite(leftRearMotorPin2, LOW);
  digitalWrite(rightRearMotorPin1, HIGH);
  digitalWrite(rightRearMotorPin2, LOW);
}

// Function for backward motion
void moveBackward() {
  digitalWrite(leftFrontMotorPin1, LOW);
  digitalWrite(leftFrontMotorPin2, HIGH);
  digitalWrite(rightFrontMotorPin1, LOW);
  digitalWrite(rightFrontMotorPin2, HIGH);

  digitalWrite(leftRearMotorPin1, LOW);
  digitalWrite(leftRearMotorPin2, HIGH);
  digitalWrite(rightRearMotorPin1, LOW);
  digitalWrite(rightRearMotorPin2, HIGH);
}

// Function for turning left
void turnLeft() {
  digitalWrite(leftFrontMotorPin1, LOW);
  digitalWrite(leftFrontMotorPin2, HIGH);
  digitalWrite(rightFrontMotorPin1, HIGH);
  digitalWrite(rightFrontMotorPin2, LOW);

  digitalWrite(leftRearMotorPin1, LOW);
  digitalWrite(leftRearMotorPin2, HIGH);
  digitalWrite(rightRearMotorPin1, HIGH);
  digitalWrite(rightRearMotorPin2, LOW);
}

// Function for turning right
void turnRight() {
  digitalWrite(leftFrontMotorPin1, HIGH);
  digitalWrite(leftFrontMotorPin2, LOW);
  digitalWrite(rightFrontMotorPin1, LOW);
  digitalWrite(rightFrontMotorPin2, HIGH);

  digitalWrite(leftRearMotorPin1, HIGH);
  digitalWrite(leftRearMotorPin2, LOW);
  digitalWrite(rightRearMotorPin1, LOW);
  digitalWrite(rightRearMotorPin2, HIGH);
}