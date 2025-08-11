#include <Servo.h>

// Define the pins for the thrusters
const int LEFT_THRUSTER_PIN = 9;
const int RIGHT_THRUSTER_PIN = 10;

// Create Servo objects
Servo leftThruster;
Servo rightThruster;

// Variables to store PWM values
int pwmLeft = 1500;  // Default to neutral (1500 microseconds)
int pwmRight = 1500; // Default to neutral

void setup() {
  // Start serial communication
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  // Attach servos to their pins
  leftThruster.attach(LEFT_THRUSTER_PIN);
  rightThruster.attach(RIGHT_THRUSTER_PIN);

  // Arm the ESCs (optional, but often necessary)
  // Send neutral signal for a couple of seconds
  Serial.println("Arming ESCs...");
  leftThruster.writeMicroseconds(1500);
  rightThruster.writeMicroseconds(1500);
  delay(2000); // Adjust delay as needed for your ESCs
  Serial.println("ESCs Armed.");
}

void loop() {
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');
    inputString.trim(); // Remove any leading/trailing whitespace

    // Find the space separating the two PWM values
    int spaceIndex = inputString.indexOf(' ');

    if (spaceIndex > 0) {
      String pwmLeftStr = inputString.substring(0, spaceIndex);
      String pwmRightStr = inputString.substring(spaceIndex + 1);

      // Convert strings to integers
      pwmLeft = pwmLeftStr.toInt();
      pwmRight = pwmRightStr.toInt();

      // Constrain PWM values to a safe range (e.g., 1000-2000 microseconds)
      // Adjust these values based on your ESC's specifications
      pwmLeft = constrain(pwmLeft, 1000, 2000);
      pwmRight = constrain(pwmRight, 1000, 2000);

      // Write the PWM values to the thrusters
      leftThruster.writeMicroseconds(pwmLeft);
      rightThruster.writeMicroseconds(pwmRight);

      // Optional: Print received values for debugging
      // Serial.print("Received Left: ");
      // Serial.print(pwmLeft);
      // Serial.print(" Right: ");
      // Serial.println(pwmRight);
    } else {
      // Optional: Handle malformed input
      // Serial.println("Error: Malformed input string");
    }
  }
  // Add a small delay to prevent overwhelming the serial buffer or ESCs
  // if commands are sent extremely rapidly without a newline.
  // Your Python script's timer_callback already paces this, so it might not be strictly needed here.
  // delay(10); 
}