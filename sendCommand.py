import serial
import time

def send_command(port, baud_rate, command):
    try:
        with serial.Serial(port, baud_rate, timeout=1) as bt:
            if command == 'stop':
                command = 'S'
            elif command == 'forward':
                command = 'F'
            elif command == 'backward':
                command = 'B'
            elif command == 'left':
                command = 'L'
            elif command == 'right':
                command = 'R'
            print(f"Sending command: {command}")
            bt.write(command.encode('utf-8'))  # Send command to HC-05
            time.sleep(0.5)  # Allow time for Arduino to process
    except serial.SerialException as e:
        print(f"Serial exception: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    port = "COM4"  # Replace with the correct outgoing port for HC-05
    baud_rate = 9600
    
    print("Robot Control")
    print("Commands: F (Forward), B (Backward), L (Left), R (Right), S (Stop)")
    
    while True:
        command = input("Enter command: ").strip().upper()
        if command in ['F', 'B', 'L', 'R', 'S']:
            send_command(port, baud_rate, command)
        elif command == 'Q':  # Quit the script
            print("Exiting...")
            break
        else:
            print("Invalid command. Try again.")