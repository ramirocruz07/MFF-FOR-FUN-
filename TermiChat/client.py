import socket
import time

def main():
    port = 3000
    chunk_size = 65535
    hostname = '127.0.0.1'

    # Create a UDP socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        print(f"Connected to server at {hostname}:{port}")
        
        while True:
            try:
                message = input("Type message (or 'exit' to quit): ")
                if message.lower() == 'exit':
                    print("Exiting...")
                    break
                
                # Encode and send the message
                data = message.encode("ascii")
                s.sendto(data, (hostname, port))
                
                # Receive a response
                response, _ = s.recvfrom(chunk_size)
                text = response.decode("ascii")
                print(f"Server response: {text} - {time.asctime()}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
