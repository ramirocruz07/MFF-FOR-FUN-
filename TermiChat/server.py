
import socket
#create a socket object
#socket .socket(family,type)
#AF_INET:family of ipv4 ip address
#SOCK_DGRAM :UDP,Sock_stream:tcp
#some ip address that the server will listen to when message comes


def main():
    port = 3000
    chunk_size = 65535
    hostname = '127.0.0.1'

    # Create a UDP socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((hostname, port))
        print(f"Server is live on {s.getsockname()}")

        try:
            while True:
                # Receive data from clients
                data, client_address = s.recvfrom(chunk_size)
                message = data.decode('ascii')  # Decode received bytes to string
                print(f"Client {client_address} says: {message}")

                # Prepare a reply
                reply_message = input("Reply: ")
                reply_data = reply_message.encode("ascii")  # Encode reply to bytes

                # Send the reply back to the client
                s.sendto(reply_data, client_address)

        except KeyboardInterrupt:
            print("\nServer shutting down...")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
