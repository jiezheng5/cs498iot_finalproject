import time
import socket
def test_get_distance():
    HOST = "10.0.0.4"
    PORT = 8080

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
            s.connect((HOST, PORT)) 
            s.sendall(b"\x01\x00\x00") 
            data = s.recv(16) 
            print("distance: {}".format((data[1] << 8) + data[2])) 
            s.close() 
        time.sleep(0.1)

if __name__ == "__main__":
    test_get_distance()
