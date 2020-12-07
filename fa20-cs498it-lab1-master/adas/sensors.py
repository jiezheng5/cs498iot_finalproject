import socket

class Sensors:
    """
    Encapsulates the connection and brake status along with any 
    retry logic needed to for setting brake status and retrieving distance. 
    
    All functions are synchronous but should not have a time penalty 
    > 100ms including when socket re-initing a socket is required

    :Example
    s = Sensors()
    s.send_brake(True)
    s.find_distance()

    TODO: Socket retry logic
    """

    def __init__(self, ip='10.0.0.4', port=8080, buffer_size=3):
        self._dist_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._dist_socket_con_resp = self._dist_socket.connect((ip, port))
        self._brake_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._brake_socket_con_resp = self._brake_socket.connect((ip, port))
        self.brake_on_packet = b'\x00\x00\x01'
        self.brake_off_packet = b'\x00\x00\x00'
        self.sounding_packet =  b'\x01x00x00'
        self.brake_status = False
        self.ip = ip
        self.port = port
        self._buffer_size = 3
        
    def __repr__(self):
        return "Sensor<ip: {}, port: {}, brake: {}>".format(self.ip, self.port, self.brake_status)
    
    def send_brake(self, brake_bool):
        """Sends a message to modify the brake. If the current brake_status is already the same as brake_bool,
        no message is sent. Reuses existing socket connection
        
        returns final brake statues (same as input)
        """
        if brake_bool and not self.brake_status:
            self._brake_socket.send(self.brake_on_packet)
        elif not brake_bool and self.brake_status:
            self._brake_socket.send(self.brake_off_packet)
        # no communication back so we just keep going
        self.brake_status = brake_bool
        
        return brake_bool

    def find_distance(self):
        """Sends a packet causing the sensor to sound and return a
        packet with distance. Reuses existing socket connection

        returns distance in cm
        """
        self._dist_socket.send(self.sounding_packet)
        data = self._dist_socket.recv(self._buffer_size)
        raw_distance = int.from_bytes(data[-2:], "big") # last two bytes
        # Display 400 cm when past the range of the sensor
        return min(raw_distance, 400)
