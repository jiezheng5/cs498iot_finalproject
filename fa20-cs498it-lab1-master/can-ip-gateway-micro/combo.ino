/* This is combo code for both IP-CAN-getway Arduino and the LED-and-sensor-controller Arduino.
 * You will need to set isGateway correctly to indicate which module this is for
 */

#include <mcp_can.h>
#include <SPI.h>
#include <string.h>
#include <Ethernet.h>

// Networking related for gateway - start
byte mac[] = {
  0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };

IPAddress ip(10, 0, 0, 4);
IPAddress myDns(10, 0, 0, 1);
IPAddress gateway(10, 0, 0, 1);
IPAddress subnet(255, 255, 255, 0);
EthernetServer server(8080);
// Networking related for gateway - end

int started = 0;

unsigned long prevProbe = 0;
unsigned long probeInvl = 100;

unsigned long canPrevTX = 0;                                        // Variable to store last execution time
const unsigned int canInvlTX = 500;                                 // Interval to send distance to gateway

const int ledPin = 8;

// Serial Output String Buffer
char msgString[128];

// CAN0 INT and CS
#define CAN0_INT 2                              // Set INT to pin 2
MCP_CAN CAN0(10);                               // Set CS to pin 10

enum {
  DEST_GATEWAY,
  DEST_IOSIDE
};

// structure used to send data on CAN
struct canMsg {
  unsigned char dest;
  unsigned short data;
};

enum {
  ETH_TYPE_LED,
  ETH_TYPE_DIST
};
// structure used to send data on ethernet
struct ipPkt {
  unsigned char type;
  unsigned char data[2];
};

long unsigned int rxId;
unsigned char rxLen;
struct canMsg rxBuf;
struct canMsg txBuf;

unsigned short ledOn = 0;
unsigned short distance = 0xFFFF;

// Local config
int isGateway = 1;
int reportDistanceEachProbe = 1;

//use digital pin 7 of arduino board to send trigger input of ultrasonic sensor
const int trigPin = 7;
//use digital pin 6 of arduino board to get echo output from ultrasonic sensor
const int echoPin = 6;
//HC-SR04 datasheet lists 10us pulse width as input of trigger signal 
int trigIn_duration_us = 10; 
//time for delay in microseconds
int delay_us = 5; 
//measrued echo time, i.e., output of pulseIn() in microsecond/us
unsigned long duration_us; 
//sound speed is 343 m/s, i.e., 0.0343 cm/us because pulseIn() of arduino outputs in microsecond
float sound_speed = 0.0343; 
//the suggested added delay to prevent trigger signal to the echo signal
int delay_meascycle_us = 60;
//unit in us, i.e., 300ms delay for testing readability during calibration, this is only used during code testing stage 
//int delayTestReadClear_us = 300000;


void setup_gateway()
{
  // initialize the ethernet device
  Ethernet.begin(mac, ip, myDns, gateway, subnet);

  // Check for Ethernet hardware present
  if (Ethernet.hardwareStatus() == EthernetNoHardware) {
    Serial.println("Ethernet shield was not found.  Sorry, can't run without hardware. :(");
    while (true) {
      delay(1); // do nothing, no point running without Ethernet hardware
    }
  }
  if (Ethernet.linkStatus() == LinkOFF) {
    Serial.println("Ethernet cable is not connected.");
  }

  // start listening for clients
  server.begin();

  Serial.print("Chat server address:");
  Serial.println(Ethernet.localIP());
}

void ioside_setup_led()
{
  pinMode(ledPin, OUTPUT);
}

void ioside_setup_ultrasonic()
{
  // sets the trigger pin 7 as output
  pinMode(trigPin, OUTPUT);  
  // sets the echo pin 6 as input
  pinMode(echoPin, INPUT); 
  //clear the trigger pin 7 
  digitalWrite(trigPin, LOW);
  delayMicroseconds(delay_us);
}

void setup_ioside()
{
  ioside_setup_led();
  ioside_setup_ultrasonic();
}

// RX IP packet and send msg to ioside
// or TX IP packet if something came from ioside
void can_request_led(unsigned short on)
{
    txBuf.dest = ETH_TYPE_LED;
    txBuf.data = on;
    byte sndStat = CAN0.sendMsgBuf(0x100, sizeof(txBuf), (byte *)&txBuf);
    
    if(sndStat != CAN_OK) {
      Serial.println("Error Sending LED Set Message...");
    }
}

void run_gateway()
{
  EthernetClient client = server.available();
  if (rxLen) {
    // Should be report of distance
    distance = rxBuf.data;
  }
  // when the client sends the first byte, say hello:
  //Serial.println("run gateway");
  if (client) {
    if (client.available() > 0) {
      // read the bytes incoming from the client:
      struct ipPkt pkt;
      memset(&pkt, sizeof(pkt), 0);
      client.read((unsigned char *)&pkt, sizeof(pkt));
      switch (pkt.type) {
        case ETH_TYPE_LED:
          //Serial.println("run_gateway: set LED");
          if (ledOn && (!pkt.data[0] && !pkt.data[1])) {
            //Serial.println("Gateway: request turn off LED");
            can_request_led(0);
            ledOn = 0;
          } else if (!ledOn && (pkt.data[0] || pkt.data[1])) {
            //Serial.println("Gateway: request turn on LED");
            can_request_led(1);
            ledOn = 1;
          }
          break;
        case ETH_TYPE_DIST:
          //Serial.println("run_gateway: reporting distance");
          // Convert to network order/sequence.
          pkt.data[1] = (distance) & 0xFF;
          pkt.data[0] = (distance >> 8) & 0xFF;
          server.write((unsigned char *)&pkt, sizeof(pkt));
          break;
        default:
          Serial.println("run_gateway: unexpected pkt.type");
          break;
      }
    }
  }
}

float measure_distance()
{
  digitalWrite(trigPin, LOW);
  delayMicroseconds(5);
  digitalWrite(trigPin, HIGH); 
  delayMicroseconds(trigIn_duration_us);           
  digitalWrite(trigPin, LOW); 
  
  //calculate the distance l=v*c/2 based on echo duration, i.e., the pulse width of the high pulse generated by echo pin
  //divided by 2 because of round trip
  duration_us = pulseIn(echoPin, HIGH);
  distance = duration_us*sound_speed/2; 

  //Using serial monitor to display the calculated distance
  //Serial.print("Duration in us and Distance in cm: ");
  //Serial.print(duration_us);
  //Serial.println(distance);
  //Serial.println(" ");
   
  //add a delay for prevent trigger signal to the echo signal
  delayMicroseconds(delay_meascycle_us);
}

void report_distance()
{
  if(reportDistanceEachProbe || (millis() - canPrevTX >= canInvlTX)){
    //Serial.println("Sending distance");
    canPrevTX = millis();
    txBuf.dest = DEST_GATEWAY;
    txBuf.data = distance;
    byte sndStat = CAN0.sendMsgBuf(0x100, sizeof(txBuf), (byte *)&txBuf);
    
    if(sndStat != CAN_OK) {
      Serial.println("Error Sending Distance Message...");
    }
  }
}

// Get distance from sensor and light up LED if needed
void run_ioside()
{
  if (rxLen) {
    // Received CAN message. Should be turn on/off LED
    if (rxBuf.data) {
      Serial.println("LED on");
      digitalWrite(ledPin, HIGH);
    } else {
      Serial.println("LED off");
      digitalWrite(ledPin, LOW);
    }
  }

  measure_distance();
  report_distance();
}

void setup_common()
{
  Serial.begin(115200);

  // Setup CAN
  if(CAN0.begin(MCP_ANY, CAN_500KBPS, MCP_16MHZ) == CAN_OK)
    Serial.println("MCP2515 Initialized Successfully!");
  else
    Serial.println("Error Initializing MCP2515...");

  CAN0.setMode(MCP_NORMAL);
  pinMode(CAN0_INT, INPUT);
}

void setup()
{
  setup_common();
  if (isGateway) {
    setup_gateway();
  } else {
    setup_ioside();
  }
  Serial.println("Finished setup");
}

void loop()
{
  // For testing, only after enter 1 in console the work will start and enter 0 in console to stop
  // to avoid to let your Arduino hang forever due to bad code
#ifdef FOR_TEST
  if (!started && Serial.read() == '1') {
      started = 1;
      
      Serial.println("Started...");
  } else if (started && Serial.read() == '0') {
      started = 0;

      Serial.println("Stopped...");
  }

  if (!started) {
    delay(100);
    return;
  }
#endif
  // Reading from CAN, shared by both IO Arduino and IP-CAN-getway Arduino
  rxLen = 0;
  if(!digitalRead(CAN0_INT))                          // If CAN0_INT pin is low, read receive buffer
  {
    CAN0.readMsgBuf(&rxId, &rxLen, (byte *)&rxBuf);
    //sprintf(msgString, "data %d %d", (int)rxBuf.dest, (int)rxBuf.data);
    //Serial.print(msgString);
    //Serial.println();
  }

  if (isGateway) {
    run_gateway();
  } else {
    run_ioside();
  }
  delay(100);
}

/*********************************************************************************************************
  END FILE
*********************************************************************************************************/
