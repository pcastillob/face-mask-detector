import serial
import time
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=0)

def getTemp():
    print("entró al getTemp")
    while True:
        cadena = arduino.readline()

        #time.sleep(4)
        try:
            var = cadena.decode()
            if(cadena != b'' and cadena!=b'100\r\n' and cadena!=b'0\r\n'):
                print("obtuvo la temperatura:")
                print(var)
                print(time.strftime("%M:%S"))
                return var
                break
            elif(cadena == b'1'):
                temp=99.9
                
                #cadena!=b'100\r\n'
            elif (cadena == b'0'):
                temp=0
               
            else:
                temp=0
                
        except:
            pass

while True:
    getTemp()