from __future__ import print_function
import openbci
import time
import sys
import serial
import glob


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


print(serial_ports())
board = openbci.OpenBCICyton(port='/dev/tty.usbserial-DM00Q4BH')#raw_input("input serial port with board: "))

s = 'x1050110Xx2050110Xx3050110Xx4050110Xx5050110Xx6050110Xx7050110Xx8050110X'
#for c in s:
#    if sys.hexversion > 0x03000000:
#        board.ser_write(bytes(c, 'utf-8'))
#    else:
#        board.ser_write(bytes(c))
#    time.sleep(0.100)


