import serial
import serial.tools.list_ports

# 寻找串口
def ser_port_find(ser_com):
    port_list = serial.tools.list_ports.comports()
    if port_list is not None:
        list_len = port_list.__len__()
        serial_port = port_list[0].device
        # serial_port = "COM3"
        print(list_len, serial_port)
        ser_com.port = serial_port
        # ser_com.port = "COM3"
        return 1  # 成功找到串口
    else:
        print("no serial found!")
        return 0  # 没有发现串口

# 设置串口参数
def ser_port_set(ser_com):
    ser_com.baudrate = 115200  # 波特率
    ser_com.bytesize = 8
    ser_com.stopbits = 1
    ser_com.parity = serial.PARITY_NONE
    ser_com.timeout = 2

    try:
        ser_com.open()
        ser_com.flushInput()
        ser_com.flushOutput()
        print("Connected to serial port " + ser_com.port)
        return 1  # 成功打开串口
    except:
        print("Failed to open serial port " + ser_com.port)
        return 0  # 串口打开失败


# 串口打开
def ser_port_open():
    ser_com = serial.Serial()  # 定义串口对象
    found = ser_port_find(ser_com)
    if found != 1:
        return None
    opened = ser_port_set(ser_com)
    if opened == 1:
        return ser_com  # 串口打开成功
    return None  # 串口打开失败


# 串口关闭
def ser_port_close(ser_com):
    if ser_com.isOpen():
        ser_com.close()
    if ser_com.isOpen():
        print("failed to close!")
    else:
        print("successfully closed!")


# 发送数据
def ser_send(ser_com, send_data):
    if ser_com.isOpen():
        ser_com.write(send_data.encode('utf-8'))  # 编码方式
        print("发送成功: ", send_data)
        return 1  # 数据发送成功
    else:
        print("发送失败: ", send_data)
        return 0  # 数据发送失败


def ser_send_gb(ser_com, send_data):
    if ser_com.isOpen():
        ser_com.write(send_data.encode('GB2312'))  # 编码方式
        print("发送成功: ", send_data)
        return 1  # 数据发送成功
    else:
        print("发送失败: ", send_data)
        return 0  # 数据发送失败


# 读取
def ser_read(ser_com):
    ser_message_read = ''
    if ser_com.isOpen() and ser_com.in_waiting:
        ser_message_read = ser_com.readline().decode('utf-8')
        print("received: ", ser_message_read)
        return ser_message_read
    return None


def ser_read_gb(ser_com):
    ser_message_read = ''
    if ser_com.isOpen() and ser_com.in_waiting:
        ser_message_read = ser_com.readline().decode('GB2312')
        return ser_message_read
    return None


if __name__ == '__main__':
    Tcom = ser_port_open()
    if Tcom is not None and Tcom.isOpen():
        print("baudrate: ", Tcom.baudrate)
        ser_send(Tcom, "python uart ready!")
        while 1:
            ser_message = input("input your message: ")
            if ser_message == 'q':
                ser_port_close(Tcom)
                break
            ser_send(Tcom, ser_message)
