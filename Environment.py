import numpy as np
import random
from serial_class import  serial_user
import struct
import time
import matlab.engine



class Environment():
    def __init__(self):
        self.v0 = np.zeros(1)
        self.v1 = np.zeros(1)
        self.p_ref = np.zeros(1)


        self.ser0 = serial_user("com1",115200)
        self.ser0.uart_open()
        self.ser1 = serial_user("com2",115200)
        self.ser1.uart_open()
        self.ser2 = serial_user("com3",115200)
        self.ser2.uart_open()




    def reset(self):
        self.v0 = random.randrange(100, 140)
        self.v1 = random.randrange(40, 50)
        self.p_ref = random.randrange(0, 200)

        y1 = np.hstack((self.v0/120, self.v1/45, self.p_ref/100))
        return y1

    def get_state(self):
        self.v0 = random.randrange(100, 140)
        self.v1 = random.randrange(40, 50)
        self.p_ref = random.randrange(0, 200)

        y1 = np.hstack((self.v0/120, self.v1/45, self.p_ref/100))
        return y1
    
    def Communicate_DSP(self,  send_data):
        if self.ser1.Ret:

            data = '53'  
            for datatmp in send_data:        

                bs = struct.pack("f",datatmp)
                for a in bs:
                    tmp = hex(a).split('0x')[1]

                    if len(tmp) == 1:
                        tmp = '0'+tmp
                    data=data+' '+tmp

            data=data+' 45'

            data = bytes.fromhex(data)
            self.ser1.serial_send_byte(data)
        else:
            print('serial_1 open error')

    
    def Communicate_DCsource_send(self,  power_data):
        string_power_data=str(power_data)
        send_data_source='VOLT '+string_power_data+ '\n'
        if  self.ser0.Ret:
            self.ser0.serial_send(send_data_source)
        else:
            print('serial_0_send open error')


    def Communicate_DCsource_read(self):
        if  self.ser0.Ret:
            self.ser0.serial_send('MEAS:POW?\n')
            time.sleep(0.15)
            data_P0w = self.ser0.serial_read_load_Pow()

            if data_P0w == 0:
                time.sleep(0.1)
                self.ser0.serial_send('MEAS:POW?\n')
                time.sleep(0.15)
                data_P0w = self.ser0.serial_read_load_Pow()

            return data_P0w
        else:
            print('serial_0_read open error')
            return -1


    def Communicate_load_set_send(self,  Res):
        string_load_data=str(Res)
        send_data_comd='SYST:REM'+ '\n'
        if  self.ser2.Ret:
            self.ser2.serial_send(send_data_comd)
        else:
            print('serial_2_send open error')
        time.sleep(0.1)
        send_data_load='RES '+string_load_data+ '\n'
        if  self.ser2.Ret:
            self.ser2.serial_send(send_data_load)
        else:
            print('serial_2_send open error')

    def Communicate_load_read(self):
            if  self.ser2.Ret:
                self.ser2.serial_send('FETC:POW?\n')
                time.sleep(0.15)
                data_P1w=self.ser2.serial_read_load_Pow()

                if data_P1w == 0:
                    time.sleep(0.1)
                    self.ser0.serial_send('FETC:POW?\n')
                    time.sleep(0.15)
                    data_P1w = self.ser0.serial_read_load_Pow()

                return data_P1w

            else:
                print('serial_2_read open error')
                return -1

    def Communicate_Read(self):
        if self.ser2.Ret:
            self.ser2.serial_send('FETC:POW?\n')
        if self.ser0.Ret:
            self.ser0.serial_send('FETC:POW?\n')
        time.sleep(0.2)
        data_P0w = self.ser0.serial_read_load_Pow()
        data_P1w = self.ser2.serial_read_load_Pow()
        return data_P0w, data_P1w


    def my_AVERAGE_main(self, data_list):
        if len(data_list) == 0:
            return 0
        if len(data_list) > 2:
            data_list.remove(min(data_list))
            data_list.remove(max(data_list))
            average_data = float(sum(data_list)) / len(data_list)
            return average_data
        elif len(data_list) <= 2:
            average_data = float(sum(data_list)) / len(data_list)
            return average_data



    def get_reward(self, action):
        trans = np.zeros(6, float)
        trans[0] = self.v0
        trans[1] = self.v1
        trans[2] = self.p_ref

        R = ((trans[1]**2) / trans[2])

        for i in range(3):
            trans[i+3] = (action[i] + 1)/2

        self.Communicate_DSP([2000, trans[3], trans[4], trans[5], 45])
        self.Communicate_DCsource_send(trans[0])
        self.Communicate_load_set_send(R)

        time.sleep(1)
        P0_1, P1_1  = self.Communicate_Read()
        P0_2, P1_2 = self.Communicate_Read()
        P0_3, P1_3 = self.Communicate_Read()
        P0_4, P1_4 = self.Communicate_Read()
        P0_5, P1_5 = self.Communicate_Read()


        list_0 = [P0_1, P0_2, P0_3, P0_4, P0_5]
        P0 = self.my_AVERAGE_main(list_0)


        list_1 = [P1_1, P1_2, P1_3, P1_4, P1_5]
        P1 = self.my_AVERAGE_main(list_1)


        PPP = abs(P1 - self.p_ref)
        self.reward = 8 * P1 / (P0 + 0.00001) - (PPP / 10)


        if self.reward >= 100:
            time.sleep(0.1)
            P0, P1 = self.Communicate_Read()
            self.reward = 8 * P1 / (P0 + 0.00001) - (PPP / 10)
            if self.reward >= 100:
                self.reward = -300

        transition = matlab.double(trans.tolist())

        print('trans: ', transition)
        print('P0_x: ', P0_1, P0_2, P0_3, P0_4, P0_5)
        print('P1_x: ', P1_1, P1_2, P1_3, P1_4, P1_5)
        print('reward=', self.reward, 'y=', 100 * P1 / (P0 + 0.0001), 'P0= ', P0, 'P1=', P1)
        return self.reward


