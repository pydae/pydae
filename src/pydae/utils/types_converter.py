import time

def read(context,reg_number,var_type,format = 'CDAB'):
    if var_type == 'int32':  value = read_int32(context,reg_number, format=format)
    if var_type == 'uint32': value = read_uint32(context,reg_number, format=format)
    # if var_type == 'uint16': value = read_uint16(reg_number, format=format)
    # if var_type == 'int16':  value = read_int16(reg_number, format=format)

    return value

def write(self,value,reg_number,var_type,format = 'CDAB'):
    if var_type == 'int32': self.write_int32(value,reg_number, format=format)
    if var_type == 'uint32': self.write_uint32(value,reg_number, format=format)
    if var_type == 'uint16': self.write_uint16(value,reg_number, format=format)
    if var_type == 'int16': self.write_int16(value,reg_number, format=format)

def write_int32(self,value, reg_number, format = 'CDAB'):
    # write INT32
    if format == 'CDAB':
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)

    builder.add_32bit_int(value)
    builder.to_registers()
    payload = builder.build()
    
    response  = self.modbus_client.write_registers(reg_number, payload,skip_encode=True) 

def read_int32(context,reg_number, format = 'CDAB'):
    if format == 'CDAB':
       high, low = context[0].getValues(3, reg_number, count=len(format))

    # Combine the two 16-bit parts into a 32-bit integer
    combined = (high << 16) | low
    
    # Convert to signed 32-bit if the highest bit (bit 31) is set
    if combined >= 2**31:  # 0x80000000 (2147483648)
        combined -= 2**32  # 0x100000000 (4294967296)

    return combined

def write_uint32(self,value, reg_number, format = 'CDAB'):
    # write INT32

    if format == 'CDAB':
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)

    builder.add_32bit_uint(value)
    builder.to_registers()
    payload = builder.build()
    
    response  = self.modbus_client.write_registers(reg_number, payload,skip_encode=True) 

def read_uint32(self,reg_number, format = 'CDAB'):
    # read INT32

    modbus_response = self.modbus_client.read_holding_registers(address = reg_number, count = 2)
    if format == 'CDAB':
        decoder = BinaryPayloadDecoder.fromRegisters(modbus_response.registers, byteorder=Endian.Big, wordorder=Endian.Little)
    value = decoder.decode_32bit_uint()
    return value

    # def write_int16(self,value, reg_number, format = 'AB'):
    #     print(value, format)
    #     # write INT32
    #     if format == 'AB':
    #         builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)

    #     builder.add_16bit_int(value)
    #     builder.to_registers()
    #     payload = builder.build()
        
    #     response  = self.modbus_client.write_registers(reg_number, payload,skip_encode=True) 
 
    def write_int16(self,value, reg_number, format = 'AB'):

        # Create a payload builder
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)

        # Add a 16-bit integer (2-byte) to the payload
        builder.add_16bit_int(int(value))

        # Build the payload
        payload = builder.build()

        # The address of the register to write to
        register_address = reg_number  # Replace with the actual register address

        # Write the payload (16-bit integer) to the register
        response  = self.modbus_client.write_registers(reg_number, payload,skip_encode=True) 


    def read_int16(self,reg_number, format = 'AB'):
        # read INT16
        modbus_response = self.modbus_client.read_holding_registers(address = reg_number, count = 1)
        if format == 'AB':
            decoder = BinaryPayloadDecoder.fromRegisters(modbus_response.registers, byteorder=Endian.Big, wordorder=Endian.Little)
        value = decoder.decode_16bit_int()
        return value

    def write_uint16(self,value, reg_number, format = 'AB'):
        # write INT32
        if format == 'AB':
            builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)

        builder.add_16bit_uint(value)
        builder.to_registers()
        payload = builder.build()
        
        response  = self.modbus_client.write_registers(reg_number, payload,skip_encode=True) 

    def read_uint16(self,reg_number, format = 'AB'):
        # read INT32
        modbus_response = self.modbus_client.read_holding_registers(address = reg_number, count = 2)
        if format == 'AB':
            decoder = BinaryPayloadDecoder.fromRegisters(modbus_response.registers, byteorder=Endian.Big, wordorder=Endian.Little)
        value = decoder.decode_16bit_uint()
        return value

if __name__ == "__main__":



    # import time

    # ip = "127.0.0.1"
    # port = 5200
    # mb = Modbus_client(ip,port=port)
    # mb.start()

    # # active powers
    # p = int(0.0e6)
    # reg_number = 1000
    # mb.write(p, reg_number, 'int32',format = 'CDAB')
    # reg_number = 2000
    # mb.write(p, reg_number, 'int32',format = 'CDAB')

    # # reactive powers
    # q = int(-0.1e6)
    # reg_number = 1004
    # mb.write(q, reg_number, 'int32',format = 'CDAB')
    # reg_number = 2004
    # mb.write(q, reg_number, 'int32',format = 'CDAB')

    # time.sleep(0.1)

    # reg_number = 0
    # value_echo = mb.read(reg_number, 'int16', format = 'AB')

    # mb.close()

    # ips = ["127.10.1.1","127.10.1.2"]
    # ports = [50101,50102]
 
    # # # active powers
    # # p = int(0.85e6)
    # # reg_number = 1000
    # # mb.write(p, reg_number, 'int32',format = 'CDAB')
    # # reg_number = 2000
    # # mb.write(p, reg_number, 'int32',format = 'CDAB')


    # # reactive powers
    # q = int(0.0e6)
    # for ip,port in zip(ips,ports):
    #     mb = Modbus_client(ip,port=port)
    #     mb.start()
    #     reg_number = 40426
    #     mb.write(q, reg_number, 'int32',format = 'CDAB')
    #     mb.close()

    # time.sleep(0.3)

    # ip = "127.100.0.1"
    # port = 5100
    # mb = Modbus_client(ip,port=port)
    # mb.start()
    # reg_number = 0
    # value_echo = mb.read(reg_number, 'int16', format = 'AB')
    # print(value_echo)

    # mb.close()


    # ip = "127.100.0.1"
    # port = 5100
    # mb = Modbus_client(ip,port=port)
    # mb.start()

    # # reactive powers
    # p = int(0.85e6)
    # reg_number = 1000
    # mb.write(p, reg_number, 'int32',format = 'CDAB')
    # reg_number = 2000
    # mb.write(p, reg_number, 'int32',format = 'CDAB')


    # # reactive powers
    # q = int(0.0e6)
    # reg_number = 1004
    # mb.write(q, reg_number, 'int32',format = 'CDAB')
    # reg_number = 2004
    # mb.write(q, reg_number, 'int32',format = 'CDAB')

    # time.sleep(0.1)

    # reg_number = 0
    # value_echo = mb.read(reg_number, 'int16', format = 'AB')
    # print(value_echo)

    # mb.close()


    # print('Client started')

    # for port in [510,511,512,513]:
    #     mb = Modbus_client(ip,port=port)
    #     mb.start()

    #     # reactive powers
    #     value = int(0.9e6)
    #     reg_number = 40426
    #     mb.write(value, reg_number, 'int32',format = 'CDAB')

    #     # active powers
    #     value = int(2.0e6)
    #     reg_number = 40424
    #     mb.write(value, reg_number, 'uint32',format = 'CDAB')

    # for port in [510,511,512,513]:
    #     mb = Modbus_client(ip,port=port)
    #     mb.start()

    #     # reactive powers
    #     value = int(0.0e6)
    #     reg_number = 40426
    #     value_echo = mb.read(reg_number, 'int32', format = 'CDAB')

    #     print(value_echo/1000)

    # mb = Modbus_client(ip,port=5002)
    # mb.start()
    # reg_number = 372
    # value_echo = mb.read(reg_number, 'int16', format = 'CDAB')
    # print(value_echo/1000)

    

