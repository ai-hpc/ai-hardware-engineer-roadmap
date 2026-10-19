"""Small read-only Modbus TCP teaching subset; generated data on localhost only.

No physical PLC, robot, coil/register writes, TLS or conformance claim.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import socket
import struct

HEADER=struct.Struct('!HHHB')


class FrameError(ValueError):pass


class ModbusException(FrameError):
    def __init__(self,code):self.code=code;super().__init__(f'Modbus exception code {code}')


def integer(value,low,high,name):
    if type(value) is not int or not low<=value<=high:raise FrameError(f'{name} outside [{low},{high}]')
    return value


@dataclass(frozen=True)
class ADU:
    transaction: int
    unit: int
    pdu: bytes

    def __post_init__(self):
        integer(self.transaction,0,65535,'transaction');integer(self.unit,0,255,'unit')
        if not isinstance(self.pdu,bytes) or not 1<=len(self.pdu)<=253:raise FrameError('PDU must be 1..253 bytes')

    def wire(self):return HEADER.pack(self.transaction,0,1+len(self.pdu),self.unit)+self.pdu


def recv_exact(connection,count):
    data=bytearray()
    while len(data)<count:
        chunk=connection.recv(count-len(data))
        if not chunk:raise EOFError('peer closed before complete frame')
        data.extend(chunk)
    return bytes(data)


def read_adu(connection):
    transaction,protocol,length,unit=HEADER.unpack(recv_exact(connection,HEADER.size))
    if protocol!=0:raise FrameError('Modbus protocol identifier must be zero')
    if not 2<=length<=254:raise FrameError('MBAP length must count unit plus PDU')
    return ADU(transaction,unit,recv_exact(connection,length-1))


def read_request(transaction,address,count,unit=1):
    integer(address,0,65535,'address');integer(count,1,125,'register count')
    if address+count>65536:raise FrameError('register range exceeds address space')
    return ADU(transaction,unit,struct.pack('!BHH',3,address,count))


def decode_response(response,request):
    if len(request.pdu)!=5 or request.pdu[0]!=3:raise FrameError('expected read-holding-registers request')
    count=struct.unpack('!H',request.pdu[3:])[0];integer(count,1,125,'request count')
    if response.transaction!=request.transaction or response.unit!=request.unit:
        raise FrameError('response transaction or unit mismatch')
    if response.pdu[0]==0x83:
        if len(response.pdu)!=2:raise FrameError('malformed exception PDU')
        raise ModbusException(response.pdu[1])
    if response.pdu[0]!=3:raise FrameError('response function mismatch')
    if len(response.pdu)!=2+2*count or response.pdu[1]!=2*count:
        raise FrameError('response byte count or register count mismatch')
    return list(struct.unpack('!'+'H'*count,response.pdu[2:]))


def respond(request,registers):
    """Fixture supports only unit 1/function 03; no write handler exists."""
    if request.unit!=1:raise FrameError('fixture only hosts unit 1')
    function=request.pdu[0]
    if function!=3:return ADU(request.transaction,request.unit,bytes([function|0x80,1]))
    if len(request.pdu)!=5:return ADU(request.transaction,request.unit,b'\x83\x03')
    address,count=struct.unpack('!HH',request.pdu[1:])
    if not 1<=count<=125:return ADU(request.transaction,request.unit,b'\x83\x03')
    if address+count>65536 or any(index not in registers for index in range(address,address+count)):
        return ADU(request.transaction,request.unit,b'\x83\x02')
    values=[integer(registers[index],0,65535,'register value') for index in range(address,address+count)]
    return ADU(request.transaction,request.unit,bytes([3,2*count])+struct.pack('!'+'H'*count,*values))


def demo():
    registers={0:2500,1:5000} # fictional temperature hundredths C and supply millivolts
    requests=[read_request(0x1234,0,2),read_request(0x1235,100,1),ADU(0x1236,1,b'\x10')]
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(3)
        address=listener.getsockname()
        def serve():
            connection,_=listener.accept()
            with connection:
                connection.settimeout(3)
                for _ in requests:
                    request=read_adu(connection);wire=respond(request,registers).wire()
                    # Separate sends may be combined by TCP. Unit tests also enforce short reads.
                    for block in (wire[:1],wire[1:5],wire[5:]):connection.sendall(block)
        rows=[]
        with ThreadPoolExecutor(max_workers=1) as pool:
            server=pool.submit(serve)
            with socket.create_connection(address,timeout=3) as client:
                for request in requests:
                    wire=request.wire();client.sendall(wire[:3]);client.sendall(wire[3:])
                    response=read_adu(client)
                    row={'transaction':request.transaction,'request_hex':wire.hex(),
                         'response_hex':response.wire().hex()}
                    if request.pdu[0]==3:
                        try:row['registers']=decode_response(response,request)
                        except ModbusException as error:row['exception_code']=error.code
                    else:
                        assert response.pdu==b'\x90\x01';row['exception_code']=response.pdu[1]
                    rows.append(row)
            server.result(timeout=3)
    assert rows[0]['registers']==[2500,5000] and rows[1]['exception_code']==2
    return {'scope':'actual localhost TCP client/server; generated register data; read-only subset',
            'cases':rows,'decoded_application_units':{'temperature_C':25.,'supply_V':5.},
            'limitations':['no physical PLC/robot or sensor','not a full Modbus conformance implementation',
                           'no TLS/authentication or deterministic timing claim','no write operation or retry policy']}


if __name__=='__main__':print(json.dumps(demo(),indent=2))
