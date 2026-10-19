#!/usr/bin/env python3
"""Check the teaching read-only server against an actual PyModbus 3.16.1 client."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
from modbus_read import read_adu,respond


def experiment():
    import pymodbus
    from pymodbus.client import ModbusTcpClient
    requests=[]
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(3)
        port=listener.getsockname()[1]
        def serve():
            connection,_=listener.accept()
            with connection:
                connection.settimeout(3)
                for _ in range(2):
                    request=read_adu(connection);requests.append(request.wire().hex())
                    wire=respond(request,{0:2500,1:5000}).wire()
                    connection.sendall(wire[:2]);connection.sendall(wire[2:])
        with ThreadPoolExecutor(max_workers=1) as pool:
            server=pool.submit(serve)
            client=ModbusTcpClient('127.0.0.1',port=port,timeout=3,retries=0)
            try:
                assert client.connect(),'localhost connection failed'
                good=client.read_holding_registers(0,count=2,device_id=1)
                assert not good.isError() and good.registers==[2500,5000]
                bad=client.read_holding_registers(100,count=1,device_id=1)
                assert bad.isError() and bad.exception_code==2
            finally:client.close()
            server.result(timeout=3)
    return {'scope':'actual PyModbus client and teaching server over localhost TCP; no external endpoint',
            'pymodbus':pymodbus.__version__,'checks_passed':2,'registers':good.registers,
            'illegal_address_exception_code':bad.exception_code,'captured_request_hex':requests,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'server_sha256':hashlib.sha256(Path(__file__).with_name('modbus_read.py').read_bytes()).hexdigest(),
            'limitations':['no physical PLC/sensor','read-only subset, not conformance certification',
                           'serial/TLS/retries and client concurrency not exercised']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();text=json.dumps(experiment(),indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')
