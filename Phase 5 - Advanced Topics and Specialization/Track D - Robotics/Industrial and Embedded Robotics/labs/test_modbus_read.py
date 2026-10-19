import struct
import unittest
from modbus_read import (ADU,HEADER,FrameError,ModbusException,read_adu,read_request,
                         decode_response,respond,demo)


class ShortReads:
    """Deterministic stream fixture: return at most chunk bytes per recv call."""
    def __init__(self,data,chunk=1):self.data=data;self.chunk=chunk
    def recv(self,count):
        result,self.data=self.data[:min(count,self.chunk)],self.data[min(count,self.chunk):]
        return result


class ModbusReadTests(unittest.TestCase):
    def test_independent_golden_request_and_response(self):
        request=read_request(0x1234,0,2)
        self.assertEqual(request.wire().hex(),'123400000006010300000002')
        response=read_adu(ShortReads(bytes.fromhex('12340000000701030409c41388')))
        self.assertEqual(decode_response(response,request),[2500,5000])

    def test_every_short_read_boundary_and_coalesced_frames(self):
        first=read_request(1,0,1);second=read_request(2,1,1)
        for chunk in range(1,30):
            stream=ShortReads(first.wire()+second.wire(),chunk)
            self.assertEqual(read_adu(stream),first);self.assertEqual(read_adu(stream),second)
            self.assertEqual(stream.data,b'')

    def test_all_truncations_fail(self):
        wire=read_request(1,0,2).wire()
        for cut in range(len(wire)):
            with self.subTest(cut=cut),self.assertRaises(EOFError):read_adu(ShortReads(wire[:cut]))

    def test_invalid_protocol_and_length_fail_before_payload(self):
        for protocol,length in [(1,6),(0,0),(0,1),(0,255),(0,65535)]:
            wire=HEADER.pack(1,protocol,length,1)
            with self.assertRaises(FrameError):read_adu(ShortReads(wire))

    def test_transaction_and_unit_must_match(self):
        request=read_request(1,0,1)
        response=respond(request,{0:42})
        for changed in [ADU(2,1,response.pdu),ADU(1,2,response.pdu)]:
            with self.assertRaises(FrameError):decode_response(changed,request)

    def test_wrong_function_byte_count_and_data_size_rejected(self):
        request=read_request(1,0,1)
        for pdu in [b'\x04\x02\x00\x01',b'\x03',b'\x03\x01\x00\x01',
                    b'\x03\x02\x00',b'\x03\x02\x00\x01\x02',b'\x83\x02\x00']:
            with self.assertRaises(FrameError):decode_response(ADU(1,1,pdu),request)

    def test_exception_is_not_successful_empty_data(self):
        request=read_request(1,100,1);response=respond(request,{0:42})
        with self.assertRaises(ModbusException) as result:decode_response(response,request)
        self.assertEqual(result.exception.code,2)

    def test_read_limits_and_type_boundaries(self):
        self.assertEqual(len(respond(read_request(1,0,125),dict.fromkeys(range(125),0)).pdu),252)
        for args in [(1,0,0),(1,0,126),(1,65535,2),(True,0,1),(1,0,True)]:
            with self.assertRaises(FrameError):read_request(*args)
        with self.assertRaises(FrameError):ADU(1,1,b'')
        with self.assertRaises(FrameError):ADU(1,1,b'x'*254)

    def test_fixture_refuses_unsupported_function_and_bad_quantity(self):
        registers={0:42}
        self.assertEqual(respond(ADU(1,1,b'\x10'),registers).pdu,b'\x90\x01')
        self.assertEqual(registers,{0:42})
        self.assertEqual(respond(ADU(1,1,struct.pack('!BHH',3,0,0)),registers).pdu,b'\x83\x03')

    def test_address_space_boundary_is_enforced_by_server(self):
        # A wire peer bypasses read_request; an oversized fixture map must not
        # accidentally make address 65536 accessible through an overflowing range.
        registers={65535:42,65536:43}
        valid=read_request(1,65535,1)
        self.assertEqual(decode_response(respond(valid,registers),valid),[42])
        invalid=ADU(2,1,struct.pack('!BHH',3,65535,2))
        self.assertEqual(respond(invalid,registers).pdu,b'\x83\x02')

    def test_actual_localhost_tcp_exchange(self):
        result=demo()
        self.assertEqual(result['cases'][0]['registers'],[2500,5000])
        self.assertEqual([row.get('exception_code') for row in result['cases']],[None,2,1])


if __name__=='__main__':unittest.main()
