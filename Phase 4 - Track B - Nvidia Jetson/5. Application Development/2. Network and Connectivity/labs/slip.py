"""RFC1055 byte framing only. Does not implement ZNSP payload, CRC or commands."""
END, ESC, ESC_END, ESC_ESC = 0xC0, 0xDB, 0xDC, 0xDD


def encode(payload):
    output=bytearray([END])
    for byte in payload:
        output.extend((ESC,ESC_END) if byte==END else (ESC,ESC_ESC) if byte==ESC else (byte,))
    output.append(END)
    return bytes(output)


class Decoder:
    def __init__(self, maximum=4096):
        if type(maximum) is not int or maximum<=0:
            raise ValueError('positive maximum required')
        self.maximum=maximum
        self.buffer=bytearray()
        self.escaped=False
        self.invalid=False
        self.rejected=0

    def feed(self, data):
        frames=[]
        for byte in data:
            if byte==END:
                if self.invalid or self.escaped:
                    self.rejected+=1
                elif self.buffer:
                    frames.append(bytes(self.buffer))
                self.buffer.clear();self.escaped=False;self.invalid=False
                continue
            if self.invalid:continue
            if self.escaped:
                self.escaped=False
                if byte not in (ESC_END,ESC_ESC):
                    self.invalid=True;self.buffer.clear();continue
                byte=END if byte==ESC_END else ESC
            elif byte==ESC:
                self.escaped=True;continue
            self.buffer.append(byte)
            if len(self.buffer)>self.maximum:
                self.invalid=True;self.buffer.clear()
        return frames
