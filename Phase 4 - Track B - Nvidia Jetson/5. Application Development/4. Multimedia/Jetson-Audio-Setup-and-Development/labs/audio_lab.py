"""Original synthetic PCM fixture and stateful FIR decimation, NumPy CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import wave
import numpy as np
from audio_contracts import stream_budget, clock_drift

RATE=48000


def pcm16_encode(samples):
    x=np.asarray(samples,dtype=np.float64)
    if x.ndim!=2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('finite nonempty [frames,channels] required')
    return np.clip(np.rint(x*32768),-32768,32767).astype('<i2').tobytes()


def pcm16_decode(payload,channels):
    if type(channels) is not int or channels<=0 or len(payload)%(2*channels):
        raise ValueError('complete interleaved S16_LE frames required')
    return np.frombuffer(payload,dtype='<i2').reshape(-1,channels).astype(np.float64)/32768


def write_wave(path,samples,rate):
    if type(rate) is not int or rate<=0:raise ValueError('positive integer sample rate required')
    payload=pcm16_encode(samples)
    with wave.open(str(path),'wb') as f:
        f.setnchannels(samples.shape[1]);f.setsampwidth(2);f.setframerate(rate);f.writeframes(payload)


def read_wave(path):
    with wave.open(str(path),'rb') as f:
        if f.getsampwidth()!=2 or f.getcomptype()!='NONE':
            raise ValueError('this exercise supports uncompressed PCM16 WAV only')
        count,channels,rate=f.getnframes(),f.getnchannels(),f.getframerate()
        raw=f.readframes(count)
        if len(raw)!=count*channels*2:raise ValueError('truncated PCM data')
    return pcm16_decode(raw,channels),rate


def lowpass_taps(length=511,cutoff_hz=7000,rate=RATE):
    if type(length) is not int or length<3 or length%2!=1 or not 0<cutoff_hz<rate/2:
        raise ValueError('odd length >=3 and cutoff within Nyquist required')
    n=np.arange(length)-(length-1)/2
    h=2*cutoff_hz/rate*np.sinc(2*cutoff_hz/rate*n)*np.hanning(length)
    return h/h.sum()


class FIRDecimator:
    def __init__(self,taps,factor=3):
        self.taps=np.asarray(taps,dtype=np.float64)
        if self.taps.ndim!=1 or not len(self.taps) or not np.isfinite(self.taps).all():
            raise ValueError('finite one-dimensional taps required')
        if type(factor) is not int or factor<=0:raise ValueError('positive factor required')
        self.factor=factor
        self.history=np.zeros(len(self.taps)-1)
        self.seen=0

    def process(self,chunk):
        chunk=np.asarray(chunk,dtype=np.float64)
        if chunk.ndim!=1 or not np.isfinite(chunk).all():raise ValueError('finite mono samples required')
        if not len(chunk):return np.empty(0)
        joined=np.concatenate((self.history,chunk))
        filtered=np.convolve(joined,self.taps,mode='valid')
        output=filtered[np.arange(len(chunk))%self.factor==(-self.seen)%self.factor]
        self.seen+=len(chunk)
        if len(self.history):self.history=joined[-len(self.history):].copy()
        return output


def amplitude(samples,frequency,rate,trim=200):
    y=np.asarray(samples)[trim:-trim]
    t=(np.arange(len(y))+trim)/rate
    return float(2*abs(np.sum(y*np.exp(-2j*np.pi*frequency*t)))/len(y))


def run(output, filter_length=511):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    t=np.arange(RATE)/RATE
    channel0=.4*np.sin(2*np.pi*1000*t)+.2*np.sin(2*np.pi*12000*t)
    original=np.column_stack((channel0,.3*np.sin(2*np.pi*2000*t),-channel0,np.zeros_like(t)))
    path=output/'four-channel-48k.wav';write_wave(path,original,RATE)
    decoded,rate=read_wave(path)
    mono=decoded[:,0]
    taps=lowpass_taps(filter_length)
    full=FIRDecimator(taps).process(mono)
    stream=FIRDecimator(taps)
    chunks=[];offset=0
    for size in [127,480,17,1024,333,19]*30:
        if offset>=len(mono):break
        chunks.append(stream.process(mono[offset:offset+size]));offset+=size
    if offset<len(mono):chunks.append(stream.process(mono[offset:]))
    streamed=np.concatenate(chunks)
    naive=mono[::3]
    write_wave(output/'filtered-16k.wav',streamed[:,None],16000)
    write_wave(output/'naive-decimation-16k.wav',naive[:,None],16000)
    # Deliberately incorrect metadata-only conversion: same samples, different rate.
    write_wave(output/'wrong-header-16k.wav',mono[:,None],16000)
    a_bad=amplitude(naive,4000,16000)
    a_good=amplitude(streamed,4000,16000)
    report={'python':platform.python_version(),'numpy':np.__version__,
            'fixture':'original generated tones; no external recordings or personal data',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_rate':rate,'input_frames':len(decoded),'channels':decoded.shape[1],
            'pcm_roundtrip_max_error':float(np.max(np.abs(decoded-original))),
            'channel_rms':np.sqrt(np.mean(decoded**2,axis=0)).tolist(),
            'opposite_channel_average_rms':float(np.sqrt(np.mean(((decoded[:,0]+decoded[:,2])/2)**2))),
            'output_frames':len(streamed),'output_rate':16000,
            'stream_vs_single_block_max_error':float(np.max(np.abs(streamed-full))),
            'passband_1khz_amplitude':amplitude(streamed,1000,16000),
            'naive_4khz_alias_amplitude':a_bad,'filtered_4khz_alias_amplitude':a_good,
            'alias_reduction_db':float(20*np.log10(a_bad/max(a_good,1e-15))),
            'filter_taps':len(taps),'causal_group_delay_ms':1000*(len(taps)-1)/(2*RATE),
            'correct_duration_seconds':len(streamed)/16000,
            'wrong_header_duration_seconds':len(mono)/16000,
            'wrong_header_original_1khz_becomes_hz':1000*16000/RATE,
            'stream_budget':stream_budget(48000,4,2,480,4,4,32),
            'clock_drift_50ppm_600seconds':clock_drift(48000,50,600),
            'limitations':['no ALSA device or Jetson execution','fixed 3:1 mono FIR teaching resampler; no arbitrary rates or adaptive drift correction','synthetic audio does not measure speech accuracy or microphone quality']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--taps',type=int,default=511,help='odd FIR length, default 511')
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.taps),indent=2))
