"""Executed tinygrad 0.14.0 CPU lessons; no vehicle connection or model download."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile

# Configure before import. These lessons deliberately execute only on the CPU.
os.environ['DEV'] = 'CPU'
os.environ['BEAM'] = '0'
os.environ['DEBUG'] = os.environ.get('COURSE_TINYGRAD_DEBUG', '0')
if version('tinygrad') != '0.14.0':
    raise RuntimeError('This source is tested with tinygrad 0.14.0; use the pinned requirements file.')
from tinygrad import Tensor, TinyJit, Context, Device, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import safe_save, safe_load
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'labs'))
import reference


def flat(values):
    return [x for row in values for x in flat(row)] if isinstance(values, list) and values and isinstance(values[0],list) else values


def close(actual, expected, atol=1e-5):
    def shape(x):
        if not isinstance(x,list):return ()
        if x and any(shape(row)!=shape(x[0]) for row in x):raise AssertionError('ragged result')
        return (len(x),)+ (shape(x[0]) if x else ())
    if shape(actual)!=shape(expected):raise AssertionError('result shape differs from reference')
    aa, bb = flat(actual), flat(expected)
    if len(aa) != len(bb) or any(not math.isfinite(a) or abs(a-b) > atol for a,b in zip(aa,bb)):
        raise AssertionError(f'numerical mismatch: {actual} vs {expected}')
    return max((abs(a-b) for a,b in zip(aa,bb)), default=0.)


def tensor(values):
    return Tensor(values, device='CPU', dtype=dtypes.float32).realize()


def intro():
    x = tensor([-1., 0., 2.]); y = (x*2+1).relu().tolist()
    close(y, [0.,1.,5.])
    return {'input':x.tolist(), 'output':y, 'device':str(Device.DEFAULT), 'dtype':'float32'}


def basics():
    x = tensor([[-1,0,2],[3,-2,1]])
    GlobalCounters.reset()
    result = (x*2+1).relu().realize()
    Device['CPU'].synchronize()
    kernels = GlobalCounters.kernel_count
    close(result.tolist(), [[0,1,5],[7,0,3]])
    transposed = x.transpose().contiguous().tolist()
    close(transposed, [[-1,3],[0,-2],[2,1]])
    return {'shape':list(x.shape), 'output':result.tolist(), 'realization_kernel_count':kernels,
            'transpose_shape':[3,2], 'materialized_transpose':transposed}


def operations():
    a = [[1,2,3],[-1,0,2]]; b = [[1,0],[2,1],[-1,3]]
    x, y = tensor(a), tensor(b)
    manual = (x.reshape(2,3,1) * y.reshape(1,3,2)).sum(axis=1)
    expected = reference.matmul(a,b)
    close(manual.tolist(), expected);close((x@y).tolist(), expected)
    close(x.sum(axis=0).tolist(), [0,2,5])
    return {'manual_matmul':manual.tolist(), 'matmul':(x@y).tolist(),
            'sum_axis0':x.sum(axis=0).tolist(), 'row_mean':x.mean(axis=1).tolist()}


def autograd():
    x = tensor([2.,3.]);(x*x).sum().backward()
    close(x.grad.tolist(),[4.,6.])
    z = tensor([.5,-1.,2.]); normalized = z/(z.square().mean()+1e-6).sqrt()
    normalized.sum().backward()
    numerical = reference.finite_gradient(lambda v:sum(reference.rmsnorm(v)), [.5,-1.,2.])
    error = close(z.grad.tolist(), numerical, atol=2e-4)
    w = tensor([0.]); optim = SGD([w], lr=.1)
    losses = []
    with Context(TRAINING=1):
        for _ in range(20):
            optim.zero_grad();loss=(w-2).square().sum();losses.append(loss.item())
            loss.backward();optim.step()
    close(w.tolist(),[2*(1-.8**20)],atol=2e-6)
    return {'sum_squares_gradient':x.grad.tolist(), 'rmsnorm_gradient':z.grad.tolist(),
            'finite_difference_max_error':error, 'sgd_losses':losses, 'sgd_weight':w.tolist(),
            'sgd_hand_formula':'w_t = 2*(1-0.8**t)'}


def classifier(mnist=False, steps=40):
    """Bounded classification + save/reload. Download MNIST only with --mnist."""
    if not 1 <= steps <= 500: raise ValueError('steps must be in [1,500]')
    Tensor.manual_seed(17)
    if mnist:
        from tinygrad.nn.datasets import mnist as load_mnist
        tx,ty,vx,vy = load_mnist(device='CPU')
        x = (tx[:512].reshape(512,784).cast(dtypes.float32)/255).realize()
        labels = ty[:512].cast(dtypes.int32).realize()
        test_x = (vx[:256].reshape(256,784).cast(dtypes.float32)/255).realize()
        test_y = vy[:256].cast(dtypes.int32).realize()
        width,classes = 784,10
        dataset = 'MNIST: first 512 training / first 256 test examples; small biased subset'
    else:
        x = tensor([[-2,-1],[-1,-2],[1,2],[2,1]])
        labels = Tensor([0,0,1,1],device='CPU',dtype=dtypes.int32).realize()
        test_x=tensor([[-1,-1],[1,1]])
        test_y=Tensor([0,1],device='CPU',dtype=dtypes.int32).realize()
        width,classes=2,2
        dataset='synthetic two-class fixture; not MNIST or driving data'
    w = Tensor.zeros(width,classes,device='CPU',dtype=dtypes.float32).realize()
    bias = Tensor.zeros(classes,device='CPU',dtype=dtypes.float32).realize()
    optim=SGD([w,bias],lr=.2)
    first=(x@w+bias).sparse_categorical_crossentropy(labels).item()
    with Context(TRAINING=1):
        for step in range(steps):
            # Deterministic bounded batches; each training sample stays in the train split.
            start=(step*32)%x.shape[0];end=min(start+32,x.shape[0])
            optim.zero_grad();loss=(x[start:end]@w+bias).sparse_categorical_crossentropy(labels[start:end])
            loss.backward();optim.step()
    final=(x@w+bias).sparse_categorical_crossentropy(labels).item()
    logits=(test_x@w+bias).realize()
    accuracy=(logits.argmax(axis=1)==test_y).cast(dtypes.float32).mean().item()
    with tempfile.TemporaryDirectory(prefix='course-weights-') as directory:
        path=Path(directory)/'weights.safetensors';safe_save({'weight':w,'bias':bias},path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();saved=safe_load(path)
        restored=(test_x@saved['weight'].to('CPU')+saved['bias'].to('CPU')).tolist()
    error=close(restored,logits.tolist())
    if not final < first: raise AssertionError('training did not lower full training loss')
    return {'dataset':dataset,'steps':steps,'training_examples':x.shape[0],'test_examples':test_x.shape[0],
            'first_training_loss':first,'final_training_loss':final,'test_accuracy':accuracy,
            'save_reload_max_error':error,'checkpoint_sha256':digest}


def compiler():
    x=tensor([-.5,.2,1.5]); results={}
    for mode in ('materialized','lazy'):
        GlobalCounters.reset()
        y=x*2
        if mode=='materialized': y.realize()
        y=y+1
        if mode=='materialized': y.realize()
        y=y.relu().realize();Device['CPU'].synchronize()
        count=GlobalCounters.kernel_count
        close(y.tolist(),[0,1.4,4])
        results[mode]={'kernel_count':count,'output':y.tolist()}
    @TinyJit
    def predict(z): return (z*2+1).relu().realize()
    replay=[]
    for data in [[-1,0,2],[1,2,3],[-2,-1,0],[3,0,-3]]:
        out=predict(tensor(data));Device['CPU'].synchronize()
        values=out.tolist();close(values,[max(2*v+1,0) for v in data]);replay.append(values)
    results['changing_input_jit_outputs']=replay
    # IR inspection is kept before execution, without invoking a consuming scheduler.
    source_graph=(x*2+1).relu()
    results['tensor_ir_root']=source_graph.uop.op.name
    return results


def custom_ops():
    values=[.5,-1.,2.]; x=tensor(values)
    norm=x/(x.square().mean()+1e-6).sqrt()
    norm_error=close(norm.tolist(),reference.rmsnorm(values))
    value=tensor([2,3,5]);gate=tensor([-1,0,2]);swiglu=value*gate*gate.sigmoid()
    swiglu_error=close(swiglu.tolist(),reference.swiglu([2,3,5],[-1,0,2]))
    q=[[1,0],[0,1],[1,1]]; k=[[1,0],[0,1],[-1,1]];v=[[2,3],[4,1],[100,200]]
    scores=(tensor(q)@tensor(k).transpose())/math.sqrt(2)
    causal=Tensor([[True,False,False],[True,True,False],[True,True,True]],device='CPU')
    output=causal.where(scores,-math.inf).softmax(axis=-1)@tensor(v)
    error=close(output.tolist(),reference.causal_attention(q,k,v),atol=2e-5)
    close(output.tolist()[0],v[0])
    return {'rmsnorm':norm.tolist(),'rmsnorm_max_error':norm_error,'swiglu':swiglu.tolist(),
            'swiglu_max_error':swiglu_error,'causal_attention':output.tolist(),'attention_max_error':error}


def model_contract():
    # This artifact is a named teaching linear model, not any openpilot neural network.
    path=Path(__file__).resolve().parents[2]/'1. Fundamentals'/'labs'
    sys.path.insert(0,str(path))
    from vehicle_lab import MODEL_CONTRACT, decode_model_output
    values=[[1.,.1],[-1.,-.1],[0.,0.]]
    weight=tensor([[-.08],[-.4]])
    outputs=(tensor(values)@weight).tolist()
    close(outputs,[[-.12],[.12],[0.]])
    decoded=[decode_model_output([row],MODEL_CONTRACT) for row in outputs]
    rejected=False
    try:decode_model_output([[.1]],{**MODEL_CONTRACT,'output_units':'steering_wheel_deg'})
    except ValueError:rejected=True
    if not rejected:raise AssertionError('unit mismatch was accepted')
    return {'manifest':MODEL_CONTRACT,'inputs':values,'decoded_road_wheel_radians':decoded,
            'wrong_units_rejected':rejected,'learned_driving_quality':'not evaluated; fixed illustrative weights'}


LESSONS={'intro':intro,'basics':basics,'operations':operations,'autograd':autograd,
         'classifier':classifier,'compiler':compiler,'custom_ops':custom_ops,'model_contract':model_contract}


def main(default=None):
    parser=argparse.ArgumentParser(description=__doc__)
    if default is None:parser.add_argument('lesson',choices=[*LESSONS,'all'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--mnist',action='store_true',help='explicitly download and run bounded MNIST classifier')
    parser.add_argument('--steps',type=int,default=40)
    args=parser.parse_args();selected=default or args.lesson
    names=LESSONS if selected=='all' else [selected]
    results={name:(classifier(args.mnist,args.steps) if name=='classifier' else LESSONS[name]()) for name in names}
    report={'tinygrad':version('tinygrad'),'python':sys.version.split()[0],'device':'CPU','beam':0,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lessons':results}
    encoded=json.dumps(report,indent=2,allow_nan=False)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded+'\n')
    print(encoded)


if __name__=='__main__':main()
