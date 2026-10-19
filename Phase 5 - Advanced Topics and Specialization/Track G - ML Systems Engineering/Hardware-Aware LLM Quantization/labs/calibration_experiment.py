"""Train a small real PyTorch network and evaluate three CPU FP4 references."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import torch
from fp4_reference import pack_fp4, unpack_fp4, select_clipping

ROOT=Path(__file__).resolve().parents[4]
ROBOT=ROOT/'Phase 3 - Artificial Intelligence/2. Deep Learning Frameworks/PyTorch/CPU-Robot-Stop-Lab/labs'
# Import the existing fixture and network rather than maintaining a second dataset.
sys.path.insert(0,str(ROBOT))
import train_stop_model as robot
from data import make_dataset, split_episodes, fit_scaler, sensor_shift, dataset_sha256
sys.path.pop(0)


def prepare(seed=31,epochs=40):
    if not 1<=epochs<=100:
        raise ValueError('epochs must be 1..100')
    torch.manual_seed(seed); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rows=make_dataset(); pool,development,test=split_episodes(rows)
    calibration_ids=set(sorted({r.episode for r in pool})[-16:])
    training=[r for r in pool if r.episode not in calibration_ids]
    calibration=[r for r in pool if r.episode in calibration_ids]
    mean,std=fit_scaler(training); model=robot.StopModel(mean,std)
    x,y=robot.tensors(training); dx,dy=robot.tensors(development)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    best_loss=float('inf');best_state=None;best_epoch=None
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(training)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x[indices]),y[indices])
            loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode():value=float(torch.nn.functional.cross_entropy(model(dx),dy))
        if value<best_loss:
            best_loss=value;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);model.eval()
    return model, {'training':training,'calibration':calibration,'development':development,'test':test},best_epoch


def quantize_model(reference,calibration,objective):
    if objective not in ('absmax','weight_mse','output_mse'):
        raise ValueError('unsupported calibration objective')
    model=copy.deepcopy(reference).eval(); activations={};handles=[]
    def record(name):
        def hook(module,inputs,output):activations[name]=inputs[0].detach().clone().reshape(-1,module.in_features)
        return hook
    for name,module in reference.named_modules():
        if isinstance(module,torch.nn.Linear):handles.append(module.register_forward_hook(record(name)))
    try:
        with torch.inference_mode():reference(robot.tensors(calibration)[0])
    finally:
        for h in handles:h.remove()
    artifacts={};choices={};quantized_names=set()
    with torch.no_grad():
        for name,module in model.named_modules():
            if not isinstance(module,torch.nn.Linear):continue
            if objective=='absmax':
                artifact=pack_fp4(module.weight);decision={'selected_ratio':1.}
            else:
                artifact,decision=select_clipping(module.weight,activations[name] if objective=='output_mse' else None)
            module.weight.copy_(unpack_fp4(artifact));artifacts[name]=artifact;choices[name]=decision;quantized_names.add(name+'.weight')
    retained=sum(v.numel()*v.element_size() for k,v in reference.state_dict().items() if k not in quantized_names)
    payload=sum(a.payload_bytes for a in artifacts.values())+retained
    return model,artifacts,{'layers':choices,'packed_reference_payload_bytes':payload,'retained_wide_bytes':retained}


def summary(reference,candidate,rows):
    with torch.inference_mode():
        x,_=robot.tensors(rows);lp=reference(x).log_softmax(-1);lq=candidate(x).log_softmax(-1)
        result=robot.evaluate(candidate,rows,.5)  # fixed threshold, never retuned on test
        result.pop('errors');result['reference_kl_nats']=float((lp.exp()*(lp-lq)).sum(-1).mean())
        result['reference_logit_mse']=float((reference(x)-candidate(x)).square().mean())
        return result


def run(output,epochs=40):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reference,parts,best_epoch=prepare(epochs=epochs)
    # Fit all policies first. Test and shifted test have not entered calibration.
    policies={name:quantize_model(reference,parts['calibration'],name) for name in ('absmax','weight_mse','output_mse')}
    shifts=sensor_shift(parts['test'],.8)
    report={'scope':'CPU reference dequantization into float32; not native FP4 or real robot validation',
            'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':torch.get_num_threads(),
            'seed':31,'epochs':epochs,'selected_epoch':best_epoch,'threshold':.5,'sensor_shift_m':.8,
            'dataset_sha256':dataset_sha256(make_dataset()),
            'split_episode_ids':{k:sorted({r.episode for r in rows}) for k,rows in parts.items()},
            'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROBOT/'data.py',ROBOT/'train_stop_model.py',Path(__file__),Path(__file__).with_name('fp4_reference.py')]},
            'reference_payload_bytes':sum(v.numel()*v.element_size() for v in reference.state_dict().values()),
            'reference':{'test':summary(reference,reference,parts['test']),'shifted_test':summary(reference,reference,shifts)},'policies':{}}
    for name,(candidate,artifacts,details) in policies.items():
        # Save tensors/metadata only, with scales encoded as bytes for robust loading.
        saved={k:{'codes':a.codes,'scale_bytes':a.scales.view(torch.uint8),'global_scale':a.global_scale,'shape':a.shape,'padded_cols':a.padded_cols} for k,a in artifacts.items()}
        target=output/(name+'.pt');torch.save({'packed':saved,'retained':{k:v for k,v in reference.state_dict().items() if k not in {n+'.weight' for n in artifacts}}},target)
        loaded=torch.load(target,map_location='cpu',weights_only=True)
        from fp4_reference import PackedFP4
        restored=copy.deepcopy(reference)
        with torch.no_grad():
            for k,a in loaded['packed'].items():
                decoded=unpack_fp4(PackedFP4(a['codes'],a['scale_bytes'].view(torch.float8_e4m3fn),a['global_scale'],a['shape'],a['padded_cols']))
                restored.get_submodule(k).weight.copy_(decoded)
            for k,v in loaded['retained'].items():restored.state_dict()[k].copy_(v)
            tx,_=robot.tensors(parts['test']);reload_error=float((candidate(tx)-restored(tx)).abs().max())
        if reload_error!=0:raise AssertionError('packed artifact reload changed outputs')
        report['policies'][name]={**details,'serialized_file_bytes':target.stat().st_size,'artifact_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'reload_max_abs_logit_error':reload_error,
                                 'test':summary(reference,candidate,parts['test']),'shifted_test':summary(reference,candidate,shifts)}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args();report=run(args.output,args.epochs)
    print(json.dumps({'reference':{'payload_bytes':report['reference_payload_bytes'],'test_accuracy':report['reference']['test']['accuracy']},'policies':{k:{'payload_bytes':v['packed_reference_payload_bytes'],'file_bytes':v['serialized_file_bytes'],'test_accuracy':v['test']['accuracy'],'shifted_stop_recall':v['shifted_test']['stop_recall'],'ratios':{n:d['selected_ratio'] for n,d in v['layers'].items()}} for k,v in report['policies'].items()}},indent=2))
