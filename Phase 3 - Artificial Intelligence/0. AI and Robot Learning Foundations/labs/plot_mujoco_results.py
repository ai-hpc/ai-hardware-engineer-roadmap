"""Plot recorded CPU dynamics results; requires Matplotlib, not MuJoCo."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={
 'matched_100ms':'Matched coarse step',
 'substeps_10ms':'Fine step, correct mass',
 'mass3_wrong_adapter':'Wrong mass conversion',
 'mass3_correct_adapter':'3 kg, corrected conversion',
 'mass3_force_limited':'3 kg, limited force',
 'damping_without_compensation':'Added joint damping',
}
COLORS=['#617585','#1672a6','#bf5200','#14826c','#8d498b','#665da4']

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();r=json.loads(args.input.read_text())
 plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
 fig,(ax,bar)=plt.subplots(2,1,figsize=(9.2,7.7),layout='constrained',gridspec_kw={'height_ratios':[1.3,1]})
 for i in (1,2):
  case=r['cases'][i];trace=case['first_episode_trace']
  ax.plot([x['time_s'] for x in trace],[x['position_m'] for x in trace],color=COLORS[i],label=LABELS[case['config']['name']],linewidth=2)
 ax.axhspan(-.1,.1,color='#14826c',alpha=.07,label='Position tolerance only')
 ax.axhline(0,color='#8c969e',linewidth=.8)
 ax.set(xlabel='Simulated time (s)',ylabel='Cart position (m)',title='One paired episode: changed force semantics change the motion')
 ax.grid(alpha=.18);ax.legend(fontsize=9,loc='best')
 values=[x['mean_abs_position_m'] for x in r['cases']]
 bars=bar.barh(range(6),values,color=COLORS)
 bar.set_yticks(range(6),[LABELS[x['config']['name']] for x in r['cases']]);bar.invert_yaxis()
 bar.set(xlabel='Mean absolute position over twelve episodes (m)',title='Final success alone hides different transients',xlim=(0,max(values)*1.35))
 for b,case in zip(bars,r['cases']):
  bar.text(b.get_width()+.003,b.get_y()+b.get_height()/2,f'{case["successes"]}/12 final success',va='center',fontsize=9)
 bar.grid(axis='x',alpha=.18)
 fig.suptitle('Actual MuJoCo CPU simulation · synthetic sliding cart · same saved policy',fontsize=12)
 args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output);plt.close(fig)

if __name__=='__main__':main()
