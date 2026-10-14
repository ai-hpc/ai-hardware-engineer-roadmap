"""Original teaching figures. Run with matplotlib; emits editable SVG and 180-dpi PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':12})
INK='#15243a'; MUTED='#516478'; BLUE='#245cbe'; TEAL='#087e83'; GOLD='#a56808'

def canvas(title,subtitle,height=8):
 fig,ax=plt.subplots(figsize=(14,height)); fig.patch.set_facecolor('#f5f8fc');ax.set_facecolor('#f5f8fc')
 ax.set_xlim(0,14);ax.set_ylim(0,height);ax.axis('off')
 ax.text(.35,height-.45,title,fontsize=23,fontweight='bold',color=INK,va='top')
 ax.text(.35,height-1,subtitle,fontsize=11,color=MUTED,va='top')
 return fig,ax

def box(ax,x,y,w,h,title,body,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.12',facecolor='white',edgecolor='#c8d5e4',lw=1.3))
 ax.add_patch(Rectangle((x+.02,y+.08),.055,h-.16,facecolor=color,lw=0))
 ax.text(x+.18,y+h-.22,title,color=color,fontsize=13,fontweight='bold',va='top')
 ax.text(x+.18,y+h-.61,body,color=INK,fontsize=11.2,va='top',linespacing=1.55)

def arrow(ax,a,b,color=MUTED):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color=color,lw=1.7))

def save(fig,name):
 fig.subplots_adjust(left=.025,right=.985,bottom=.025,top=.98)
 for ext in ('svg','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor=fig.get_facecolor())
 plt.close(fig)

fig,ax=canvas('One mission, nine engineering gates','Arrows carry reviewed artifacts. Return upstream when a later measurement breaks an assumption.',9)
items=[('01  Specify the robot task','Task, units, frames, timestamps\nDeadline and fallback contract',BLUE),('02  Build the model reference','Data split + preprocessing\nActions + held-out evaluation',BLUE),('03  Validate the robot loop','Simulation, resets, action adapter\nObservation age + task success',TEAL),('04  Optimize on real hardware','Quality gate + measured trace\nMemory, latency, energy budget',TEAL),('05  Lower the workload','Graph + legal transformations\nLayouts, buffers, runtime ABI',BLUE),('06  Prototype the accelerator','Reference vectors + RTL\nDMA/backpressure + timing',GOLD),('07  Integrate Linux and I/O','Boot image + driver + sensors\nBuffer ownership + recovery',GOLD),('08  Add wireless deliberately','Command envelope + freshness\nLocal control during link loss',TEAL),('09  Defend the system design','End-to-end evidence + limits\nFPGA results / ASIC estimates',BLUE)]
coords=[]
for i,(title,body,color) in enumerate(items):
 row,col=divmod(i,3);x=.4+col*4.55;y=5.7-row*2.1;box(ax,x,y,4.05,1.65,title,body,color);coords.append((x,y))
 if col:arrow(ax,(x-.48,y+.83),(x-.08,y+.83))
for row in range(2):
 x,y=coords[row*3+2]; ax.plot([x+4.15,13.78,13.78,.22,.22],[y+.8,y+.8,y-.22,y-.22,y-1.25],color=MUTED,lw=1.3)
 arrow(ax,(.22,y-1.25),(.4,y-1.25))
ax.text(.4,.52,'Start with one operator and a simulated task. A full custom chip is the destination, not the first lab.',color=MUTED,fontsize=12)
save(fig,'mission-gates')

fig,ax=canvas('A VLA proposes actions; the local loop owns execution','Use separate contracts for observation age, policy updates, control ticks, and wireless commands.',7.8)
box(ax,.4,3.45,2.9,1.65,'SENSE','Camera + proprioception\nTimestamp + frame ID',BLUE)
box(ax,3.8,3.45,2.9,1.65,'INFER','VLA + action adapter\nChunk + source time',BLUE)
box(ax,7.2,3.45,2.9,1.65,'CHECK','Age, units, bounds\nFresh action or fallback',TEAL)
box(ax,10.6,3.45,2.9,1.65,'CONTROL','Local control tick\nActuator + plant',GOLD)
for x in [3.3,6.7,10.1]:arrow(ax,(x+.03,4.25),(x+.46,4.25))
ax.plot([12.1,12.1,1.85,1.85],[3.4,2.6,2.6,3.35],color=TEAL,lw=2)
arrow(ax,(1.85,2.6),(1.85,3.4),TEAL)
ax.text(6.8,2.76,'new observations close the loop',ha='center',color=TEAL)
box(ax,3.8,.35,6.3,1.6,'OPTIONAL NETWORK / LANGUAGE PLANNER','Task intent, logs, updates — with expiry and authentication.\nLink loss must not stall the local control tick.',TEAL)
arrow(ax,(5.25,1.98),(5.25,3.37))
ax.text(.45,5.6,'A lower model latency helps only if the accepted action is still useful when applied.',color=INK,fontsize=13)
save(fig,'robot-loop')

fig,ax=canvas('Count critical-path time, not overlapping durations','Analytical example: two independent tiles; each copy takes 6 ms and each compute takes 4 ms.',7.2)
origin=.7;scale=.59
for label,y,bars in [('Serial',4.45,[(0,6,'copy 0',TEAL),(6,4,'compute 0',BLUE),(10,6,'copy 1',TEAL),(16,4,'compute 1',BLUE)]),('Copy engine',2.9,[(0,6,'copy 0',TEAL),(6,6,'copy 1',TEAL)]),('Compute engine',1.95,[(6,4,'compute 0',BLUE),(12,4,'compute 1',BLUE)])]:
 ax.text(origin,y+.69,label,color=INK,fontweight='bold',fontsize=12)
 for start,duration,text,color in bars:
  ax.add_patch(FancyBboxPatch((origin+start*scale,y),duration*scale-.06,.52,boxstyle='round,pad=.015',facecolor=color,edgecolor='none'))
  ax.text(origin+(start+duration/2)*scale,y+.26,text,ha='center',va='center',color='white',fontsize=11)
for t in range(0,21,2):
 x=origin+t*scale;ax.plot([x,x],[1.67,1.78],color=MUTED);ax.text(x,1.42,str(t),ha='center',color=MUTED,fontsize=10)
ax.text(13,1.42,'ms',color=MUTED)
ax.text(.7,.83,'Serial: 2 × (6 + 4) = 20 ms       Overlap: 6 + 4 + max(6, 4) = 16 ms',color=INK,fontsize=14,fontweight='bold')
ax.text(.7,.36,'Requires separate engines, independent buffers, and supported asynchronous execution. These are not benchmark results.',color=MUTED,fontsize=10.5)
save(fig,'critical-path')

fig,ax=canvas('Four-bit values do not mean a four-bit tensor','A teaching example of one-dimensional NVFP4 block storage; actual packing and scale layout belong to the backend.',7.4)
box(ax,.45,3.8,6.1,1.75,'16 E2M1 VALUES','16 × 4 bits = 64 bits = 8 bytes\nMagnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6',BLUE)
box(ax,7.05,3.8,3.05,1.75,'BLOCK SCALE','1 E4M3 value\n8 bits = 1 byte',TEAL)
box(ax,10.6,3.8,2.9,1.75,'GLOBAL SCALE','FP32, tensor-wide\n32 bits = 4 bytes',GOLD)
ax.text(6.8,4.66,'+',ha='center',fontsize=24,color=MUTED);ax.text(10.35,4.66,'+',ha='center',fontsize=24,color=MUTED)
ax.text(.6,2.94,'Reconstruction: value = E2M1 payload × block scale × tensor scale',fontsize=17,color=INK)
ax.text(.6,2.15,'Per 16-value block: 9 bytes → 4.5 bits/value, before the shared global scale.',fontsize=15,color=INK)
ax.text(.6,1.48,'For N values, when N is divisible by 16:  bytes = N/2 + N/16 + 4',fontsize=15,color=INK)
ax.text(.6,.66,'Then account for padding, scale swizzles, unquantized tensors, runtime workspaces, and calibration quality.',fontsize=11.5,color=MUTED)
save(fig,'nvfp4-storage')
