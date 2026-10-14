"""Render original, editable course diagrams; requires cairosvg for PNG export."""
from pathlib import Path
from html import escape
import cairosvg
ROOT = Path(__file__).resolve().parent

def text(x, y, label, size=18, fill='#334155', weight='400', anchor='start'):
    return f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(label)}</text>'

def box(x, y, w, h, title, lines, color='#2563eb'):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="white" stroke="{color}" stroke-width="2"/>'
    out += f'<rect x="{x}" y="{y}" width="7" height="{h}" rx="3" fill="{color}"/>'
    out += text(x+22, y+36, title, 22, '#0f172a', '700')
    for i, line in enumerate(lines): out += text(x+22,y+66+26*i,line,17)
    return out

def arrow(x1,y1,x2,y2,color='#64748b',dash=False):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#arrow)"'+(' stroke-dasharray="8 6"' if dash else '')+'/>'

def save(name, title, desc, content, height):
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto-start-reverse"><path d="M1 1 L8 5 L1 9" fill="none" stroke="#64748b" stroke-width="1.5"/></marker></defs><rect width="1200" height="{height}" fill="#f8fafc"/>{content}</svg>'''
    (ROOT/f'{name}.svg').write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(ROOT/f'{name}.png'),output_width=1800)

s=text(48,56,'One observation, two contracts',34,'#0f172a','700')
s+=text(48,88,'The compiler describes tensor work. The system decides when a result is still usable.',18)
s+=text(48,136,'DATA PATH · buffers, shapes, and completion',17,'#1d4ed8','700')
s+=box(48,156,240,150,'Capture',['Sequence + timestamp','Format + stride','Buffer owner'])
s+=box(336,156,240,150,'Prepare',['Layout + normalization','Shape + dtype','Reference tensor'])
s+=box(624,156,240,150,'Accelerate',['Compiled operators','DMA + scratchpad','Completion event'])
s+=box(912,156,240,150,'Decode action',['Units + coordinate frame','Valid range + state','Output lifetime'])
for a,b in [(288,336),(576,624),(864,912)]:s+=arrow(a+5,228,b-8,228)
s+=text(48,346,'CONTROL PATH · age, deadlines, and recovery',17,'#047857','700')
s+=box(48,390,336,143,'Linux runtime',['Own queues and buffer lifetimes','Trace submission → completion','Handle timeout / device reset'],'#059669')
s+=box(432,390,336,143,'Control task / MCU',['Validate sequence and freshness','Apply limits and watchdog policy','Reject stale or invalid commands'],'#059669')
s+=box(816,390,336,143,'Wireless + management',['Configuration and telemetry','Reconnect and update state','Explicit access to command path'],'#059669')
s+=arrow(1034,314,600,382,dash=True)
s+=arrow(392,459,424,459);s+=arrow(808,459,776,459,dash=True)
s+=text(48,587,'Read the solid arrows as data or validated control flow; dashed arrows are coordination.',18)
s+=text(48,621,'Exercise: mark who may read, write, free, or reuse every buffer at each boundary.',19,'#0f172a','700')
save('observation-contract','One observation, two contracts','A timestamped camera observation passes through preparation, an accelerator, and action decoding. Linux manages buffers; a separate control task validates freshness and limits. Wireless is a managed control-plane input.',s,655)
s=text(48,55,'A Thread mesh is a routed IPv6 network',32,'#0f172a','700')
s+=text(48,90,'Role, radio behavior, and application meaning are separate design decisions.',18)
s+=box(48,170,274,132,'IP application host',['Linux service / dashboard','Records age + sequence'],'#7c3aed')
s+=box(450,170,310,132,'Border router',['Routes between IP networks','Router role shown here'],'#2563eb')
s+=box(877,170,275,132,'Thread router',['Forwards mesh packets','One router is leader'],'#2563eb')
s+=arrow(330,235,442,235);s+=arrow(768,235,869,235)
s+=text(355,220,'IPv6',16,'#475569','600');s+=text(778,220,'mesh',16,'#475569','600')
s+=box(451,407,309,140,'Sleepy end device',['Polls parent for messages','Does not route for others','Report includes freshness'],'#059669')
s+=box(876,407,275,140,'Another router',['Provides another path','Can serve child nodes'],'#2563eb')
s+=arrow(606,399,606,310);s+=arrow(1012,399,1012,310)
s+=arrow(768,279,868,445)
s+=text(624,360,'parent / child',16,'#475569','600')
s+=text(48,374,'LEARNER CHECK',17,'#7c3aed','700')
s+=text(48,408,'A ping tests reachability.',19)
s+=text(48,443,'An application report tests',19)
s+=text(48,470,'units, identity, and age.',19)
s+=text(48,596,'Failure experiment: restart a parent; record loss, reattachment, duplicates, and stale reports.',18,'#0f172a','600')
s+=text(48,630,'This is a conceptual topology. RCP/NCP placement and specific routes depend on the implementation.',16)
save('thread-role-map','Thread mesh roles and application freshness','An IP application host connects through a border router to a Thread router mesh. A sleepy end device attaches to a parent and does not route for others. The diagram distinguishes IP reachability from valid application data.',s,666)
print('Rendered two SVGs and PNGs')
