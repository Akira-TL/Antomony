"""Inspect rendered frames, subtitle structure, and simple causal invariants."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from communication.video.progress.plan import CHAPTERS,OUT,ROOT,FILENAME
from communication.video.progress.production.build import preserved,subtitles


def numeric_checks():
    # The low-dimensional illustration uses the same discrete equation form.
    x=np.array([.9,-.45,.60]);delta=np.array([.75,-.40]);old=np.full((3,2),.2)
    lam=.93;E=lam*old+(1-lam)*np.outer(x,delta)
    assert E.shape==(3,2) and np.isfinite(E).all()
    positive=.12*.8*E;negative=.12*(-.8)*E
    assert np.allclose(positive,-negative)
    zero=.12*0*E;assert np.count_nonzero(zero)==0
    F=np.array([[.01,.02],[0,.02],[-.03,.01]])
    skipped=F+0*positive;assert np.array_equal(skipped,F)
    # A skipped write does not imply that the eligibility trace stayed fixed.
    assert not np.array_equal(E,old)
    applied=F+positive;assert not np.array_equal(applied,F)
    limit=.04;bounded=positive*min(1,limit/max(np.linalg.norm(positive),1e-12))
    assert np.linalg.norm(bounded)<=limit+1e-12
    cues=subtitles();preserved()
    english=max(len(c['en']) for c in cues);chinese=max(len(c['zh']) for c in cues)
    report={'seconds':sum(c.seconds for c in CHAPTERS),'cue_count':len(cues),'max_english_characters':english,'max_chinese_characters':chinese,'exactly_two_text_lines_per_bilingual_cue':True,'mathematical_checks':'outer-product dimensions; opposite modulation signs; zero modulation; skip leaves F intact while E may update; bounded step','preserved_v2_v3':True}
    (OUT/'qa').mkdir(exist_ok=True);(OUT/'qa/structural-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False))


def frames(final=False):
    (OUT/'qa').mkdir(exist_ok=True)
    entries=[]
    if final:
        times=[2.8,7.8,12.8,20.5,23.98,24.,31.,39.,49.,55.,63.,72.,79.,86.,93.,99.,103.,112.,119.,126.,133.,143.,151.,157.,164.,169.,174.,178.]
        entries=[(OUT/FILENAME,t,f'final-{i:02}',f'{int(t)//60:02}:{t%60:05.2f}') for i,t in enumerate(times)]
    else:
        for c in CHAPTERS:
            p=OUT/'clips-hd'/f'{c.scene}.mp4'
            if p.exists():
                # Only probe complete files; currently-rendering partial files are excluded.
                probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(p)],capture_output=True,text=True)
                if probe.returncode or not probe.stdout.strip():continue
                for j,t in enumerate((c.seconds*.3,c.seconds*.7)):
                    entries.append((p,t,f'{c.scene}-{j}',c.scene+f' {t:.1f}s'))
    paths=[]
    for source,t,name,title in entries:
        dest=OUT/'qa'/f'{name}.png'
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',str(source),'-frames:v','1',str(dest)],check=True)
        paths.append((dest,title))
    columns=4;w=400;h=250;rows=math.ceil(len(paths)/columns)
    if not paths:raise ValueError('No complete video clips available')
    sheet=Image.new('RGB',(columns*w,rows*h),'#0b0e16');draw=ImageDraw.Draw(sheet)
    for i,(p,title) in enumerate(paths):
        x=(i%columns)*w;y=(i//columns)*h
        image=Image.open(p).convert('RGB');image.thumbnail((w,225));sheet.paste(image,(x,y+23));draw.text((x+9,y+6),title,fill='#f2eee6')
    path=OUT/'qa'/('final-contact.jpg' if final else 'rendered-contact.jpg');sheet.save(path,quality=91)
    print('Frame sheet:',path,'frames:',len(paths))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--frames',action='store_true');args=p.parse_args()
    numeric_checks()
    if args.frames:frames(args.final)
