"""Assemble a clean v4 master and independent bilingual editing assets."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import wave
import zipfile
from communication.video.progress.plan import CHAPTERS,ROOT,OUT,FPS,RATE,FILENAME
from communication.video.production.assemble import metadata


def run(args):
    result=subprocess.run(args,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(str(args[:3])+'\n'+result.stderr[-5000:])


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def preserved():
    provenance=json.loads((ROOT/'communication/video/progress/provenance.json').read_text())
    for master in provenance['preserved_masters']:
        if sha(ROOT/master['path'])!=master['sha256']:raise RuntimeError('Old movie changed: '+master['path'])
    for name,digest in provenance['preserved_sources'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Old video source changed: '+name)
    return provenance


def subtitles():
    path=OUT/'Learning_While_Acting_v4.en-zh.srt'
    blocks=path.read_text(encoding='utf-8-sig').strip().split('\n\n')
    expected=[s for c in CHAPTERS for s in c.sentences]
    if len(blocks)!=len(expected):raise RuntimeError('Caption count mismatch')
    for block,sentence in zip(blocks,expected):
        lines=block.splitlines()
        if len(lines)!=4 or lines[2]!=sentence.en or lines[3]!=sentence.zh:
            raise RuntimeError('Bilingual SRT must contain exactly English then Chinese')
    cues=json.loads((OUT/'speech-visual-timeline.json').read_text())
    previous=0
    for cue in cues:
        if not previous<=cue['start']<cue['end']<=180:raise RuntimeError('Caption overlap or out-of-bounds cue')
        previous=cue['end']
    return cues


def check():
    preserved();cues=subtitles()
    data=metadata(OUT/FILENAME)
    types=sorted(s['codec_type'] for s in data['streams'])
    if types!=['audio','video']:raise RuntimeError('Movie must have one audio and one video stream, no subtitles')
    v=next(s for s in data['streams'] if s['codec_type']=='video')
    a=next(s for s in data['streams'] if s['codec_type']=='audio')
    if (v.get('width',0),v.get('height',0),v.get('r_frame_rate',''),int(v.get('nb_frames','0')))!=(1920,1080,'60/1',10800):raise RuntimeError('Unexpected video geometry or frame count')
    if abs(float(data['format']['duration'])-180)>.04 or abs(float(a.get('duration','0'))-180)>.04:raise RuntimeError('Wrong media duration')
    with wave.open(str(OUT/'Narration.en.wav')) as w:
        if w.getframerate()!=RATE or w.getnframes()!=180*RATE:raise RuntimeError('Wrong narration sample count')
    silent=metadata(OUT/'Visuals_v4_silent.mp4')
    if [s['codec_type'] for s in silent['streams']]!=['video']:raise RuntimeError('Silent edit asset contains extra streams')
    run(['ffmpeg','-v','error','-i',str(OUT/FILENAME),'-f','null','-'])
    print(f'PASS: 180s, 10800 frames, 1080p60, full decode; {len(cues)} EN/ZH two-line cues; v2/v3 preserved.')


def player():
    cues=subtitles();offset=0;buttons=[]
    for c in CHAPTERS:
        buttons.append(f'<button data-time="{offset}"><span>{offset//60:02}:{offset%60:02}</span>{html.escape(c.title)}</button>');offset+=c.seconds
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learning While Acting | v4</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b0e16;color:#f2eee6;font-family:system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:28px}small{color:#57d6bd;letter-spacing:2px}h1{font-size:38px;font-weight:500}p,footer{color:#a4a9b5;line-height:1.6}.player{position:relative;background:#0b0e16}video{display:block;width:100%;max-height:76vh}.captions{position:absolute;bottom:9%;left:2%;width:96%;text-align:center;pointer-events:none;text-shadow:0 2px 5px #000,0 0 3px #000;font-weight:500;display:none}.captions div{white-space:nowrap;font-size:clamp(12px,1.8vw,26px);line-height:1.4}.captions .zh{font-size:clamp(12px,1.65vw,24px)}label{display:block;margin:16px 0;color:#a4a9b5}.chapters{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:8px}button{display:flex;gap:14px;background:#151c28;border:1px solid #2c3849;color:#f2eee6;padding:13px;text-align:left;font:14px system-ui;cursor:pointer}button:hover{border-color:#57d6bd}button span{color:#57d6bd;white-space:nowrap}a{color:#70b8e8}footer{font-size:13px;margin-top:24px}
</style></head><body><main><small>MATHHACKSON / RESEARCH CUT 04</small><h1>Learning While Acting</h1><p>3:00 · 1080p60 · English narration · Clean master + separate bilingual subtitles</p><div class="player"><video id="film" controls preload="metadata" src="__FILE__"></video><div id="captions" class="captions"><div id="en"></div><div id="zh" class="zh"></div></div></div><label><input type="checkbox" id="toggle"> Preview sidecar subtitles: English above Chinese (off by default)</label><div class="chapters">__CHAPTERS__</div><footer>Moving ants and local-mechanism scenes are explanatory illustrations, not footage from the research engine.<br>The recorded-results scene uses the completed continuous-adaptation experiment: 4 world seeds, 3 changed conditions, one initialization set. It is not an outcome claim for the new plasticity course. The latest paired-baseline analysis stopped at a validity audit.<br>Online local adaptation and outer training are distinct; the latter can use backpropagation. Real-time behavior is a design target, not a measured hardware guarantee.<br>English synthetic voice: Microsoft en-US-AndrewNeural. Original artwork; no copied 3Blue1Brown footage or voice clone.<br><a href="Learning_While_Acting_v4.en-zh.srt" download>Bilingual SRT</a> · <a href="Learning_While_Acting_v4.en.srt" download>English SRT</a> · <a href="Learning_While_Acting_v4.zh.srt" download>Chinese SRT</a> · <a href="Visuals_v4_silent.mp4" download>Silent visuals</a> · <a href="Narration.en.wav" download>Narration WAV</a> · <a href="ManimGL_sources_v4.zip" download>Sources</a> · <a href="outline.md">Outline</a></footer></main><script>
const cues=__CUES__;const film=document.getElementById('film'),cap=document.getElementById('captions'),toggle=document.getElementById('toggle');function update(){const c=cues.find(c=>film.currentTime>=c.start&&film.currentTime<c.end);cap.style.display=toggle.checked&&c?'block':'none';document.getElementById('en').textContent=c?c.en:'';document.getElementById('zh').textContent=c?c.zh:'';}film.addEventListener('timeupdate',update);toggle.addEventListener('change',update);document.querySelectorAll('[data-time]').forEach(b=>b.onclick=()=>{film.currentTime=Number(b.dataset.time);film.play();film.scrollIntoView({behavior:'smooth',block:'center'});});
</script></body></html>'''
    page=page.replace('__FILE__',FILENAME).replace('__CHAPTERS__','\n'.join(buttons)).replace('__CUES__',json.dumps(cues,ensure_ascii=False).replace('</','<\\/'))
    (OUT/'index.html').write_text(page,encoding='utf-8')


def assemble():
    provenance=preserved();subtitles()
    normalized=OUT/'normalized-hd';normalized.mkdir(exist_ok=True)
    receipts=[];listing=[]
    for c in CHAPTERS:
        source=OUT/'clips-hd'/f'{c.scene}.mp4';target=normalized/source.name
        info=metadata(source);v=next(s for s in info['streams'] if s['codec_type']=='video')
        if (v.get('width',0),v.get('height',0),v.get('r_frame_rate',''))!=(1920,1080,'60/1'):raise RuntimeError(f'Bad geometry: {source}')
        if abs(float(info['format']['duration'])-c.seconds)>2/FPS:raise RuntimeError(f'Bad scene duration: {source}')
        run(['ffmpeg','-y','-v','error','-i',str(source),'-an','-vf',f'trim=end_frame={c.seconds*FPS},setpts=PTS-STARTPTS','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(target)])
        count=int(metadata(target)['streams'][0].get('nb_frames','0'))
        if count!=c.seconds*FPS:raise RuntimeError('Wrong normalized frame count')
        receipts.append({'scene':c.scene,'seconds':c.seconds,'frames':count,'source_sha256':sha(source),'sha256':sha(target)})
        listing.append(f"file '{target}'\n")
        print('Normalized',c.scene,flush=True)
    concat=OUT/'concat-hd.txt';concat.write_text(''.join(listing))
    silent=OUT/'Visuals_v4_silent.mp4'
    run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-map','0:v:0','-c:v','copy','-an','-movflags','+faststart',str(silent)])
    run(['ffmpeg','-y','-v','error','-i',str(silent),'-i',str(OUT/'Narration.en.wav'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','160k','-movflags','+faststart','-metadata','title=Learning While Acting | Research cut v4',str(OUT/FILENAME)])
    check()
    manifest={'title':'Learning While Acting','version':'v4','seconds':180,'fps':FPS,'frames':10800,'dimensions':[1920,1080],'captions':'external UTF-8 SRT; exactly English then Chinese per cue','narration':'synthetic English en-US-AndrewNeural','status':'illustrated mechanism; recorded continuous-trial evidence separately labelled','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sha256':sha(OUT/FILENAME),'bytes':(OUT/FILENAME).stat().st_size,'chapters':receipts,'research_provenance':provenance}
    (OUT/'render-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for name in ('outline.md','narration-bilingual.md','provenance.json'):shutil.copy2(ROOT/'communication/video/progress'/name,OUT/name)
    player();print('FINAL:',OUT/FILENAME,flush=True)


def package():
    output=OUT/'ManimGL_sources_v4.zip'
    names=subprocess.check_output(['git','ls-files','communication/video','scripts/video'],cwd=ROOT,text=True).splitlines()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            p=ROOT/name
            if p.is_file() and p.suffix in {'.py','.sh','.md','.txt','.json'} and 'rendered' not in p.parts:
                archive.write(p,name)
    print('Source package:',output)


def review():
    check();package();player()
    profile=subprocess.check_output(['cmd.exe','/c','echo %USERPROFILE%'],text=True,stderr=subprocess.DEVNULL).strip()
    user_root=Path(subprocess.check_output(['wslpath','-u',profile],text=True).strip())
    downloads=user_root/'Downloads'
    if not downloads.is_dir():raise RuntimeError('Windows Downloads was not resolved')
    destination=downloads/'MathHackson_Learning_While_Acting_v4'
    if destination.exists():destination=downloads/(destination.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    destination.mkdir()
    names=[FILENAME,'Visuals_v4_silent.mp4','Narration.en.wav','Learning_While_Acting_v4.en-zh.srt','Learning_While_Acting_v4.en.srt','Learning_While_Acting_v4.zh.srt','index.html','render-manifest.json','speech-visual-timeline.json','ManimGL_sources_v4.zip','outline.md','narration-bilingual.md','provenance.json']
    for name in names:shutil.copy2(OUT/name,destination/name)
    if sha(destination/FILENAME)!=sha(OUT/FILENAME):raise RuntimeError('Windows copy mismatch')
    windows=subprocess.check_output(['wslpath','-w',str(destination/'index.html')],text=True).strip()
    opened=subprocess.run(['cmd.exe','/c','start','',windows],cwd=str(user_root),capture_output=True,text=True)
    receipt={'directory':str(destination),'windows_index':windows,'open_returncode':opened.returncode,'movie_sha256':sha(destination/FILENAME)}
    (OUT/'delivery-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    if opened.returncode:print('The files were copied, but automatic opening failed:',opened.stderr)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['assemble','check','review','package']);args=p.parse_args()
    {'assemble':assemble,'check':check,'review':review,'package':package}[args.action]()
