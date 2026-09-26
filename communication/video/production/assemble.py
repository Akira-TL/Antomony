"""Frame-exact clean-video assembly. Subtitles are sidecars, never video filters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import TypedDict, Required, cast

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from communication.video.film_plan import CHAPTERS

OUT=ROOT/'communication/video/rendered/v2'


@dataclass(frozen=True)
class ClipCheck:
    scene: str
    width: int
    height: int
    fps: int
    frames: int
    seconds: int


def run(args:list[str])->None:
    subprocess.run(args,check=True,capture_output=True)


class StreamInfo(TypedDict, total=False):
    codec_type: Required[str]
    width: int
    height: int
    r_frame_rate: str
    nb_frames: str
    duration: str


class FormatInfo(TypedDict):
    duration: str


class ProbeInfo(TypedDict):
    streams: list[StreamInfo]
    format: FormatInfo


def metadata(path:Path)->ProbeInfo:
    return cast(ProbeInfo,json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],text=True)))


def player(filename:str)->None:
    seconds=0
    buttons=[]
    for ch in CHAPTERS:
        buttons.append(f'<button data-time="{seconds}"><time>{seconds//60:02}:{seconds%60:02}</time>{ch.title}</button>')
        seconds+=ch.seconds
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learning While Acting — Moving Ants</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b0e16;color:#f2eee6;font-family:system-ui,sans-serif}main{max-width:1440px;padding:30px;margin:auto}h1{font-size:36px;font-weight:500;margin:12px 0}p{color:#a4a9b5;line-height:1.6}small{color:#57d6bd;letter-spacing:2px}video{width:100%;max-height:78vh;display:block;background:#0b0e16;margin:22px 0}.chapters{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}button{display:flex;gap:14px;text-align:left;background:#141b27;border:1px solid #2c3849;color:#f2eee6;padding:14px;font:14px system-ui;cursor:pointer}button:hover{border-color:#57d6bd}time{color:#57d6bd}a{color:#70b8e8}footer{font-size:13px;color:#8995a6;line-height:1.7;margin:24px 0}
</style></head><body><main><small>MATHHACKSON / MOTION STUDY 02</small><h1>Learning While Acting</h1><p>3:00 · English narration · Clean picture — no embedded or burned-in captions</p><video id="film" controls preload="metadata" src="__FILE__"></video><div class="chapters">__CHAPTERS__</div><footer>Toy simulations and proposed mechanisms, not footage from the research model.<br>Ant trajectories, articulated gait, equations and diagrams are original ManimGL scenes.<br>Voice: synthetic Microsoft en-US-AndrewNeural. <a href="Learning_While_Acting.en.srt" download>Separate English subtitles</a> · <a href="ManimGL_sources_v2.zip" download>Scene sources</a></footer></main><script>
const v=document.getElementById('film');for(const b of document.querySelectorAll('[data-time]'))b.onclick=()=>{v.currentTime=Number(b.dataset.time);v.play();v.scrollIntoView({behavior:'smooth',block:'center'});};
</script></body></html>'''
    (OUT/'index.html').write_text(page.replace('__FILE__',filename).replace('__CHAPTERS__','\n'.join(buttons)))


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--quality',choices=('hd','draft'),default='hd');args=parser.parse_args()
    fps=60 if args.quality=='hd' else 30
    width,height=(1920,1080) if args.quality=='hd' else (1280,720)
    normalized=OUT/f'normalized-{args.quality}';normalized.mkdir(exist_ok=True)
    listing=[];checks=[]
    for ch in CHAPTERS:
        source=OUT/f'clips-{args.quality}'/f'{ch.scene}.mp4'
        info=metadata(source);vs=next(s for s in info['streams'] if s['codec_type']=='video')
        if vs.get('width',0)!=width or vs.get('height',0)!=height or vs.get('r_frame_rate','')!=f'{fps}/1':
            raise RuntimeError(f'Unexpected clip geometry: {source}')
        if abs(float(info['format']['duration'])-ch.seconds)>2/fps:
            raise RuntimeError(f'Clip duration differs: {source}')
        target=normalized/source.name
        run(['ffmpeg','-y','-v','error','-i',str(source),'-an','-vf',f'trim=end_frame={ch.seconds*fps},setpts=PTS-STARTPTS','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(target)])
        measured=metadata(target)['streams'][0]
        frames=int(measured.get('nb_frames','0'))
        if frames!=ch.seconds*fps:raise RuntimeError(f'Wrong frame count in {target}')
        checks.append(ClipCheck(ch.scene,width,height,fps,frames,ch.seconds))
        listing.append(f"file '{target}'\n")
    concat=OUT/f'concat-{args.quality}.txt';concat.write_text(''.join(listing))
    filename=f'Learning_While_Acting_v2_{"1080p60" if args.quality=="hd" else "720p30"}.mp4'
    final=OUT/filename
    # Only video and narration are mapped. There is deliberately no subtitle input,
    # subtitle stream, text overlay, ASS renderer or subtitles video filter here.
    run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-i',str(OUT/'narration.en.wav'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','160k','-movflags','+faststart','-metadata','title=Learning While Acting | Moving Ants | MathHackson',str(final)])
    result=metadata(final)
    kinds=[s['codec_type'] for s in result['streams']]
    if sorted(kinds)!=['audio','video']:raise RuntimeError(f'Unexpected embedded stream: {kinds}')
    video=next(s for s in result['streams'] if s['codec_type']=='video')
    audio=next(s for s in result['streams'] if s['codec_type']=='audio')
    if int(video.get('nb_frames','0'))!=180*fps:raise RuntimeError('Final video frame count wrong')
    if abs(float(audio.get('duration','0'))-180)>.06:raise RuntimeError('Narration duration wrong')
    run(['ffmpeg','-v','error','-i',str(final),'-f','null','-'])
    manifest = {
        'title':'Learning While Acting','version':2,'status':'teaching simulation and proposed mechanisms, not research results',
        'caption_mode':'separate SRT only; no burned-in or embedded captions','duration':180,'fps':fps,'width':width,'height':height,
        'voice':'synthetic Microsoft en-US-AndrewNeural','sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
        'bytes':final.stat().st_size,'clips':[asdict(c) for c in checks],
        'scene_source_sha256':hashlib.sha256((ROOT/'communication/video/scenes/ants_learning.py').read_bytes()).hexdigest(),
        'ant_artwork_sha256':hashlib.sha256((ROOT/'communication/video/scenes/ant_drawing.py').read_bytes()).hexdigest(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
    }
    (OUT/'render-manifest.json').write_text(json.dumps(manifest,indent=2))
    player(filename)
    print(f'Validated: {final}\n180 seconds / {180*fps} frames / no subtitle stream / {final.stat().st_size/1048576:.2f} MiB')


if __name__=='__main__':main()
