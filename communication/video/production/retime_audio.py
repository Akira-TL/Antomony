"""Place full spoken sentences on the visual beats; never burn subtitles.

Cuts use the TTS provider's sentence boundaries. No claim of phoneme alignment.
The original paragraph MP3s and timing metadata remain untouched.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from communication.video.film_plan import DURATIONS,SENTENCE_STARTS
OUT=ROOT/'communication/video/rendered/v2'
RATE=48000


def run(args:list[str])->None:
    subprocess.run(args,check=True,capture_output=True)


def load(path:Path)->np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate()!=RATE or w.getnchannels()!=1 or w.getsampwidth()!=2:raise ValueError('Need mono 48kHz PCM16')
        return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').copy()


def save(path:Path,samples:np.ndarray)->None:
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(RATE);w.writeframes(samples.astype('<i2').tobytes())


def timestamp(seconds:float)->str:
    ms=round(seconds*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def main()->None:
    work=OUT/'sentence-audio';work.mkdir(exist_ok=True)
    subtitles=[];timeline=[];all_audio=[];absolute=0
    for i,(slot,starts) in enumerate(zip(DURATIONS,SENTENCE_STARTS),1):
        meta=json.loads((OUT/f'audio/s{i:02}.json').read_text())
        boundaries=meta['boundaries']
        if len(starts)!=len(boundaries):raise ValueError(f'S{i}: sentence count differs')
        raw=work/f's{i:02}-raw.wav'
        run(['ffmpeg','-v','error','-y','-i',str(OUT/f'audio/s{i:02}.mp3'),'-ar',str(RATE),'-ac','1','-c:a','pcm_s16le',str(raw)])
        source=load(raw);out=np.zeros(slot*RATE,dtype=np.int16)
        cuts=[0]
        for j in range(len(boundaries)-1):
            left=boundaries[j]['start']+boundaries[j]['duration'];right=boundaries[j+1]['start']
            cuts.append(round(.5*(left+right)*RATE))
        cuts.append(len(source))
        for j,(b,start) in enumerate(zip(boundaries,starts)):
            speech=source[cuts[j]:cuts[j+1]]
            next_start=starts[j+1] if j+1<len(starts) else slot-.18
            available=next_start-start-.045
            ratio=max(1.,len(speech)/RATE/available)
            if ratio>1.27:raise ValueError(f'S{i}/{j}: speaking rate would be unnatural ({ratio:.3f})')
            if ratio>1:
                piece=work/f's{i:02}-{j:02}-raw.wav';fit=work/f's{i:02}-{j:02}-fit.wav';save(piece,speech)
                run(['ffmpeg','-v','error','-y','-i',str(piece),'-af',f'atempo={ratio*1.007:.7f}','-ar',str(RATE),'-ac','1','-c:a','pcm_s16le',str(fit)])
                speech=load(fit)
            begin=round(start*RATE);end=begin+len(speech)
            if end>round(next_start*RATE)-12:raise ValueError(f'S{i}/{j}: overlapping sentences')
            # Tiny fades avoid clicks at sentence cuts, not word-length truncation.
            fade=min(120,len(speech)//4)
            floating=speech.astype(float)
            floating[:fade]*=np.linspace(0,1,fade);floating[-fade:]*=np.linspace(1,0,fade)
            out[begin:end]=np.clip(floating,-32768,32767).astype(np.int16)
            text=textwrap.fill(b['text'],width=58)
            subtitles.append(f'{len(subtitles)+1}\n{timestamp(absolute+start)} --> {timestamp(absolute+end/RATE)}\n{text}\n')
            timeline.append({'chapter':i,'sentence':j,'start':absolute+start,'end':absolute+end/RATE,'tempo':ratio,'text':b['text']})
        save(OUT/f'audio/s{i:02}-timed.wav',out);all_audio.append(out);absolute+=slot
        print(f'S{i:02}: {len(starts)} complete sentences placed on visual beats',flush=True)
    combined=np.concatenate(all_audio)
    if len(combined)!=180*RATE:raise ValueError('Whole film length incorrect')
    save(work/'combined.wav',combined)
    run(['ffmpeg','-v','error','-y','-i',str(work/'combined.wav'),'-af','loudnorm=I=-17:TP=-1.5:LRA=11,aresample=48000,asetpts=PTS-STARTPTS,apad=whole_len=8640000,atrim=end_sample=8640000','-ar',str(RATE),'-ac','1','-c:a','pcm_s16le',str(OUT/'narration.en.wav')])
    if len(load(OUT/'narration.en.wav'))!=180*RATE:raise ValueError('Normalized audio sample count wrong')
    (OUT/'Learning_While_Acting.en.srt').write_text('\n'.join(subtitles))
    (OUT/'speech-visual-timeline.json').write_text(json.dumps(timeline,indent=2))
    print('Sentence timing complete; clean film uses audio only. SRT stays separate.')


if __name__=='__main__':main()
