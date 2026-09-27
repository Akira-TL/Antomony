"""Sentence-aligned English narration and two-line EN/ZH sidecars for v4."""
from __future__ import annotations
import asyncio
from dataclasses import asdict,dataclass
import json
import re
import numpy as np
from communication.video.progress.plan import CHAPTERS,OUT,RATE,VOICE
from communication.video.production.retime_audio import load,save,run,timestamp
from communication.video.stage import voice as synthesis

@dataclass(frozen=True)
class Cue:
    chapter:int
    sentence:int
    start:float
    end:float
    en:str
    zh:str
    tempo:float

def validate(cues:list[Cue])->None:
    previous=0.
    for c in cues:
        if not (previous <= c.start < c.end <=180.001):
            raise ValueError(f'Invalid cue timing: {c}')
        if '\n' in c.en or '\n' in c.zh or not re.search('[\u3400-\u9fff]',c.zh):
            raise ValueError('Each cue must have one English and one Chinese text line')
        previous=c.end

async def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audio').mkdir(exist_ok=True)
    # Reuse the established synthesizer/cache implementation in an isolated process.
    # No old source is modified; its output variable points exclusively to v4.
    synthesis.OUT=OUT
    synthesis.VOICE=VOICE
    timeline:list[Cue]=[]
    all_samples=[]
    offset=0
    for i,chapter in enumerate(CHAPTERS,1):
        paragraph=' '.join(s.en for s in chapter.sentences)
        raw,bounds=await synthesis.raw_speech(i,paragraph)
        if len(bounds)!=len(chapter.sentences):
            raise ValueError(f'{chapter.scene}: TTS returned {len(bounds)} boundaries for {len(chapter.sentences)} sentences')
        cuts=[0]+[round((a.start+a.duration+b.start)*.5*RATE) for a,b in zip(bounds,bounds[1:])]+[len(raw)]
        samples=np.zeros(chapter.seconds*RATE,dtype=np.int16)
        for j,(sentence,bound) in enumerate(zip(chapter.sentences,bounds)):
            segment=raw[cuts[j]:cuts[j+1]]
            next_start=chapter.sentences[j+1].start if j+1<len(chapter.sentences) else chapter.seconds-.08
            available=next_start-sentence.start-.04
            tempo=max(1.,len(segment)/RATE/available)
            if tempo>1.18:
                raise ValueError(f'{chapter.scene} sentence {j}: speed {tempo:.3f}; adjust timing, never truncate speech')
            used_tempo=tempo*1.007 if tempo>1 else 1.
            if tempo>1:
                before=OUT/f'audio/p{i:02}-s{j}-raw.wav';after=OUT/f'audio/p{i:02}-s{j}-fit.wav'
                save(before,segment)
                run(['ffmpeg','-y','-v','error','-i',str(before),'-af',f'atempo={used_tempo:.8f}','-ar',str(RATE),'-ac','1','-c:a','pcm_s16le',str(after)])
                segment=load(after)
            begin=round(sentence.start*RATE);end=begin+len(segment)
            if end>round(next_start*RATE):raise ValueError('Speech overlap')
            fade=min(100,len(segment)//4);floating=segment.astype(float)
            floating[:fade]*=np.linspace(0,1,fade);floating[-fade:]*=np.linspace(1,0,fade)
            samples[begin:end]=np.clip(floating,-32768,32767).astype(np.int16)
            audible_start=sentence.start+max(0.,bound.start-cuts[j]/RATE)/used_tempo
            audible_end=min(end/RATE,sentence.start+max(0.,bound.start+bound.duration-cuts[j]/RATE)/used_tempo+.10)
            timeline.append(Cue(i,j,offset+audible_start,offset+audible_end,sentence.en,sentence.zh,used_tempo))
        save(OUT/f'audio/p{i:02}-timed.wav',samples)
        all_samples.append(samples);offset+=chapter.seconds
        print(chapter.scene,'placed',len(bounds),'sentences',flush=True)
    validate(timeline)
    combined=np.concatenate(all_samples)
    if len(combined)!=180*RATE:raise ValueError('Incorrect sample count')
    save(OUT/'audio/combined.wav',combined)
    run(['ffmpeg','-y','-v','error','-i',str(OUT/'audio/combined.wav'),'-af','loudnorm=I=-17:TP=-1.5:LRA=11,aresample=48000,asetpts=PTS-STARTPTS,apad=whole_len=8640000,atrim=end_sample=8640000','-ar',str(RATE),'-ac','1','-c:a','pcm_s16le',str(OUT/'Narration.en.wav')])
    if len(load(OUT/'Narration.en.wav'))!=180*RATE:raise ValueError('Normalization changed duration')
    for mode in ('en-zh','en','zh'):
        entries=[]
        for k,c in enumerate(timeline,1):
            content=c.en+'\n'+c.zh if mode=='en-zh' else getattr(c,mode)
            entries.append(f'{k}\n{timestamp(c.start)} --> {timestamp(c.end)}\n{content}\n')
        (OUT/f'Learning_While_Acting_v4.{mode}.srt').write_text('\n'.join(entries),encoding='utf-8-sig',newline='\n')
    (OUT/'speech-visual-timeline.json').write_text(json.dumps([asdict(c) for c in timeline],ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 180s audio;',len(timeline),'nonoverlapping cues; bilingual text has exactly 2 lines per cue.')

if __name__=='__main__':asyncio.run(main())
