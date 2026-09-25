import type {ComparisonFrame,Frame} from '../types';

export function contactRate(frame:Pick<Frame,'contacts'|'samples'>):number {
  return frame.samples>0?100*frame.contacts/frame.samples:0;
}

/** 直接报告观测差值；分母为零不显示无穷大或虚构百分比。 */
export function comparisonText(neural:number,rules:number):{title:string;detail:string} {
  const difference=neural-rules;
  const title=difference===0?'两组搬运量相同':`神经组${difference>0?'多':'少'}搬回 ${Math.abs(difference)} 块`;
  if(rules===0)return {title:neural===0?'等待第一批搬运':title,detail:'普通组暂为 0；不计算相对百分比'};
  const percent=100*difference/rules;
  return {title,detail:`相对普通组 ${percent>0?'+':''}${percent.toFixed(1)}% · 本次运行，不代表普遍优势`};
}

interface Sample {seconds:number;neural:number;rules:number}
export class ComparisonHistory {
  private generation=-1;
  private tick=-1;
  private samples:Sample[]=[];

  observe(frame:ComparisonFrame,generation:number):boolean {
    if(generation!==this.generation||frame.tick<this.tick){this.samples=[];this.tick=-1;this.generation=generation;}
    if(!frame.reference||frame.tick===this.tick)return false;
    this.tick=frame.tick;
    this.samples.push({seconds:frame.seconds,neural:frame.delivered,rules:frame.reference.delivered});
    if(this.samples.length>600)this.samples.shift();
    return true;
  }

  path(side:'neural'|'rules'):string {
    if(!this.samples.length)return '';
    const first=this.samples[0].seconds,last=this.samples.at(-1)!.seconds;
    const max=Math.max(1,...this.samples.flatMap(s=>[s.neural,s.rules]));
    return this.samples.map((sample,i)=>{
      const x=4+492*(sample.seconds-first)/Math.max(.1,last-first);
      const y=66-60*sample[side]/max;
      return `${i===0?'M':'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
  }
}
