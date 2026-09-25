import type {TrainingState,WeightTrace,WeightTracePoint} from './types';

const number=(value:number):string=>value===0?'0.00000000':Math.abs(value)<.00001?value.toExponential(6):value.toFixed(8);
const axisNumber=(value:number):string=>value!==0&&Math.abs(value)<.001?value.toExponential(2):value.toFixed(4);

export class NetworkView {
  row=0;column=0;
  onSelect:(row:number,column:number)=>void=()=>{};
  private state:TrainingState|null=null;
  private mode:'weight'|'self'|'outer'='weight';
  private ctx:CanvasRenderingContext2D;
  private left=118;private top=27;private cellW=35;private cellH=15;

  constructor(private canvas:HTMLCanvasElement,private detail:HTMLElement,
    private traceCanvas:HTMLCanvasElement,private traceStatus:HTMLElement,private traceEvents:HTMLElement) {
    canvas.width=700;canvas.height=628;
    this.ctx=canvas.getContext('2d')!;
    canvas.addEventListener('pointerdown',event=>{
      const rect=canvas.getBoundingClientRect();
      const col=Math.floor(((event.clientX-rect.left)*700/rect.width-this.left)/this.cellW);
      const row=Math.floor(((event.clientY-rect.top)*628/rect.height-this.top)/this.cellH);
      if(col>=0&&col<16&&row>=0&&row<39){this.column=col;this.row=row;this.paint();}
    });
    canvas.addEventListener('keydown',event=>{
      if(!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key))return;
      event.preventDefault();
      this.row=Math.max(0,Math.min(38,this.row+(event.key==='ArrowDown'?1:event.key==='ArrowUp'?-1:0)));
      this.column=Math.max(0,Math.min(15,this.column+(event.key==='ArrowRight'?1:event.key==='ArrowLeft'?-1:0)));
      this.paint();
    });
  }

  setMode(mode:'weight'|'self'|'outer'):void {this.mode=mode;this.paint();}
  select(row:number,column:number):void {this.row=row;this.column=column;this.paint();}
  update(state:TrainingState):void {this.state=state;this.paint();}

  setTrace(trace:WeightTrace):void {
    const points=trace.points;
    if(!points.length)return;
    const latest=points.at(-1)!;
    const recent=points.slice(-64);
    const self=recent.filter(point=>point.source==='self'&&point.delta!==0).length;
    const outer=recent.filter(point=>point.source==='outer'&&point.delta!==0).length;
    this.traceStatus.textContent=`W[${trace.row}, ${trace.column}] ${number(latest.value)} · 最近 ${recent.length} 条：自写入 ${self}、外部更新 ${outer}`;
    this.traceCanvas.setAttribute('aria-label',`W[${trace.row}, ${trace.column}] 真实逐步权重，当前 ${number(latest.value)}，最近 ${recent.length} 条记录`);
    this.drawTrace(points);
    const events=points.filter(point=>point.source==='self'||point.source==='outer').slice(-3);
    if(latest.source==='skip')events.push(latest);
    if(!events.length)events.push(latest);
    this.traceEvents.innerHTML=events.reverse().map(point=>{
      const label=point.source==='self'?'自写入':point.source==='outer'?'外部更新':point.source==='initial'?'初始':'跳过';
      const change=point.delta===0?'所选参数未变化':`${point.delta>0?'+':''}${number(point.delta)}`;
      return `<li class="${point.source}"><span>第 ${point.episode} 回合 · ${point.tick} 步</span><strong>${label}</strong><span>${change}</span></li>`;
    }).join('');
  }

  private drawTrace(points:WeightTracePoint[]):void {
    const ctx=this.traceCanvas.getContext('2d')!;
    const width=700,height=200,left=82,right=16,top=21,bottom=29;
    ctx.clearRect(0,0,width,height);ctx.fillStyle='#191d20';ctx.fillRect(0,0,width,height);
    const values=points.map(point=>point.value);
    const low=Math.min(...values),high=Math.max(...values);
    const padding=Math.max((high-low)*.12,1e-6);
    const min=low-padding,max=high+padding;
    const x=(index:number)=>left+index*(width-left-right)/Math.max(1,points.length-1);
    const y=(value:number)=>top+(max-value)/(max-min)*(height-top-bottom);
    ctx.strokeStyle='#394149';ctx.lineWidth=1;ctx.font='12px ui-monospace, monospace';ctx.fillStyle='#a8b6bd';
    for(const [value,position] of [[max,top],[min,height-bottom]] as const){
      ctx.beginPath();ctx.moveTo(left,position);ctx.lineTo(width-right,position);ctx.stroke();
      ctx.fillText(axisNumber(value),5,position+4);
    }
    ctx.fillText(`步 ${points[0].tick}`,left,height-8);
    ctx.textAlign='right';ctx.fillText(`步 ${points.at(-1)!.tick}`,width-right,height-8);ctx.textAlign='left';
    ctx.strokeStyle='#dce9e5';ctx.lineWidth=1.5;ctx.beginPath();
    points.forEach((point,index)=>{if(index===0)ctx.moveTo(x(index),y(point.value));else ctx.lineTo(x(index),y(point.value));});ctx.stroke();
    points.forEach((point,index)=>{
      if(point.delta===0||point.source==='skip')return;
      ctx.fillStyle=point.source==='self'?'#77d6b2':'#f0ad6e';
      ctx.beginPath();ctx.arc(x(index),y(point.value),point.source==='outer'?4:2.5,0,Math.PI*2);ctx.fill();
    });
    const last=points.at(-1)!;
    ctx.fillStyle=last.source==='self'?'#77d6b2':last.source==='outer'?'#f0ad6e':'#dce9e5';
    ctx.beginPath();ctx.arc(x(points.length-1),y(last.value),4,0,Math.PI*2);ctx.fill();
  }

  private paint():void {
    const state=this.state;if(!state)return;
    const ctx=this.ctx;
    const values=this.mode==='weight'?state.weights:this.mode==='self'?state.self_delta:state.outer_delta;
    const maximum=Math.max(this.mode==='weight'?.1:1e-7,...values.flat().map(Math.abs));
    ctx.clearRect(0,0,700,628);ctx.fillStyle='#15171a';ctx.fillRect(0,0,700,628);
    ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillStyle='#a5abb1';
    for(let col=0;col<16;col++)ctx.fillText(String(col+1),this.left+(col+.5)*this.cellW,16);
    for(const group of state.groups){
      ctx.textAlign='left';ctx.fillStyle=group.frozen?'#9da2aa':'#c3e6d8';
      ctx.fillText(group.label,4,this.top+group.start*this.cellH+12);
      if(group.end-group.start>3){ctx.font='10px system-ui';ctx.fillText(group.frozen?'冻结':'可更新',4,this.top+group.start*this.cellH+24);ctx.font='12px system-ui';}
    }
    for(let row=0;row<39;row++){
      const group=state.groups.find(item=>row>=item.start&&row<item.end)!;
      for(let col=0;col<16;col++){
        const frozen=!(state.trainable?.[row]?.[col]??!group.frozen);
        const value=values[row][col],strength=Math.min(1,Math.abs(value)/maximum);
        const rgb=value>=0?[230,160,86]:[63,184,211];
        const alpha=.13+.87*Math.sqrt(strength);
        ctx.fillStyle=`rgba(${rgb.join(',')},${alpha*(frozen?.5:1)})`;
        const x=this.left+col*this.cellW,y=this.top+row*this.cellH;
        ctx.fillRect(x+1,y+1,this.cellW-2,this.cellH-2);
        if(frozen){ctx.strokeStyle='#afb2bc40';ctx.beginPath();ctx.moveTo(x+2,y+this.cellH-2);ctx.lineTo(x+10,y+2);ctx.stroke();}
      }
    }
    ctx.strokeStyle='#f5f7fa';ctx.lineWidth=2;
    ctx.strokeRect(this.left+this.column*this.cellW,this.top+this.row*this.cellH,this.cellW,this.cellH);
    ctx.fillStyle='#a5abb1';ctx.textAlign='left';ctx.font='11px system-ui';
    const active=state.trainable?state.trainable.flat().filter(Boolean).length:
      state.groups.reduce((count,group)=>count+(group.frozen?0:(group.end-group.start)*16),0);
    ctx.fillText(`色阶 ±${maximum.toExponential(2)}  ·  可更新 ${active} / 624`,this.left,624);
    const row=this.row,col=this.column,group=state.groups.find(g=>row>=g.start&&row<g.end)!;
    const trainable=state.trainable?.[row]?.[col]??!group.frozen;
    const reason=!trainable&&state.phase==='motor'&&group.id==='action'&&!group.manual?'基础阶段未启用':group.reason;
    this.onSelect(row,col);
    this.detail.innerHTML=`<div class="parameter-title"><strong>W[${row}, ${col}]</strong><span>${group.label} · ${reason}</span></div>
      <dl><div><dt>当前值</dt><dd>${number(state.weights[row][col])}</dd></div><div><dt>初始化</dt><dd>${number(state.initial[row][col])}</dd></div>
      <div><dt>最近自写入 Δ</dt><dd>${number(state.self_delta[row][col])}</dd></div><div><dt>最近外部更新 Δ</dt><dd>${number(state.outer_delta[row][col])}</dd></div></dl>
      <div class="input-detail">输入 ${col+1} · ${state.inputs[col]} <b>${number(state.observation[col])}</b></div>`;
    this.canvas.setAttribute('aria-label',`参数矩阵，${group.label}，第 ${row+1} 行，第 ${col+1} 列，当前值 ${state.weights[row][col]}，${reason}`);
  }
}
