import type {TrainingState} from './types';

export class NetworkView {
  row=0;column=0;
  onSelect:(row:number,column:number)=>void=()=>{};
  private state:TrainingState|null=null;
  private mode:'weight'|'self'|'outer'='weight';
  private ctx:CanvasRenderingContext2D;
  private left=118;private top=27;private cellW=35;private cellH=15;

  constructor(private canvas:HTMLCanvasElement,private detail:HTMLElement) {
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
      const frozen=state.groups.find(group=>row>=group.start&&row<group.end)!.frozen;
      for(let col=0;col<16;col++){
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
    ctx.fillText(`色阶 ±${maximum.toExponential(2)}  ·  624 / 624`,this.left,624);
    const row=this.row,col=this.column,group=state.groups.find(g=>row>=g.start&&row<g.end)!;
    this.onSelect(row,col);
    const number=(v:number)=>v===0?'0.000000':Math.abs(v)<.00001?v.toExponential(6):v.toFixed(8);
    this.detail.innerHTML=`<div class="parameter-title"><strong>W[${row}, ${col}]</strong><span>${group.label} · ${group.reason}</span></div>
      <dl><div><dt>当前值</dt><dd>${number(state.weights[row][col])}</dd></div><div><dt>初始化</dt><dd>${number(state.initial[row][col])}</dd></div>
      <div><dt>最近自写入 Δ</dt><dd>${number(state.self_delta[row][col])}</dd></div><div><dt>最近外部更新 Δ</dt><dd>${number(state.outer_delta[row][col])}</dd></div></dl>
      <div class="input-detail">输入 ${col+1} · ${state.inputs[col]} <b>${number(state.observation[col])}</b></div>`;
    this.canvas.setAttribute('aria-label',`参数矩阵，${group.label}，第 ${row+1} 行，第 ${col+1} 列，当前值 ${state.weights[row][col]}，${group.reason}`);
  }
}
