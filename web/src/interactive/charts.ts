import type {ParameterModule} from './types';

export class ParameterCharts {
  private module:ParameterModule|null=null;private page=0;private tick=0;
  private hovering=false;
  private tip:HTMLDivElement;
  constructor(private host:HTMLElement,private counter:HTMLElement){
    this.tip=document.createElement('div');this.tip.className='chart-tip';this.tip.hidden=true;host.append(this.tip);
    new ResizeObserver(()=>this.draw()).observe(host);
  }
  set(module:ParameterModule|null,tick:number,reset=false):void{if(!reset&&this.module===module&&this.tick===tick)return;if(reset||module?.key!==this.module?.key){this.page=0;this.hovering=false;}this.module=module;this.tick=tick;if(!this.hovering)this.draw();}
  turn(delta:number):void{const total=this.module?.points[0]?.values.length??0;this.page=Math.max(0,Math.min(Math.ceil(total/9)-1,this.page+delta));this.hovering=false;this.draw();}
  private draw():void{
    this.host.querySelectorAll('.spark').forEach(n=>n.remove());this.tip.hidden=true;
    const module=this.module,points=module?.points.filter(p=>p.tick<=this.tick)??[],total=points[0]?.values.length??0;
    if(module?.key!=='adaptive'&&points.length===1&&this.tick>points[0].tick)points.push({tick:this.tick,values:points[0].values});
    this.counter.textContent=total?`${this.page*9+1}–${Math.min(total,(this.page+1)*9)} / ${total}`:'无参数';
    if(!points.length)return;
    for(let index=this.page*9;index<Math.min(total,(this.page+1)*9);index++){
      const box=document.createElement('div');box.className='spark';
      const name=module?.key==='adaptive'?`接收点 ${Math.floor(index/5)+1} · R${index%5+4}`:`θ ${index+1}`;
      const title=document.createElement('span');title.textContent=name;box.append(title);
      const canvas=document.createElement('canvas');box.append(canvas);this.host.append(box);
      const w=Math.max(70,box.clientWidth),h=66,dpr=Math.min(devicePixelRatio,2);canvas.width=w*dpr;canvas.height=h*dpr;
      const ctx=canvas.getContext('2d')!;ctx.scale(dpr,dpr);
      const values=points.map(p=>p.values[index]);let lo=Math.min(...values),hi=Math.max(...values);
      if(hi-lo<1e-7){lo-=.02;hi+=.02;}else{const gap=(hi-lo)*.15;lo-=gap;hi+=gap;}
      const from=points[0].tick,to=Math.max(from+1,this.tick);
      const x=(tick:number)=>6+(tick-from)/(to-from)*(w-12),y=(value:number)=>h-8-(value-lo)/(hi-lo)*(h-16);
      ctx.strokeStyle='#d7dcd8';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(6,h-8);ctx.lineTo(w-6,h-8);ctx.stroke();
      if(lo<0&&hi>0){ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(6,y(0));ctx.lineTo(w-6,y(0));ctx.stroke();ctx.setLineDash([]);}
      ctx.strokeStyle=module?.frozen?'#72817e':'#257763';ctx.lineWidth=1.6;ctx.beginPath();
      points.forEach((p,i)=>{if(i)ctx.lineTo(x(p.tick),y(p.values[index]));else ctx.moveTo(x(p.tick),y(p.values[index]));});ctx.stroke();
      if(points.length===1){ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.arc(x(points[0].tick),y(values[0]),2,0,Math.PI*2);ctx.fill();}
      box.addEventListener('pointermove',e=>{
        this.hovering=true;
        const rect=canvas.getBoundingClientRect(),wanted=from+Math.max(0,Math.min(1,(e.clientX-rect.left-6)/(w-12)))*(to-from);
        const p=points.reduce((a,b)=>Math.abs(b.tick-wanted)<Math.abs(a.tick-wanted)?b:a);
        this.tip.textContent=`${name} · 第 ${p.tick} 步 · ${p.values[index].toPrecision(7)}`;this.tip.hidden=false;
        const parent=this.host.getBoundingClientRect();this.tip.style.left=`${Math.max(0,Math.min(parent.width-this.tip.offsetWidth,e.clientX-parent.left+10))}px`;
        this.tip.style.top=`${Math.max(0,e.clientY-parent.top-this.tip.offsetHeight-8)}px`;
      });box.addEventListener('pointerleave',()=>{this.hovering=false;this.tip.hidden=true;this.draw();});
    }
  }
}
