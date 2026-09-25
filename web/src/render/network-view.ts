interface Edge {layer:number;source:number;target:number;weight:number;contribution:number;update:number}
interface Signals {tick:number;ant:number;paused:boolean;frozen:boolean;updates:number;layers:number[][];widths:number[];connections:Edge[]}

/** 线宽来自实际权重；亮度来自输入×权重；流光只表示新一帧前向计算。 */
export class NetworkDiagram {
  private signals:Signals|null=null;
  private received=0;
  private selected=0;
  private generation=-1;
  private loading=false;
  private lastPoll=0;
  private updateFlash=false;
  private paused=false;
  constructor(private canvas:HTMLCanvasElement){requestAnimationFrame(this.draw);}
  select(id:number,paused:boolean,generation=0):void{
    this.paused=paused;
    if(id!==this.selected||generation!==this.generation){this.selected=id;this.generation=generation;this.signals=null;this.lastPoll=0;this.updateFlash=false;}
    if(!this.loading&&performance.now()-this.lastPoll>180)void this.poll();
  }
  private async poll():Promise<void>{
    this.loading=true;this.lastPoll=performance.now();const id=this.selected,generation=this.generation;
    try{
      const response=await fetch(`/api/network?ant=${id}`,{signal:AbortSignal.timeout(2000)});
      if(!response.ok)return;
      const signals=await response.json() as Signals|null;
      if(id!==this.selected||generation!==this.generation||!signals)return;
      if(!this.signals||signals.tick!==this.signals.tick){
        this.updateFlash=!!this.signals&&signals.updates>this.signals.updates;
        this.received=performance.now();
      }
      this.signals=signals;
      this.canvas.dataset.signalTick=String(signals.tick);
      this.canvas.dataset.connections=String(signals.connections.length);
    }catch{ /* 断开时保留最后一次观测，不编造活动。 */ }
    finally{this.loading=false;}
  }
  private draw=():void=>{
    requestAnimationFrame(this.draw);
    const ctx=this.canvas.getContext('2d');if(!ctx)return;
    const w=this.canvas.width,h=this.canvas.height;
    ctx.clearRect(0,0,w,h);const s=this.signals;if(!s)return;
    const xs=[30,w*.34,w*.65,w-30];
    const point=(layer:number,index:number)=>({x:xs[layer],y:22+index*(h-58)/Math.max(1,s.layers[layer].length-1)});
    const age=(performance.now()-this.received)/1000;
    const maxWeight=Math.max(.01,...s.connections.map(e=>Math.abs(e.weight)));
    const maxSignal=Math.max(.02,...s.connections.map(e=>Math.abs(e.contribution)));
    for(const edge of s.connections){
      const a=point(edge.layer,edge.source),b=point(edge.layer+1,edge.target);
      const strength=Math.sqrt(Math.min(1,Math.abs(edge.contribution)/maxSignal));
      const color=edge.contribution>=0?'143,223,183':'241,183,105';
      ctx.strokeStyle=`rgba(${color},${.06+.55*strength})`;
      ctx.lineWidth=.35+2.8*Math.sqrt(Math.abs(edge.weight)/maxWeight);
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
      const phase=(age-edge.layer*.035)/.13;
      if(!this.paused&&!s.paused&&phase>=0&&phase<=1&&strength>.18){
        ctx.fillStyle=`rgba(${color},${strength})`;ctx.beginPath();ctx.arc(a.x+(b.x-a.x)*phase,a.y+(b.y-a.y)*phase,1+strength*2,0,Math.PI*2);ctx.fill();
      }
      if(!this.paused&&this.updateFlash&&!s.frozen&&age<.18&&Math.abs(edge.update)>.0001){
        ctx.strokeStyle=`rgba(244,222,143,${Math.min(.8,Math.abs(edge.update)*60)*(1-age/.18)})`;ctx.lineWidth=3.5;
        ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
      }
    }
    s.layers.forEach((values,layer)=>values.forEach((value,i)=>{
      const p=point(layer,i),alpha=.3+.7*Math.min(1,Math.abs(value));
      ctx.fillStyle=value>=0?`rgba(194,248,199,${alpha})`:`rgba(250,183,105,${alpha})`;
      ctx.beginPath();ctx.arc(p.x,p.y,layer===3?6:4,0,Math.PI*2);ctx.fill();
    }));
    ctx.fillStyle='rgba(188,215,201,.65)';ctx.font='18px system-ui';ctx.textAlign='center';
    ['感知','隐藏层 24','隐藏层 16','预测'].forEach((label,i)=>ctx.fillText(label,xs[i],h-6));
  };
}
