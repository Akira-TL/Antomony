import './style.css';
import {ColonyScene} from './scene';
import {NetworkDiagram} from './render/network-view';
import {layout} from './ui/layout';
import {ComparisonHistory,comparisonText,contactRate} from './ui/comparison';
import type {Command,ComparisonFrame,Tool,AntState} from './types';

const app=document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML=layout;
const el=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const scene=new ColonyScene(el('viewport'));
const referenceScene=new ColonyScene(el('viewport-reference'));
referenceScene.active=false;
referenceScene.interactive=false;
referenceScene.selected=-1;
const diagram=new NetworkDiagram(el<HTMLCanvasElement>('network'));
const history=new ComparisonHistory();
let current:ComparisonFrame|null=null;
let ws:WebSocket;
let generation=0;
let lastToast='';
let toastTimer=0;
let changingComparison=false;

function send(command:Command):void {
  if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify(command));
  else toast('本地仿真尚未连接');
}
function toast(message:string):void {
  clearTimeout(toastTimer);
  el('toast').textContent=message;el('toast').classList.add('visible');
  toastTimer=window.setTimeout(()=>el('toast').classList.remove('visible'),3500);
}
function syncCamera():void {
  if(!referenceScene.active)return;
  referenceScene.camera.position.copy(scene.camera.position);
  referenceScene.camera.quaternion.copy(scene.camera.quaternion);
  referenceScene.controls.target.copy(scene.controls.target);
  referenceScene.controls.update();
}
scene.controls.addEventListener('change',syncCamera);
function fitView():void {scene.fitArena();syncCamera();}
el('fit-view').onclick=fitView;
el('fullscreen').onclick=()=>{
  const request=document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();
  void request.catch(()=>toast('当前浏览器不允许全屏，可使用 F11'));
};

const hints:Record<Tool,string>={
  inspect:'拖动旋转 · 滚轮缩放 · 点击观察个体',
  wall:'移动预览 · 滚轮转角 · 点击放墙，蚂蚁就近让位',
  erase:'点击墙体拆除',food:'移动鼠标预览 · 点击放资源',
  scent:'点击喷洒食物信号；这里未必有食物',
};
function setTool(tool:Tool):void {
  scene.tool=tool;
  document.querySelectorAll<HTMLElement>('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===tool));
  el('hint').textContent=hints[tool];el('wall-settings').hidden=tool!=='wall';
  scene.wallPreview.hide();scene.foodPreview.hide();
  el('hint').textContent=hints[tool];
}
document.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach(button=>{
  button.onclick=()=>setTool(button.dataset.tool as Tool);
});
scene.onPoint=(x,y,id)=>{
  if(scene.tool==='inspect'){
    if(id!==null){scene.selected=id;el<HTMLSelectElement>('ant-select').value=String(id);if(current)inspect(current.ants[id]);}
    return;
  }
  send({kind:scene.tool,x,y,...(scene.tool==='wall'?{
    hx:scene.wallPreview.hx,hy:scene.wallPreview.hy,angle:scene.wallPreview.angle,
  }:{})});
};
scene.wallPreview.onStatus=message=>{
  if(scene.tool==='wall')el('hint').textContent=message?`${message} · 滚轮调角`:hints.wall;
  el('wall-angle').textContent=`${Math.round(scene.wallPreview.angle*180/Math.PI)}°`;
};
scene.foodPreview.onStatus=message=>{if(scene.tool==='food')el('hint').textContent=message||hints.food;};
el<HTMLInputElement>('wall-length').oninput=()=>{
  const length=Number(el<HTMLInputElement>('wall-length').value);
  el('wall-length-value').textContent=length.toFixed(1);scene.wallPreview.setSize(.4,length/2);
};
function rotateWall():void {scene.wallPreview.rotate(Math.PI/2);el('wall-angle').textContent=`${Math.round(scene.wallPreview.angle*180/Math.PI)}°`;}
el('wall-rotate').onclick=rotateWall;
window.addEventListener('keydown',event=>{
  if(event.target instanceof HTMLInputElement||event.target instanceof HTMLSelectElement)return;
  if(scene.tool==='wall'&&event.key.toLowerCase()==='r'){event.preventDefault();rotateWall();}
  if(event.key==='Escape')setTool('inspect');
});
el('pause').onclick=()=>send({kind:'pause'});
el('step').onclick=()=>send({kind:'step'});
el('speed').onclick=()=>send({kind:'speed',value:current?.rate===1?2:current?.rate===2?4:1});
el('reset').onclick=()=>{
  const seed=crypto.getRandomValues(new Uint32Array(1))[0]%2147483647;
  send({kind:'reset',seed,count:current?.ants.length??32});toast('新种子，重新独立预热…');
};
el('compare').onclick=()=>{
  if(!current||changingComparison)return;
  changingComparison=true;
  send({kind:'compare',value:current.reference?0:1});
  toast(current.reference?'返回单场景':'按当前种子重开，两组从同一起点开始');
};
el<HTMLInputElement>('field-view').onchange=e=>{
  scene.showField=referenceScene.showField=(e.target as HTMLInputElement).checked;
};
el<HTMLInputElement>('wind').onchange=e=>send({kind:'wind',value:Number((e.target as HTMLInputElement).value)});
el<HTMLInputElement>('wind').oninput=e=>el('wind-value').textContent=Number((e.target as HTMLInputElement).value).toFixed(1);
el('clear-scent').onclick=()=>send({kind:'clear'});
el('freeze-all').onclick=()=>send({kind:'learning',value:current?.ants.every(a=>a.frozen)?1:0});
el('freeze-one').onclick=()=>send({kind:'freeze-ant',ant:scene.selected});
el<HTMLSelectElement>('ant-select').onchange=e=>{
  scene.selected=Number((e.target as HTMLSelectElement).value);if(current)inspect(current.ants[scene.selected]);
};
el('apply-seed').onclick=()=>{
  const seed=Number(el<HTMLInputElement>('seed').value);
  if(Number.isInteger(seed)&&seed>=0&&seed<2147483647)send({kind:'reset',seed,count:current?.ants.length??32});
  else toast('请输入有效的非负整数种子');
};
el('export').onclick=()=>{const a=document.createElement('a');a.href='/api/export';a.download='colony-snapshot.json';a.click();};
const dialog=el<HTMLDialogElement>('explanation');
el('about').onclick=()=>dialog.showModal();
el('close-about').onclick=el('understood').onclick=()=>dialog.close();

function inspect(ant:AntState):void {
  if(!ant)return;
  el('selected-label').textContent=`个体 ${String(ant.id).padStart(2,'0')}`;
  const state=ant.carrying?'携食返巢':ant.following_trail?'循迹寻食':'局部探索';
  el('ant-state').textContent=state+(ant.frozen?' · 学习暂停':'');
  el('freeze-one').textContent=ant.frozen?'继续学习':'暂停学习';
  el('ant-updates').textContent=ant.updates.toLocaleString();el('ant-error').textContent=ant.error.toFixed(4);
  diagram.select(ant.id,current?.paused??true,generation);
}
function update(frame:ComparisonFrame):void {
  if(!frame||!Array.isArray(frame.ants))return;
  const compared=!!frame.reference,wasCompared=!!current?.reference;
  const fresh=!current||current.seed!==frame.seed||current.ants.length!==frame.ants.length||frame.tick<current.tick||compared!==wasCompared;
  if(fresh){
    generation++;lastToast='';
    el<HTMLSelectElement>('ant-select').replaceChildren(...frame.ants.map(a=>{
      const option=document.createElement('option');option.value=String(a.id);option.textContent=`个体 ${String(a.id).padStart(2,'0')}`;return option;
    }));
    scene.selected=0;el<HTMLInputElement>('seed').value=String(frame.seed);
  }
  if(compared!==wasCompared||!current){
    referenceScene.active=compared;el('reference-pane').hidden=!compared;
    el('comparison-results').hidden=!compared;el('arenas').classList.toggle('comparing',compared);
    requestAnimationFrame(fitView);
  }
  current=frame;changingComparison=false;
  scene.update(frame);if(frame.reference)referenceScene.update(frame.reference);
  app.dataset.tick=String(frame.tick);app.dataset.ready='true';app.dataset.generation=String(generation);app.dataset.comparing=String(compared);
  app.dataset.referenceTick=frame.reference?String(frame.reference.tick):'';
  el('connection').textContent=frame.paused?'已暂停':'实时运行';el('lamp').classList.add('online');
  el('clock').textContent=`${String(Math.floor(frame.seconds/60)).padStart(2,'0')}:${(frame.seconds%60).toFixed(1).padStart(4,'0')}`;
  el('heading').textContent=compared?'同一场改变，两种应对。':'一条路，如何被发现。';
  el('compare').textContent=compared?'返回单场景':'开启同条件对比';
  el('population').textContent=`${frame.ants.length} 只${compared?' / 组':''}`;
  const learning=frame.ants.filter(a=>!a.frozen).length;
  el('neural-state').textContent=`${frame.ants.length} 只 · ${frame.paused?'仿真已暂停':learning?`${learning} 只学习中`:'学习已暂停'}`;
  el('delivered').textContent=String(frame.delivered);el('contact-rate').textContent=contactRate(frame).toFixed(1);el('stalled').textContent=String(frame.stalled);
  el('trail-count').textContent=`循迹 ${frame.ants.filter(a=>a.following_trail).length} 只`;
  el('edit-scope').textContent=compared?'左侧编辑，同时改变两组环境':'每个体独立学习；信息素连接群体';
  el('pause').textContent=frame.paused?'▶':'Ⅱ';el<HTMLButtonElement>('step').disabled=!frame.paused;el('speed').textContent=`${frame.rate}×`;
  el('freeze-all').textContent=learning?'暂停学习':'继续学习';
  if(document.activeElement!==el('wind')){el<HTMLInputElement>('wind').value=String(frame.wind);el('wind-value').textContent=frame.wind.toFixed(1);}
  inspect(frame.ants[scene.selected]??frame.ants[0]);
  if(frame.reference){
    const ref=frame.reference;
    el('reference-delivered').textContent=String(ref.delivered);el('reference-contacts').textContent=contactRate(ref).toFixed(1);el('reference-stalled').textContent=String(ref.stalled);
    const text=comparisonText(frame.delivered,ref.delivered);
    el('difference').textContent=text.title;el('difference-detail').textContent=text.detail;
    if(history.observe(frame,generation)){
      el('neural-line').setAttribute('d',history.path('neural'));el('rule-line').setAttribute('d',history.path('rules'));
    }
  }
  el('runtime').textContent=`本地运行 · 渲染 ${scene.fps.toFixed(0)} FPS · 仿真步 ${frame.tick_ms.toFixed(1)} ms${frame.reference?` + ${frame.reference.tick_ms.toFixed(1)} ms`:''}`;
  el('events').replaceChildren(...frame.events.filter(e=>e.kind!=='learn'&&e.kind!=='notice').slice(0,3).map(e=>{
    const row=document.createElement('div');row.className='event';const time=document.createElement('time');time.textContent=(e.tick/10).toFixed(1)+'s';
    const text=document.createElement('span');text.textContent=e.message;row.append(time,text);return row;
  }));
  const notice=frame.events.find(e=>e.kind==='notice'||e.kind==='error');
  if(notice){const key=notice.tick+notice.message;if(key!==lastToast){lastToast=key;toast(notice.message);}}
}
function connect():void {
  ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
  ws.onmessage=e=>{try{update(JSON.parse(String(e.data)) as ComparisonFrame);}catch(error){console.error('显示快照失败',error);toast('快照显示异常；保持上一帧');}};
  ws.onclose=()=>{
    changingComparison=false;el('connection').textContent='连接断开，正在重连';el('lamp').classList.remove('online');
    diagram.select(scene.selected,true,generation);setTimeout(connect,1500);
  };
  ws.onerror=()=>ws.close();
}
connect();
