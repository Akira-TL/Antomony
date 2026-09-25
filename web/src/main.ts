import './style.css';
import {ColonyScene} from './scene';
import type {Command,Frame,Tool,AntState} from './types';

const app=document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML=`
<header class="topbar"><a class="brand" href="/" aria-label="MathHackson 首页"><span class="brand-mark">M</span><span>MathHackson <small>研究现场 / 02</small></span></a><div class="top-title">独立神经蚁群 <span>交互实验</span></div><div class="connection"><i id="lamp"></i><span id="connection">连接本地模型…</span><button id="about" class="quiet">实验说明 ↗</button></div></header>
<main class="workspace"><section class="arena-panel"><div class="arena-heading"><div><span class="eyebrow">局部感知 · 环境协作</span><h1>每一只，都有自己的模型。</h1></div><div class="session"><span id="clock">00:00.0</span><small>仿真时间 · <span id="fps">—</span> FPS</small></div></div>
<div class="canvas-wrap"><div id="viewport"></div><div class="scene-label"><span class="live-dot"></span> 真实仿真 <span class="divider">/</span> <span id="population">32</span> 个独立个体</div><div class="legend"><span><i class="home-dot"></i> 回巢信息素</span><span><i class="food-dot"></i> 食物信息素</span><label><input type="checkbox" id="field-view" checked> 显示热力</label></div><div id="hint" class="interaction-hint">拖动旋转 · 滚轮缩放 · 点击蚂蚁查看独立模型</div><div id="toast" class="toast"></div></div>
<div class="toolbar"><div class="tools" role="group" aria-label="场景操作"><button class="tool active" data-tool="inspect"><b>⌖</b> 观察</button><button class="tool" data-tool="wall"><b>▥</b> 放墙</button><button class="tool" data-tool="erase"><b>⌫</b> 拆墙</button><button class="tool" data-tool="food"><b>◈</b> 资源</button><button class="tool" data-tool="scent"><b>∴</b> 信息素</button></div><div class="play-controls"><button id="pause" class="round" title="暂停或继续">Ⅱ</button><button id="step" class="round" title="暂停后单步">▹</button><button id="speed" class="quiet">1×</button><button id="reset" class="quiet" title="新的独立随机种子">重新开始 ↻</button></div></div>
<div class="metrics"><article><span>已搬回像素块</span><strong id="delivered">0</strong><small>真实拾取与归巢</small></article><article><span>实际参数更新</span><strong id="updates">0</strong><small>各自经历，各自更新</small></article><article><span>接触 / 100 次行动</span><strong id="contact-rate">—</strong><small>碰撞约束不等于学会避障</small></article><article><span>仿真单步耗时</span><strong><span id="tick-ms">—</span><em> ms</em></strong><small>含感知、推理、物理与学习</small></article></div>
</section>
<aside class="sidebar"><section class="card control-card"><div class="section-title"><h2>改变环境</h2><span class="small-tag">你来干预</span></div><p>模型只收到局部观察，不知道你的操作目的。</p><div id="wall-settings" class="wall-settings" hidden><label>墙长 <output id="wall-length-value">4.0</output><input id="wall-length" type="range" min="1" max="8" step=".2" value="4" aria-label="墙体长度"></label><button id="wall-rotate">旋转 90° / R</button><small>蓝色可放 · 黄色需让位 · 红色禁放</small></div><div class="control-row"><span>中央横向外力</span><output id="wind-value">0.0</output></div><input id="wind" class="slider" type="range" min="-1.4" max="1.4" step="0.2" value="0" aria-label="中央区域横向外力"><div class="range-caption"><span>← 反向</span><span>中央区域 | x | &lt; 5</span><span>正向 →</span></div><div class="dual"><button id="clear-scent">清空信息素</button><button id="freeze-all">冻结全部参数</button></div><label class="seed-row">世界种子<input id="seed" type="number" value="42" min="0" max="2147483646"><button id="apply-seed">应用</button></label></section>
<section class="card inspector"><div class="section-title"><h2>观察一个独立模型</h2><select id="ant-select" aria-label="选择蚂蚁"></select></div><div class="ant-id"><span class="ant-icon">✳</span><div><strong id="selected-label">个体 00</strong><small id="ant-state">局部探索中</small></div><button id="freeze-one" class="quiet">冻结</button></div><div class="network"><canvas id="network" width="560" height="210" title="节点颜色来自实际激活；层间连线仅为结构示意"></canvas><div><span>局部观察</span><span>独立隐藏状态</span><span>运动预测</span></div></div><dl class="details"><div><dt>在线更新次数</dt><dd id="ant-updates">—</dd></div><div><dt>本次更新幅度</dt><dd id="ant-delta">—</dd></div><div><dt>相对预热参数变化</dt><dd id="ant-drift">—</dd></div><div><dt>更新前预测误差</dt><dd id="ant-error">—</dd></div><div><dt>独立参数指纹</dt><dd id="fingerprint">—</dd></div></dl><div class="micro-note">8 → 24 → 16 → 3 · 各自独立预热<br>在线更新输出层 · 首版未启用回退<br>激活摘录 · 连线仅为结构示意</div></section>
<section class="card event-card"><div class="section-title"><h2>真实事件</h2><button id="export" class="quiet">导出快照 ↓</button></div><div id="events" class="events"><p>等待模型完成独立预热…</p></div></section></aside></main>
<footer><span><i class="footer-dot"></i> 信息素是共享环境，不是共享神经权重</span><span>本机运行 · 不读取图像 · 学习增益需对照验证</span></footer>
<dialog id="explanation"><button id="close-about" class="dialog-close">×</button><span class="eyebrow">这不是一段预录动画</span><h2>看得见行为，也要看得清原因。</h2><p>每只蚂蚁从独立随机参数、独立生成的局部运动练习开始。神经网络预测候选动作的位移与接触，结合局部信息素、近距离资源和自身运动积分选择方向。</p><p>规则负责拾取、放下、信息素沉积与探索偏好；物理负责不能穿墙或相互重叠。网络只从自己实际执行后得到的局部反馈更新。移除一堵墙、信息素蒸发或物理推开都不是参数学习的证据。</p><p>可冻结全部或单独个体的参数，观察参数指纹不再变化。局部预测训练可验证，但本页不宣称已经证明全群搬运效率提高。回退功能尚未启用，不会生成假回退事件。</p><p>所有个体与场景由本地服务驱动；同一服务的多个页面共享同一个现场。</p><button id="understood" class="primary">进入现场</button></dialog>`;

const el=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const scene=new ColonyScene(el('viewport'));
scene.foodPreview.onStatus=message=>{if(scene.tool==='food')el('hint').textContent=message||'移动鼠标预览 · 点击放资源';};
let current:Frame|null=null;let ws:WebSocket;let lastToast='';let generation=0;
const send=(command:Command)=>{if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify(command));else toast('本地模型尚未连接');};
function toast(message:string){el('toast').textContent=message;el('toast').classList.add('visible');window.setTimeout(()=>el('toast').classList.remove('visible'),3500);}
const hints:Record<Tool,string>={inspect:'拖动旋转 · 滚轮缩放 · 点击蚂蚁查看独立模型',wall:'移动鼠标预览 · 滚轮转角 / R 转90° · 点击放墙；蚂蚁会就近让位',erase:'点击一道墙将其拆除 · 不改变已有模型参数',food:'点击空地增加资源点 · 蚂蚁必须靠局部感知发现',scent:'点击地面喷洒食物信号 · 模型不知道这里是否有食物'};
document.querySelectorAll<HTMLButtonElement>('[data-tool]').forEach(button=>button.addEventListener('click',()=>{scene.tool=button.dataset.tool as Tool;document.querySelectorAll('.tool').forEach(b=>b.classList.toggle('active',b===button));el('hint').textContent=hints[scene.tool];el('wall-settings').hidden=scene.tool!=='wall';scene.wallPreview.hide();scene.foodPreview.hide();}));
scene.onPoint=(x,y,id)=>{if(scene.tool==='inspect'){if(id!==null){scene.selected=id;el<HTMLSelectElement>('ant-select').value=String(id);if(current)inspect(current.ants[id]);}}else send({kind:scene.tool,x,y,...(scene.tool==='wall'?{hx:scene.wallPreview.hx,hy:scene.wallPreview.hy,angle:scene.wallPreview.angle}:{})});};
scene.wallPreview.onStatus=message=>{if(scene.tool==='wall')el('hint').textContent=message?message+' · 滚轮调角 · 点击放置':hints.wall;};
function updateWallSize():void {const length=Number(el<HTMLInputElement>('wall-length').value);el('wall-length-value').textContent=length.toFixed(1);scene.wallPreview.setSize(.4,length/2);}
el<HTMLInputElement>('wall-length').oninput=updateWallSize;
el('wall-rotate').onclick=()=>{scene.wallPreview.rotate(Math.PI/2);};
window.addEventListener('keydown',event=>{if(scene.tool==='wall'&&event.key.toLowerCase()==='r'&&!(event.target instanceof HTMLInputElement)){scene.wallPreview.rotate(Math.PI/2);}});
el('pause').onclick=()=>send({kind:'pause'});el('step').onclick=()=>send({kind:'step'});
el('speed').onclick=()=>send({kind:'speed',value:current?.rate===1?2:current?.rate===2?4:1});
el('reset').onclick=()=>{const seed=crypto.getRandomValues(new Uint32Array(1))[0]%2147483647;send({kind:'reset',seed,count:current?.ants.length??32});toast('正在为每只蚂蚁重新独立预热…');};
el<HTMLInputElement>('field-view').onchange=e=>scene.showField=(e.target as HTMLInputElement).checked;
el<HTMLInputElement>('wind').onchange=e=>send({kind:'wind',value:Number((e.target as HTMLInputElement).value)});
el<HTMLInputElement>('wind').oninput=e=>el('wind-value').textContent=Number((e.target as HTMLInputElement).value).toFixed(1);
el('clear-scent').onclick=()=>send({kind:'clear'});
el('freeze-all').onclick=()=>send({kind:'learning',value:current?.ants.every(a=>a.frozen)?1:0});
el('freeze-one').onclick=()=>send({kind:'freeze-ant',ant:scene.selected});
el<HTMLSelectElement>('ant-select').onchange=e=>{scene.selected=Number((e.target as HTMLSelectElement).value);if(current)inspect(current.ants[scene.selected]);};
el('apply-seed').onclick=()=>{const seed=Number(el<HTMLInputElement>('seed').value);if(Number.isInteger(seed)&&seed>=0&&seed<2147483647)send({kind:'reset',seed,count:current?.ants.length??32});else toast('请输入有效的非负整数种子');};
el('export').onclick=()=>{const a=document.createElement('a');a.href='/api/export';a.download='colony-snapshot.json';a.click();};
const dialog=el<HTMLDialogElement>('explanation');el('about').onclick=()=>dialog.showModal();el('close-about').onclick=el('understood').onclick=()=>dialog.close();

function network(ant:AntState):void {
  const ctx=el<HTMLCanvasElement>('network').getContext('2d')!;ctx.clearRect(0,0,560,210);
  const columns=[ant.inputs,ant.hidden.slice(0,8),ant.prediction];const xs=[40,275,515];
  for(let layer=0;layer<columns.length-1;layer++)for(let i=0;i<columns[layer].length;i++)for(let j=0;j<columns[layer+1].length;j++){
    const activity=Math.min(1,Math.abs(columns[layer][i]*columns[layer+1][j]));ctx.strokeStyle=`rgba(153,199,170,${.03+activity*.14})`;ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(xs[layer],20+i*170/Math.max(1,columns[layer].length-1));ctx.lineTo(xs[layer+1],20+j*170/Math.max(1,columns[layer+1].length-1));ctx.stroke();}
  columns.forEach((values,l)=>values.forEach((v,i)=>{ctx.fillStyle=v>=0?`rgba(180,233,189,${.25+Math.min(1,Math.abs(v))*.75})`:`rgba(238,175,111,${.3+Math.min(1,Math.abs(v))*.7})`;ctx.beginPath();ctx.arc(xs[l],20+i*170/Math.max(1,values.length-1),l===2?7:5,0,Math.PI*2);ctx.fill();}));
}
function inspect(ant:AntState):void {
  if(!ant)return;el('selected-label').textContent=`个体 ${String(ant.id).padStart(2,'0')}`;
  el('ant-state').textContent=(ant.carrying?'携带像素 · 回巢中':'局部探索中')+(ant.frozen?' · 参数冻结':'');
  el('freeze-one').textContent=ant.frozen?'恢复学习':'冻结';el('ant-updates').textContent=ant.updates.toLocaleString();
  el('ant-delta').textContent=ant.frozen?'未更新':ant.delta.toFixed(5);el('ant-drift').textContent=ant.drift.toFixed(4);el('ant-error').textContent=ant.error.toFixed(4);el('fingerprint').textContent=ant.fingerprint;network(ant);
}
function update(frame:Frame):void {
  if(!frame||!Array.isArray(frame.ants))return;
  if(!current||current.seed!==frame.seed||current.ants.length!==frame.ants.length){el<HTMLSelectElement>('ant-select').replaceChildren(...frame.ants.map(a=>{const option=document.createElement('option');option.value=String(a.id);option.textContent=`个体 ${String(a.id).padStart(2,'0')}`;return option;}));scene.selected=0;el<HTMLInputElement>('seed').value=String(frame.seed);generation++;}
  current=frame;scene.update(frame);app.dataset.tick=String(frame.tick);app.dataset.ready='true';app.dataset.generation=String(generation);
  el('connection').textContent=frame.paused?'仿真已暂停':'本地模型运行中';el('lamp').classList.add('online');
  el('clock').textContent=`${String(Math.floor(frame.seconds/60)).padStart(2,'0')}:${(frame.seconds%60).toFixed(1).padStart(4,'0')}`;
  el('population').textContent=String(frame.ants.length);el('fps').textContent=scene.fps.toFixed(0);el('delivered').textContent=String(frame.delivered);
  el('updates').textContent=frame.ants.reduce((sum,a)=>sum+a.updates,0).toLocaleString();el('contact-rate').textContent=(100*frame.contacts/Math.max(1,frame.samples)).toFixed(1);el('tick-ms').textContent=frame.tick_ms.toFixed(1);
  el('pause').textContent=frame.paused?'▶':'Ⅱ';el<HTMLButtonElement>('step').disabled=!frame.paused;el('speed').textContent=`${frame.rate}×`;
  el('freeze-all').textContent=frame.ants.every(a=>a.frozen)?'恢复全部学习':'冻结全部参数';
  if(document.activeElement!==el('wind')){el<HTMLInputElement>('wind').value=String(frame.wind);el('wind-value').textContent=frame.wind.toFixed(1);}
  inspect(frame.ants[scene.selected]??frame.ants[0]);
  const container=el('events');container.replaceChildren(...frame.events.slice(0,5).map(e=>{const row=document.createElement('div');row.className='event';const t=document.createElement('time');t.textContent=(e.tick/10).toFixed(1)+'s';const text=document.createElement('span');text.textContent=e.message;row.append(t,text);return row;}));
  const notice=frame.events.find(e=>e.kind==='notice'||e.kind==='error');if(notice){const key=notice.tick+notice.message;if(key!==lastToast){lastToast=key;toast(notice.message);}}
}
function connect():void {
  ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
  ws.onmessage=e=>{try{update(JSON.parse(String(e.data)) as Frame);}catch{toast('收到无效快照；保持上一帧');}};
  ws.onclose=()=>{el('connection').textContent='连接断开，正在重连';el('lamp').classList.remove('online');setTimeout(connect,1500);};
  ws.onerror=()=>ws.close();
}
connect();
