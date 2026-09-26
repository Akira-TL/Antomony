import {createElement,Play,Pause,StepForward,RotateCcw,Focus,Download,Save,MousePointer2,BrickWall,Apple,TriangleAlert,Eraser,Trash2,ChevronLeft,ChevronRight,Radio,X} from 'lucide';
import type {IconNode} from 'lucide';
import {api,labels} from './types';
import type {Control,Counts,Edit,Frame,GroupKey,Parameters,Preview,Tool} from './types';
import {LiveWorld} from './world';
import {ParameterCharts} from './charts';
import './style.css';

const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const icon=(id:string,node:IconNode)=>$(id).replaceChildren(createElement(node,{width:18,height:18,'stroke-width':1.7}));
const keys:GroupKey[]=['adaptive','mlp','rules'];
document.querySelector<HTMLDivElement>('#app')!.innerHTML=`
<header class="app-header"><div class="brand">Antomony <span>三组现场验收</span></div>
  <span class="qualification">研究状态：更新时机学习未通过</span><div class="transport">
  <button id="play" title="运行" aria-label="运行"></button><button id="step" title="单步" aria-label="单步"></button>
  <select id="speed" aria-label="运行倍速"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select>
  <span class="divider"></span><button id="fit" title="三组全景" aria-label="三组全景"></button>
  <button id="checkpoint" title="保存参数和记忆" aria-label="保存参数和记忆"></button><button id="export" title="导出本轮记录" aria-label="导出本轮记录"></button>
  <button id="new-run" title="新一轮" aria-label="新一轮"></button></div></header>
<main class="workspace">
  ${keys.map((key,i)=>`<section class="world-panel ${key}" aria-label="${labels[key]}">
    <div class="panel-heading"><div><span class="group-index">0${i+1}</span><strong>${labels[key]}</strong><span class="method">${i===0?'固定接受 · 4步判断':i===1?'17宽 · 冻结参数':'无神经网络'}</span></div><span id="brief-${key}" class="brief">等待状态</span></div>
    <div id="world-${key}" class="world-canvas"></div><div id="overlay-${key}" class="scene-overlay"></div>
  </section>`).join('')}
  <section class="control-panel"><div class="panel-heading"><nav class="tabs"><button id="tab-controls" class="active">控制与统计</button><button id="tab-parameters">参数与个体</button></nav><span id="mode" class="mode">连接中</span></div>
  <div id="controls-content" class="panel-scroll">
    <div class="section-line"><span>场景编辑</span><span class="legend"><i class="home-dot"></i>离巢轨迹 <i class="food-dot"></i>携食轨迹</span></div>
    <div class="tool-row"><div class="tools">${[['inspect','选择'],['wall','墙体'],['food','食物'],['trap','作用区'],['erase','移除']].map(([id,title])=>`<button id="tool-${id}" title="${title}" aria-label="${title}" aria-pressed="${id==='inspect'}"></button>`).join('')}</div>
      <label class="check"><input id="show-field" type="checkbox" checked>信息素</label><button id="clear" title="清空三组信息素" aria-label="清空三组信息素"></button></div>
    <div id="wall-settings" class="settings" hidden><label>长度<input id="wall-length" type="number" min="0.4" max="8" step="0.2" value="3"></label><label>厚度<input id="wall-width" type="number" min="0.4" max="8" step="0.2" value="0.5"></label><label>角度<input id="wall-angle" type="number" min="-180" max="180" step="15" value="0"></label></div>
    <div id="food-settings" class="settings" hidden><label>食物份数<input id="food-stock" type="number" min="1" max="4096" step="16" value="64"></label></div>
    <div id="trap-settings" class="settings trap-settings" hidden><label>类型<select id="trap-type"><option value="injury">持续伤害</option><option value="slow">减速</option><option value="periodic">周期伤害</option><option value="moving">移动伤害</option></select></label><label>半径<input id="trap-radius" type="number" min="0.2" max="2" step="0.1" value="0.6"></label><label id="injury-label">每步伤害<input id="trap-injury" type="number" min="0" max="0.25" step="0.01" value="0.08"></label><label id="slow-label" hidden>速度倍率<input id="trap-slow" type="number" min="0.1" max="1" step="0.1" value="0.3"></label><label id="period-label" hidden>周期步数<input id="trap-period" type="number" min="4" max="4096" step="4" value="64"></label><label id="motion-label" hidden>移动振幅<input id="trap-motion" type="number" min="0" max="2" step="0.2" value="1.2"></label></div>
    <div id="placement" class="placement" role="status">观察模式</div>
    <div class="learning-row"><label class="check"><input id="learning" type="checkbox" checked>允许受限参数写入</label><span id="saved">保存点 0</span></div>
    <table class="stats"><thead><tr><th>统计</th><th>自训练</th><th>MLP</th><th>规则</th><th>自训练 − MLP</th></tr></thead><tbody id="statistics"></tbody></table>
    <div id="notices" class="notices"></div>
  </div>
  <div id="parameters-content" class="panel-scroll" hidden><div class="parameter-selectors"><select id="group" aria-label="参数组"><option value="adaptive">自训练模型</option><option value="mlp">普通MLP</option><option value="rules">代码规则</option></select><label>个体<select id="individual" aria-label="个体"></select></label><button id="focus-ant" title="跟随所选个体" aria-label="跟随所选个体"></button><select id="module" aria-label="参数模块"></select><span id="frozen" class="freeze-state"></span></div>
    <div class="individual-line" id="individual-state"></div><div id="receptors" class="receptors"></div>
    <div class="chart-heading"><span id="chart-caption">实际参数 · 最近2048步</span><div><button id="prev-weights" title="上一组参数" aria-label="上一组参数"></button><span id="weight-count"></span><button id="next-weights" title="下一组参数" aria-label="下一组参数"></button></div></div>
    <div id="parameter-charts" class="parameter-charts"></div><div id="parameter-empty" class="empty" hidden>规则组不含神经网络参数</div>
  </div></section>
</main>
<footer><div class="timeline"><span id="tick">0 / 4096 步</span><input id="timeline" type="range" min="0" max="0" value="0" aria-label="本轮回看时点"><button id="return-live" title="返回当前现场" aria-label="返回当前现场"></button><span id="timeline-mode">现场</span></div><div class="status-row"><span id="connection">正在连接本地服务</span><span id="run-name"></span><span id="performance"></span></div></footer>
<div id="toast" role="alert" hidden></div>
<dialog id="reset-dialog"><form id="reset-form"><div class="dialog-heading"><h2>新一轮验收</h2><button type="button" id="close-reset" title="取消" aria-label="取消"></button></div><label>随机种子<input id="seed" type="number" min="0" max="33554431" value="20260927" required></label><div class="settings"><label>每组个体<input id="ants" type="number" min="1" max="32" value="32" required></label><label>总步数<input id="horizon" type="number" min="16" max="32768" step="16" value="4096" required></label><label>初始库存<input id="stock" type="number" min="1" max="20000" value="384" required></label></div><div class="dialog-status">当前记录保留 · 新一轮从暂停开始</div><button type="submit" class="primary">开始新一轮</button></form></dialog>`;

const icons:[string,IconNode][]=[['play',Play],['step',StepForward],['fit',Focus],['focus-ant',Focus],['checkpoint',Save],['export',Download],['new-run',RotateCcw],['tool-inspect',MousePointer2],['tool-wall',BrickWall],['tool-food',Apple],['tool-trap',TriangleAlert],['tool-erase',Eraser],['clear',Trash2],['prev-weights',ChevronLeft],['next-weights',ChevronRight],['return-live',Radio],['close-reset',X]];
icons.forEach(([id,node])=>icon(id,node));
let live:Frame|null=null,view:Frame|null=null,connected=false,busy=false,tool:Tool='inspect',replayTick:number|null=null,mutationVersion=0;
let selectedGroup:GroupKey='adaptive',selectedAnt=0,parameterData:Parameters|null=null,parameterTab=false;
let previewSerial=0,previewTimer=0,lastPoint:{x:number;y:number}|null=null,lastWorld:LiveWorld|null=null,replaySerial=0;
let toastTimer=0;
const worlds=keys.map(key=>new LiveWorld($(`world-${key}`),id=>{selectedGroup=key;selectedAnt=id;syncSelection();setTab(true);void refreshParameters(true);render();}));
const charts=new ParameterCharts($('parameter-charts'),$('weight-count'));
const number=(id:string)=>Number($(id) instanceof HTMLInputElement?($(id) as HTMLInputElement).value:0);
const signed=(x:number,decimal=false)=>`${x>0?'+':''}${decimal?x.toFixed(2):x}`;
function toast(message:string):void{clearTimeout(toastTimer);$('toast').textContent=message;$('toast').hidden=false;toastTimer=window.setTimeout(()=>$('toast').hidden=true,6500);}
async function attempt(fn:()=>Promise<void>):Promise<void>{try{await fn();}catch(e){toast(e instanceof Error?e.message:'操作失败');}}
async function command(value:Control):Promise<void>{
  if(busy)return;mutationVersion++;busy=true;renderControls();
  try{live=await api<Frame>('control',value);if(replayTick===null)view=live;}finally{busy=false;render();}
}
function renderControls():void{
  const disabled=!connected||busy||!live;
  for(const id of ['play','step','checkpoint','clear','learning','speed'])$<HTMLButtonElement|HTMLInputElement|HTMLSelectElement>(id).disabled=disabled||!!live?.error;
  ($('play') as HTMLButtonElement).disabled=disabled||!!live?.error||!!live?.done||replayTick!==null;
  ($('step') as HTMLButtonElement).disabled=disabled||!!live?.error||!!live?.done||!live?.paused||replayTick!==null;
  ($('clear') as HTMLButtonElement).disabled=disabled||!!live?.done||!!live?.error||replayTick!==null;
  ($('learning') as HTMLInputElement).disabled=disabled||!!live?.done||!!live?.error||replayTick!==null;
  for(const key of ['wall','food','trap','erase'])($(`tool-${key}`) as HTMLButtonElement).disabled=disabled||!!live?.done||!!live?.error||replayTick!==null;
  for(const w of worlds)w.setEditing(tool!=='inspect'&&connected&&!busy&&!live?.done&&!live?.error&&replayTick===null);
  ($('export') as HTMLButtonElement).disabled=!live||busy;
  ($('new-run') as HTMLButtonElement).disabled=!connected||busy;
  ($('return-live') as HTMLButtonElement).disabled=replayTick===null;
}
const stats:[keyof Counts,string,boolean][]=[['delivered','交付',true],['pickups','拾取',true],['deaths','危险死亡',false],['exhaustions','体力耗尽',false],['revivals','复活',false],['injury','累计伤害',false],['writes','参数写入',true],['decisions','更新判断',true],['stock','剩余库存',true]];
function render():void{
  if(!view||!live)return;
  worlds.forEach((world,i)=>{
    const state=view!.groups[i];world.render(view!,state,selectedGroup===state.key?selectedAnt:-1,$<HTMLInputElement>('show-field').checked);
    $(`brief-${state.key}`).textContent=`交付 ${state.counts.delivered} · 死亡 ${state.counts.deaths}`;
    $(`overlay-${state.key}`).textContent=`${state.counts.active}/${state.ants.length} 活动 · 库存 ${state.counts.stock}`;
  });
  $('statistics').innerHTML=stats.map(([key,label,higher])=>{const values=view!.groups.map(g=>g.counts[key]),diff=values[0]-values[1],decimal=key==='injury';
    const color=['writes','decisions','revivals','stock'].includes(key)?'':diff===0?'':(higher?diff>0:diff<0)?'positive':'negative';
    return`<tr><th>${label}</th>${values.map(v=>`<td>${decimal?v.toFixed(2):v}</td>`).join('')}<td class="${color}">${signed(diff,decimal)}</td></tr>`;}).join('');
  $('mode').textContent=live.error?'异常停止':replayTick!==null?'历史只读':live.done?'本轮结束':live.paused?'已暂停':'运行中';
  $('mode').classList.toggle('running',!live.paused&&replayTick===null);
  $('tick').textContent=`${view.tick} / ${live.horizon} 步`;
  const slider=$<HTMLInputElement>('timeline');slider.max=String(live.tick);slider.value=String(view.tick);
  $('timeline-mode').textContent=replayTick===null?'现场':'回看';
  $('saved').textContent=`保存点 ${live.checkpoint_tick}`;
  $<HTMLSelectElement>('speed').value=String(live.rate);$<HTMLInputElement>('learning').checked=live.learning;
  icon('play',live.paused?Play:Pause);$('play').title=live.paused?'运行':'暂停';$('play').setAttribute('aria-label',$('play').title);
  $('notices').replaceChildren(...view.notices.slice(0,3).map(n=>{const row=document.createElement('div');row.textContent=`第 ${n.tick} 步 · ${n.message}`;return row;}));
  $('run-name').textContent=`种子 ${live.seed} · ${live.run_id}`;
  $('performance').textContent=`三组推进 ${live.step_ms.toFixed(0)} ms`;
  if(live.error){$('connection').textContent=live.error;$('connection').className='negative';}else{$('connection').textContent=connected?'本地服务已连接 · 三组同步':'连接中断';$('connection').className=connected?'':'negative';}
  renderIndividual();renderControls();if(parameterTab)renderParameters();
}
function syncSelection():void{
  $<HTMLSelectElement>('group').value=selectedGroup;
  const select=$<HTMLSelectElement>('individual'),count=live?.groups[0].ants.length??32;
  if(select.options.length!==count)select.replaceChildren(...Array.from({length:count},(_,i)=>new Option(`#${String(i+1).padStart(2,'0')}`,String(i))));
  selectedAnt=Math.min(selectedAnt,count-1);select.value=String(selectedAnt);
}
function renderIndividual():void{
  const ant=view?.groups.find(g=>g.key===selectedGroup)?.ants[selectedAnt];if(!ant)return;
  $('individual-state').textContent=`${ant.pending?'等待复活':ant.carrying?'携食返巢':ant.exploration===0?'预算用尽 · 返巢':'空载探索'} · 探索 ${ant.exploration} · 返巢体力 ${ant.reserve} · 伤害 ${ant.injury.toFixed(2)} · 写入 ${ant.writes}`;
  $('receptors').replaceChildren(...ant.receptors.map((value,i)=>{const bar=document.createElement('div');bar.className=`receptor r${i+1}`;bar.title=`接收器 ${i+1}：${value.toPrecision(5)}`;bar.innerHTML=`<span>${i+1}</span><i style="height:${Math.max(0,Math.min(100,value/4*100))}%"></i>`;return bar;}));
}
function setTab(parameters:boolean):void{parameterTab=parameters;$('controls-content').hidden=parameters;$('parameters-content').hidden=!parameters;$('tab-controls').classList.toggle('active',!parameters);$('tab-parameters').classList.toggle('active',parameters);if(parameters)void refreshParameters(true);}
async function refreshParameters(reset=false):Promise<void>{
  if(!connected||!parameterTab||!live)return;const group=selectedGroup,id=selectedAnt,run=live.run_id;
  try{const data=await api<Parameters>(`parameters?group=${group}&individual=${id}`);if(group!==selectedGroup||id!==selectedAnt||run!==live?.run_id||data.run_id!==run)return;
    const changed=parameterData?.individual!==id||parameterData.group!==group||parameterData.run_id!==run;parameterData=data;
    const select=$<HTMLSelectElement>('module'),previous=select.value;
    if(changed||select.options.length!==data.modules.length){select.replaceChildren(...data.modules.map(m=>new Option(m.label,m.key)));if(data.modules.some(m=>m.key===previous))select.value=previous;}
    renderParameters(reset||changed);
  }catch(e){toast(e instanceof Error?e.message:'参数读取失败');}
}
function renderParameters(reset=false):void{
  if(!parameterData||!view)return;
  const module=parameterData.modules.find(m=>m.key===$<HTMLSelectElement>('module').value)??null;
  const available=module?.points.some(p=>p.tick<=view!.tick);
  $('parameter-empty').hidden=!!available;
  $('parameter-empty').textContent=parameterData.modules.length?'该时点无折线采样':'规则组不含神经网络参数';
  const frozen=module?.key==='adaptive'?!view.learning:module?.frozen;
  $('frozen').textContent=module?(frozen?'已冻结':'可写入'):'';
  $('frozen').classList.toggle('writable',!!module&&!frozen);
  $('chart-caption').textContent=module?.key==='adaptive'?'实际方向修正参数 · 最近2048步':module?`${module.label} · 实际参数`:'';
  charts.set(module,view.tick,reset);
}
function setTool(next:Tool):void{
  tool=next;previewSerial++;clearTimeout(previewTimer);lastPoint=null;worlds.forEach(w=>w.showPreview(null,null));
  for(const name of ['inspect','wall','food','trap','erase']){$(`tool-${name}`).setAttribute('aria-pressed',String(name===next));}
  for(const name of ['wall','food','trap'])$(`${name}-settings`).hidden=name!==next;
  $('placement').textContent=next==='inspect'?'观察模式':`${{wall:'墙体',food:'食物',trap:'作用区',erase:'移除'}[next]} · 待选择落点`;$('placement').className='placement';renderControls();
}
function makeEdit(p:{x:number;y:number},world:LiveWorld):Edit|null{
  if(tool==='wall')return{kind:'wall',...p,hx:number('wall-width')/2,hy:number('wall-length')/2,angle:number('wall-angle')*Math.PI/180};
  if(tool==='food')return{kind:'food',...p,stock:number('food-stock')};
  if(tool==='trap'){const type=$<HTMLSelectElement>('trap-type').value;return{kind:'trap',...p,radius:number('trap-radius'),injury:type==='slow'?0:number('trap-injury'),speed_multiplier:type==='slow'?number('trap-slow'):1,period:type==='periodic'?number('trap-period'):0,motion_amplitude:type==='moving'?number('trap-motion'):0};}
  return tool==='erase'?world.eraseAt(p.x,p.y):null;
}
function movePreview(p:{x:number;y:number}|null,world:LiveWorld):void{
  lastPoint=p;lastWorld=world;const serial=++previewSerial;clearTimeout(previewTimer);
  if(!p||tool==='inspect'||replayTick!==null){worlds.forEach(w=>w.showPreview(null,null));return;}
  const edit=makeEdit(p,world);worlds.forEach(w=>w.showPreview(edit,null));
  if(!edit){$('placement').textContent='当前位置没有可移除对象';return;}
  previewTimer=window.setTimeout(()=>void attempt(async()=>{
    const result=await api<Preview>('preview',edit);if(serial!==previewSerial)return;
    worlds.forEach(w=>w.showPreview(edit,result.valid));$('placement').textContent=result.message;$('placement').className=`placement ${result.valid?'positive':'negative'}`;
  }),90);
}
worlds.forEach(world=>{world.onMove=p=>movePreview(p,world);world.onPlace=p=>void attempt(async()=>{
  if(busy||!connected||replayTick!==null||live?.done||live?.error)return;
  const edit=makeEdit(p,world);if(!edit)return;
  mutationVersion++;busy=true;renderControls();
  try{const result=await api<Preview>('edit',edit);if(!result.valid){toast(result.message);return;}live=await api<Frame>('state');view=live;render();movePreview(p,world);}finally{busy=false;renderControls();}
});});
async function poll():Promise<void>{
  if(busy){window.setTimeout(()=>void poll(),250);return;}
  const version=mutationVersion;
  try{const frame=await api<Frame>('state');
    if(version!==mutationVersion){window.setTimeout(()=>void poll(),250);return;}
    connected=true;
    if(live&&live.run_id!==frame.run_id){replayTick=null;parameterData=null;selectedAnt=0;setTool('inspect');}
    live=frame;if(replayTick===null)view=frame;syncSelection();render();
  }catch{connected=false;$('connection').textContent='连接中断 · 正在重试';$('connection').className='negative';renderControls();}
  window.setTimeout(()=>void poll(),250);
}
$('play').onclick=()=>void attempt(()=>command({kind:'pause',enabled:!live?.paused}));
$('step').onclick=()=>void attempt(()=>command({kind:'step'}));
$('fit').onclick=()=>worlds.forEach(w=>w.fit());
$('focus-ant').onclick=()=>{setTool('inspect');worlds[keys.indexOf(selectedGroup)].focus(selectedAnt);};
$('checkpoint').onclick=()=>void attempt(async()=>{await command({kind:'checkpoint'});toast(`参数与记忆已保存，第 ${live?.checkpoint_tick} 步`);});
$('speed').onchange=()=>void attempt(()=>command({kind:'speed',rate:Number($<HTMLSelectElement>('speed').value)}));
$('learning').onchange=()=>void attempt(async()=>{await command({kind:'learning',enabled:$<HTMLInputElement>('learning').checked});await refreshParameters();});
$('clear').onclick=()=>void attempt(async()=>{if(!confirm('清空三组当前信息素？个体参数和记忆将保留。'))return;const result=await api<Preview>('edit',{kind:'clear-trails'});if(!result.valid)throw new Error(result.message);});
$('show-field').onchange=render;
$('tab-controls').onclick=()=>setTab(false);$('tab-parameters').onclick=()=>setTab(true);
$('group').onchange=()=>{selectedGroup=$<HTMLSelectElement>('group').value as GroupKey;void refreshParameters(true);render();};
$('individual').onchange=()=>{selectedAnt=Number($<HTMLSelectElement>('individual').value);void refreshParameters(true);render();};
$('module').onchange=()=>renderParameters(true);
$('prev-weights').onclick=()=>charts.turn(-1);$('next-weights').onclick=()=>charts.turn(1);
(['inspect','wall','food','trap','erase'] as Tool[]).forEach(name=>$(`tool-${name}`).onclick=()=>setTool(name));
document.querySelectorAll<HTMLInputElement|HTMLSelectElement>('.settings input,.settings select').forEach(input=>input.addEventListener('change',()=>{if(lastPoint&&lastWorld)movePreview(lastPoint,lastWorld);}));
$('trap-type').onchange=()=>{const type=$<HTMLSelectElement>('trap-type').value;$('injury-label').hidden=type==='slow';$('slow-label').hidden=type!=='slow';$('period-label').hidden=type!=='periodic';$('motion-label').hidden=type!=='moving';};
document.addEventListener('keydown',event=>{if(event.key==='Escape')setTool('inspect');});
$('return-live').onclick=()=>{replaySerial++;replayTick=null;view=live;render();};
$('timeline').oninput=()=>void attempt(async()=>{
  const tick=Number($<HTMLInputElement>('timeline').value),serial=++replaySerial;replayTick=tick;setTool('inspect');
  if(live&&!live.paused)await command({kind:'pause',enabled:true});
  const frame=await api<Frame>(`replay?tick=${tick}`);if(serial!==replaySerial||frame.run_id!==live?.run_id)return;
  view=frame;render();
});
$('export').onclick=()=>void attempt(async()=>{busy=true;renderControls();try{const result=await api<{url:string}>('export',{});const link=document.createElement('a');link.href=result.url;link.download='';link.click();toast('完整运行记录已导出');}finally{busy=false;renderControls();}});
$('new-run').onclick=()=>{$<HTMLInputElement>('seed').value=String(live?.seed??20260927);$<HTMLInputElement>('ants').value=String(live?.groups[0].ants.length??32);$<HTMLInputElement>('horizon').value=String(live?.horizon??4096);$<HTMLDialogElement>('reset-dialog').showModal();};
$('close-reset').onclick=()=>$<HTMLDialogElement>('reset-dialog').close();
$<HTMLFormElement>('reset-form').onsubmit=event=>{event.preventDefault();void attempt(async()=>{
  if(busy)return;mutationVersion++;busy=true;renderControls();
  try{const frame=await api<Frame>('reset',{seed:number('seed'),ants:number('ants'),horizon:number('horizon'),stock:number('stock')});
    live=frame;view=frame;replayTick=null;parameterData=null;selectedAnt=0;syncSelection();setTool('inspect');render();$<HTMLDialogElement>('reset-dialog').close();worlds.forEach(w=>w.fit());void refreshParameters(true);
  }finally{busy=false;renderControls();}
});};
syncSelection();renderControls();void poll();window.setInterval(()=>void refreshParameters(),2500);
