import {createElement,Play,Pause,StepForward,RotateCcw,Focus,ChevronLeft,ChevronRight} from 'lucide';
import type {IconNode} from 'lucide';
import {counts,jsonLines,pointAt} from './model';
import type {Arm,Catalog,Frame,Header,Tape,Update,WeightGroup} from './model';
import {WorldView} from './world';
import './style.css';

const $=<T extends HTMLElement=HTMLElement>(selector:string):T=>document.querySelector<T>(selector)!;
const icon=(node:IconNode)=>createElement(node,{width:17,height:17,'stroke-width':1.8}).outerHTML;
const button=(id:string,title:string,node:IconNode)=>`<button type="button" id="${id}" title="${title}" aria-label="${title}">${icon(node)}</button>`;
const names:Record<Arm,string>={learned:'学习接受',skip:'全部跳过',always:'固定接受',mlp:'普通 MLP',rules:'纯规则'};
const conditions:Record<string,string>={reference:'正常参照',slow:'持续减速',periodic:'周期减速','moving-danger':'移动危险'};
$('#app').innerHTML=`<header><div><h1>连续对照验收</h1><small>MathHackson · 20260926T123655-2</small></div><strong id="outcome">读取已登记结果</strong></header>
<div id="error" role="alert" hidden></div><div class="controls"><label>场景<select id="condition" aria-label="场景"></select></label><label>种子<select id="seed" aria-label="世界种子"></select></label><label>参照<select id="comparator" aria-label="参照组">${(['always','skip','mlp','rules'] as Arm[]).map(a=>`<option value="${a}">${names[a]}</option>`).join('')}</select></label><span id="loading" role="status">读取中</span>
<div class="transport">${button('play','播放记录',Play)}${button('step','下一步',StepForward)}${button('reset','回到起点',RotateCcw)}${button('focus','聚焦巢食区域',Focus)}<label>速度<select id="speed" aria-label="播放速度"><option value="1">1×</option><option value="4" selected>4×</option><option value="16">16×</option></select></label></div></div>
<div class="timeline"><output id="time">0 / 768 步</output><input id="time-slider" type="range" min="0" max="768" value="0" aria-label="记录时间步" step="1" /></div>
<section class="scenes">${[0,1].map(i=>`<div class="world"><div class="world-head"><strong id="name-${i}"></strong><span id="end-${i}"></span></div><div class="stats" id="stats-${i}"></div><div id="scene-${i}" class="scene"></div></div>`).join('')}</section>
<div class="comparison"><span id="difference"></span><span id="run-note">记录回放</span></div>
<section class="inspector"><div class="inspector-head"><h2>个体参数</h2><label>组别<select id="side" aria-label="参数所属组"><option value="0">学习接受</option><option value="1">参照组</option></select></label><label>个体<select id="ant" aria-label="个体编号">${Array.from({length:8},(_,i)=>`<option value="${i}">${i+1}</option>`).join('')}</select></label><label>参数<select id="group" aria-label="参数模块"></select></label><span id="freeze" class="freeze"></span></div>
<div class="details"><div><div id="charts" class="charts"></div><div class="pager">${button('previous','上一组参数',ChevronLeft)}<span id="page"></span>${button('next','下一组参数',ChevronRight)}</div></div><aside class="side"><h3>当前个体</h3><div id="ant-state" class="ant-state"></div><h3>中心位置接收器</h3><div id="receptors" class="receptors"></div><h3>最近更新决策</h3><div id="events" class="events"></div></aside></div></section>
<footer>数据：80 世界 · 4 个配对种子 · 接受策略未通过采用条件 · 全场信息素快照未记录</footer><div id="tooltip" class="tooltip" role="status" hidden></div>`;

let catalog:Catalog,tapes:Tape[]=[],groups:WeightGroup[]=[],tick=0,page=0,playing=false,busy=true,revision=0,weightRevision=0;
const views=[0,1].map(i=>new WorldView($(`#scene-${i}`),id=>{select('#ant',String(id));select('#side',String(i));void loadWeights();}));
const value=(id:string)=>$<HTMLSelectElement>(id).value;
const select=(id:string,next:string)=>{$<HTMLSelectElement>(id).value=next;};
const failure=(error:unknown)=>{$('#error').textContent=error instanceof Error?error.message:String(error);$('#error').hidden=false;};
const url=(arm:Arm)=>`/api/acceptance/${value('#condition')}/${value('#seed')}/${arm}`;
async function response(path:string):Promise<Response>{const result=await fetch(path,{signal:AbortSignal.timeout(20000)});if(!result.ok)throw new Error(`读取失败 ${result.status}`);return result;}
function setPlaying(next:boolean):void{playing=next;$('#play').innerHTML=icon(next?Pause:Play);$('#play').title=next?'暂停记录':'播放记录';$('#play').setAttribute('aria-label',$('#play').title);if(!busy)$('#loading').textContent=next?'播放中':'已暂停';}
function controls():void{document.querySelectorAll<HTMLButtonElement|HTMLSelectElement|HTMLInputElement>('.controls button,.controls select,.timeline input').forEach(e=>e.disabled=busy);}

async function loadTape(arm:Arm):Promise<Tape>{
  const base=url(arm),header=await (await response(`${base}/header`)).json() as Header;
  const frames=jsonLines<Frame>(await (await response(`${base}/trace`)).text());
  const updates=jsonLines<Update>(await (await response(`${base}/updates`)).text());
  if(frames.length!==header.result.steps||frames.some((frame,i)=>frame.tick!==i+1))throw new Error('轨迹记录不完整');
  return {header,frames,updates,counts:counts(frames)};
}
async function loadWorlds():Promise<void>{
  const current=++revision;weightRevision++;busy=true;setPlaying(false);controls();$('#loading').textContent='读取中';$('#error').hidden=true;
  try{
    const next=await Promise.all([loadTape('learned'),loadTape(value('#comparator') as Arm)]);
    if(current!==revision)return;tapes=next;tick=0;
    tapes.forEach((tape,i)=>{views[i].setTape(tape);$(`#name-${i}`).textContent=names[tape.header.result.arm];});
    $('#side').children[1].textContent=names[tapes[1].header.result.arm];
    await loadWeights();render();$('#loading').textContent='已暂停';
  }catch(error){if(current===revision){tapes=[];failure(error);$('#loading').textContent='读取失败';}}
  finally{if(current===revision){busy=false;controls();}}
}
function groupName(name:string):string{
  if(name==='direction.offset')return '方向修正 · 45 参数';
  if(name.startsWith('decision.'))return name.endsWith('weight')?'接受 · 预测差权重':'接受 · 预测差偏置';
  return name.replace('policy.base.','基础 · ').replace('policy.','策略 · ').replace('motor.','动作 · ').replace('decision.','接受 · ')
    .replace('recent_weights','近期记忆').replace('sparse_weights','稀疏记忆').replace('encoder','输入层').replace('middle','隐藏层').replace('direction','方向层').replace('value','价值输出').replace('readout','输出层').replace(/[_\.]weight/g,'权重').replace(/[_\.]bias/g,'偏置');
}
async function loadWeights():Promise<void>{
  if(tapes.length!==2)return;const current=++weightRevision;groups=[];$('#charts').replaceChildren();$('#freeze').textContent='读取参数';
  const tape=tapes[Number(value('#side'))],individual=Number(value('#ant'));
  try{
    const next=await (await response(`${url(tape.header.result.arm)}/weights/${individual}`)).json() as WeightGroup[];
    if(current!==weightRevision)return;groups=next;page=0;
    $<HTMLSelectElement>('#group').replaceChildren(...groups.map((group,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=groupName(group.name);return option;}));
    renderCharts();render();
  }catch(error){if(current===weightRevision){failure(error);$('#freeze').textContent='参数读取失败';}}
}
function selectedGroup():WeightGroup|undefined{return groups[Number(value('#group'))||0];}
function renderCharts():void{
  const group=selectedGroup(),size=group?.points[0].values.length??0;
  const start=page*12,end=Math.min(size,start+12);
  $('#freeze').textContent=group?(group.frozen?'冻结':'可更新'):`${tapes[Number(value('#side'))]?.header.result.arm==='rules'?'纯规则，无神经参数':'暂无参数'}`;
  $('#charts').innerHTML=Array.from({length:end-start},(_,i)=>`<div class="weight"><span>[${start+i}]</span><canvas width="360" height="100" data-index="${start+i}" aria-label="参数 ${start+i} 的实际权重曲线"></canvas></div>`).join('');
  $('#page').textContent=size?`${start+1}–${end} / ${size}`:'无参数';
  $<HTMLButtonElement>('#previous').disabled=page===0;$<HTMLButtonElement>('#next').disabled=end>=size;
  document.querySelectorAll<HTMLCanvasElement>('.weight canvas').forEach(canvas=>{
    canvas.addEventListener('pointermove',event=>{
      const current=selectedGroup();if(!current)return;
      const rect=canvas.getBoundingClientRect(),at=Math.round(Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))*768);
      const parameter=Number(canvas.dataset.index),point=pointAt(current.points,at);
      $('#tooltip').textContent=`参数 [${parameter}] · 第 ${at} 步后\n${point.values[parameter].toPrecision(9)}\n${current.frozen?'冻结参数':`最近记录：第 ${point.tick} 步`}`;
      $('#tooltip').style.left=`${Math.min(window.innerWidth-270,event.clientX+12)}px`;$('#tooltip').style.top=`${Math.max(8,event.clientY-78)}px`;$('#tooltip').hidden=false;
    });canvas.addEventListener('pointerleave',()=>{$('#tooltip').hidden=true;});
  });drawCharts();
}
function drawCharts():void{
  const group=selectedGroup();if(!group)return;
  document.querySelectorAll<HTMLCanvasElement>('.weight canvas').forEach(canvas=>{
    const index=Number(canvas.dataset.index),values=group.points.map(p=>p.values[index]);
    const low=Math.min(...values),high=Math.max(...values),padding=Math.max((high-low)*.12,1e-5);
    const y=(v:number)=>90-(v-low+padding)/(high-low+2*padding)*80;
    const ctx=canvas.getContext('2d')!;ctx.clearRect(0,0,360,100);ctx.strokeStyle=group.frozen?'#9facb6':'#65cfbb';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(0,y(values[0]));let last=values[0];
    for(const point of group.points){const x=point.tick/768*360;ctx.lineTo(x,y(last));last=point.values[index];ctx.lineTo(x,y(last));}
    ctx.lineTo(360,y(last));ctx.stroke();ctx.strokeStyle='#d8b169';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(tick/768*360,0);ctx.lineTo(tick/768*360,100);ctx.stroke();
  });
}
function render():void{
  if(tapes.length!==2)return;const individual=Number(value('#ant')),side=Number(value('#side'));
  $('#time').textContent=`${tick} / 768 步`;$<HTMLInputElement>('#time-slider').value=String(tick);
  tapes.forEach((tape,i)=>{
    const count=tape.counts[Math.min(tick,tape.frames.length)];views[i].render(tick,individual);
    $(`#end-${i}`).textContent=tick>=tape.frames.length?`记录结束于 ${tape.frames.length} 步`:`${tape.frames.length} 步记录`;
    $(`#stats-${i}`).innerHTML=`<span>交付<b>${count.deliveries}</b></span><span>死亡<b>${count.deaths}</b></span><span>耗尽<b>${count.exhausted}</b></span><span>写入<b>${count.writes}</b></span>`;
  });
  const a=tapes[0].counts[Math.min(tick,tapes[0].frames.length)],b=tapes[1].counts[Math.min(tick,tapes[1].frames.length)],delta=a.deliveries-b.deliveries;
  $('#difference').innerHTML=`学习接受 − ${names[tapes[1].header.result.arm]}：交付 <b class="${delta<0?'negative':''}">${delta>0?'+':''}${delta}</b> · 死亡 <b>${a.deaths-b.deaths}</b>`;
  const tape=tapes[side],actual=Math.min(tick,tape.frames.length),ant=actual?tape.frames[actual-1].ants[individual]:null;
  $('#run-note').textContent=tape.header.result.condition==='reference'?'记录回放 · 正常条件，无新增干预':'记录回放 · 外部干预从第 128 步起';
  const group=selectedGroup();if(group&&!group.frozen)$('#freeze').textContent=ant?.exhausted?'个体已终止':'可更新';
  $('#ant-state').textContent=ant?`${ant.killed?'死亡':ant.exhausted?'体力耗尽':ant.carrying?'携食返回':'空载'} · ${ant.move?'前进':'停步'} · 转向 ${(ant.turn*10).toFixed(2)}°\n探索 ${ant.exploration_left} · 返巢 ${ant.reserve_left} · 写入 ${ant.writes}`:'初始状态';
  $('#receptors').innerHTML=Array.from({length:8},(_,i)=>{const v=ant?.observation[i]??0;return `<div class="receptor" title="接收器 ${i+1}：${v.toPrecision(7)}"><i style="height:${Math.min(50,50*v/(1+v))}px"></i><span>${i+1}</span></div>`;}).join('');
  const events=tape.updates.filter(u=>u.individual===individual&&u.tick<=tick).slice(-4).reverse();
  $('#events').innerHTML=events.length?events.map(u=>`<div>第 ${u.tick} 步 · ${u.changed?'实际写入':u.accepted?'接受但未变化':u.eligible?'跳过':'无可执行更新'}${u.prediction!==null?` · 预测差 ${u.prediction.toFixed(5)}`:''}</div>`).join(''):'暂无更新决策';
  drawCharts();
}
for(const id of ['#condition','#seed','#comparator'])$(id).addEventListener('change',()=>void loadWorlds());
for(const id of ['#side','#ant'])$(id).addEventListener('change',()=>void loadWeights());
$('#group').addEventListener('change',()=>{page=0;renderCharts();});
$('#previous').addEventListener('click',()=>{page=Math.max(0,page-1);renderCharts();});
$('#next').addEventListener('click',()=>{page++;renderCharts();});
$('#play').addEventListener('click',()=>{if(busy||!tapes.length)return;if(tick>=768)tick=0;setPlaying(!playing);});
$('#step').addEventListener('click',()=>{setPlaying(false);tick=Math.min(768,tick+1);render();});
$('#reset').addEventListener('click',()=>{setPlaying(false);tick=0;render();});
$('#focus').addEventListener('click',()=>views.forEach(view=>view.focus()));
$('#time-slider').addEventListener('input',()=>{setPlaying(false);tick=Number($<HTMLInputElement>('#time-slider').value);render();});
let last=performance.now(),remaining=0;
function animate(now:number):void{
  const elapsed=Math.min(250,now-last);last=now;
  if(playing&&!busy){remaining+=elapsed*Number(value('#speed'))/100;if(remaining>=1){tick=Math.min(768,tick+Math.floor(remaining));remaining%=1;render();if(tick===768)setPlaying(false);}}
  else remaining=0;requestAnimationFrame(animate);
}
requestAnimationFrame(animate);controls();
async function start():Promise<void>{
  try{
    catalog=await (await response('/api/acceptance/catalog')).json() as Catalog;
    $('#outcome').textContent=catalog.summary.development_continue?'通过本批开发条件':'接受策略未通过采用条件';
    $('#condition').innerHTML=catalog.execution.plan.conditions.map(c=>`<option value="${c.name}">${conditions[c.name]}</option>`).join('');
    $('#seed').innerHTML=catalog.execution.plan.seeds.map(seed=>`<option>${seed}</option>`).join('');
    await loadWorlds();
  }catch(error){failure(error);$('#loading').textContent='载入失败';}
}
void start();
