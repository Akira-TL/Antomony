import {createElement, FlaskConical, Play, Pause, StepForward, RotateCcw, Focus} from 'lucide';
import type {IconNode} from 'lucide';
import * as THREE from 'three';
import {ColonyScene} from '../scene';
import type {RoundTripCommand, RoundTripState, RecurrentPhase, WriteMode} from './types';
import './style.css';
import './roundtrip-style.css';

const $=<T extends HTMLElement=HTMLElement>(selector:string):T=>document.querySelector<T>(selector)!;
const icon=(node:IconNode)=>createElement(node,{width:18,height:18,'stroke-width':1.7}).outerHTML;
const button=(id:string,title:string,node:IconNode)=>`<button type="button" class="icon-button" id="${id}" title="${title}" aria-label="${title}">${icon(node)}</button>`;
const phaseNames:Record<RecurrentPhase,string>={motor:'基础动作',memory:'双向循迹',adaptive:'条件自修改',autonomous:'停止外部训练'};

$('#app').innerHTML=`
  <header><div class="identity">${icon(FlaskConical)}<div><span>MathHackson</span><h1>往返训练验收</h1></div></div>
    <div class="session-status"><span id="connection" role="status">连接中</span><span class="muted" id="session-id"></span></div>
  </header>
  <main>
    <div class="toolbar"><div class="segmented phases" role="group" aria-label="训练阶段">
      <button type="button" data-phase="memory" aria-pressed="true">双向循迹</button>
      <button type="button" data-phase="adaptive" aria-pressed="false">条件自修改</button>
      <button type="button" data-phase="autonomous" aria-pressed="false">停止外部训练</button>
    </div><div class="run-controls">${button('play','开始训练',Play)}${button('step','单步',StepForward)}${button('reset','重置回合并保留参数',RotateCcw)}
      <label class="speed-label">速度<select id="speed" aria-label="运行速度"><option value="1">1×</option><option value="4">4×</option><option value="16">16×</option></select></label>
      <label class="speed-label">写入<select id="write-mode" aria-label="运行时写入方式"><option value="off">关闭</option><option value="learned">模型判断</option><option value="always">始终写入</option></select></label>
    </div></div>
    <div id="error" role="alert" hidden></div>
    <div class="workspace">
      <section class="experiment"><div class="section-heading"><h2>单蚁往返</h2><span id="running" class="state-tag">已暂停</span></div>
        <div class="scene-wrap"><div id="scene"></div>
          <div class="scene-caption"><span id="trip-status">去程</span><span id="position">x 0.00 · y 0.00</span></div>
          <div class="camera-tools">${button('focus','聚焦巢穴与食物',Focus)}</div>
          <div class="scene-bottom"><span class="scent-key"><i class="home-swatch"></i>归巢信息素 <i class="food-swatch"></i>食物信息素</span><span id="scents">局部浓度 0.00 / 0.00</span></div>
        </div>
        <div class="telemetry"><div><span>当前回合</span><strong id="episode">1</strong><small id="step-count">0 / 256 步</small></div>
          <div><span>拾取 / 交付</span><strong id="deliveries">0 / 0</strong><small id="cargo">未携食</small></div>
          <div><span>本步动作</span><strong id="action">—</strong><small id="action-detail">等待推理</small></div>
          <div><span>本回合奖励</span><strong id="reward">0.000</strong><small id="step-reward">本步 0.000</small></div></div>
        <progress id="progress" value="0" max="256"></progress>
        <div class="history-heading"><h2>回合记录</h2><span class="muted" id="history-count">尚无完整回合</span></div>
        <div class="history-scroll"><table><thead><tr><th>回合</th><th>阶段</th><th>拾取</th><th>交付</th><th>步数</th><th>写入</th><th>奖励</th><th>快照</th></tr></thead><tbody id="history"></tbody></table></div>
      </section>
      <section class="network"><div class="section-heading"><h2>局部信号与写入</h2><span class="muted" id="training-count">—</span></div>
        <div class="signal-row"><i class="home-swatch"></i><span>归巢释放</span><strong id="home-release">—</strong><small id="home-probability">—</small></div>
        <div class="signal-row"><i class="food-swatch"></i><span>食物释放</span><strong id="food-release">—</strong><small id="food-probability">—</small></div>
        <div class="signal-row"><span>写入判断</span><strong id="write-status">尚未推理</strong><small id="write-probability">—</small></div>
        <div class="signal-row"><span>外部 / 临时更新</span><strong id="updates">0 / 0</strong><small id="fast-values">—</small></div>
        <div class="section-heading"><h2>参数冻结</h2><span class="muted">当前阶段</span></div>
        <div id="groups" class="roundtrip-groups"></div>
      </section>
    </div>
    <footer><span>第 2570 回合基础模型 · 动作参数冻结</span><span>开发验收；不代表自修改优势已验证</span></footer>
  </main>`;

const scene=new ColonyScene($('#scene'),true);
scene.showField=true;
const nest=new THREE.Mesh(new THREE.CylinderGeometry(1.2,1.25,.09,48),new THREE.MeshStandardMaterial({color:0x326d66,roughness:.65}));
nest.position.y=.05;scene.scene.add(nest);
const nestRing=new THREE.Mesh(new THREE.RingGeometry(1.08,1.17,48),new THREE.MeshBasicMaterial({color:0x67d7e7,side:THREE.DoubleSide}));
nestRing.rotation.x=-Math.PI/2;nestRing.position.y=.11;scene.scene.add(nestRing);
let current:RoundTripState|null=null;
let connected=false,busy=false,revision=0,lastTick=-1,lastEpisode=-1,lastSession='',historyKey='';

function focus():void {
  if(!current)return;
  const x=current.food_x/2,z=current.food_y/2;
  scene.controls.target.set(x,0,z);scene.camera.position.set(x+2,9,z+8);scene.camera.lookAt(x,0,z);scene.controls.update();
}

function render(state:RoundTripState):void {
  current=state;connected=true;
  $('#connection').textContent='已连接';$('#connection').className='connected';
  $('#session-id').textContent=state.session;
  $('#running').textContent=state.paused?'已暂停':'运行中';$('#running').classList.toggle('live',!state.paused);
  $('#play').innerHTML=icon(state.paused?Play:Pause);$('#play').title=state.paused?'开始训练':'暂停';$('#play').setAttribute('aria-label',$('#play').title);
  document.querySelectorAll<HTMLButtonElement>('[data-phase]').forEach(control=>control.setAttribute('aria-pressed',String(control.dataset.phase===state.phase)));
  $<HTMLSelectElement>('#speed').value=String(state.speed);$<HTMLSelectElement>('#write-mode').value=state.write_mode;
  $<HTMLSelectElement>('#write-mode').disabled=busy||state.phase!=='autonomous';
  $('#trip-status').textContent=state.carrying?'返程 · 携食':state.delivered?'再次出巢':'首趟出巢';
  $('#position').textContent=`x ${state.x.toFixed(2)} · y ${state.y.toFixed(2)}`;
  $('#scents').textContent=`局部浓度 ${state.home_scent.toFixed(3)} / ${state.food_scent.toFixed(3)}`;
  $('#episode').textContent=String(state.episode);$('#step-count').textContent=`${state.steps} / ${state.horizon} 步`;
  $('#deliveries').textContent=`${state.pickups} / ${state.delivered}`;$('#cargo').textContent=state.carrying?'正在搬运':'未携食';
  $('#action').textContent=state.steps?`${state.move?'前进':'停步'} · ${(state.turn*10).toFixed(1)}°`:'—';
  $('#action-detail').textContent=state.steps?`前进概率 ${(state.move_probability*100).toFixed(1)}%`:'等待推理';
  $('#reward').textContent=state.total_reward.toFixed(3);$('#step-reward').textContent=`本步 ${state.reward.toFixed(3)}`;
  $<HTMLProgressElement>('#progress').max=state.horizon;$<HTMLProgressElement>('#progress').value=state.steps;
  $('#home-release').textContent=state.steps?(state.release_home?'释放':'跳过'):'—';
  $('#food-release').textContent=state.steps?(state.release_food?'释放':'跳过'):'—';
  $('#home-probability').textContent=`概率 ${(state.release_home_probability*100).toFixed(1)}%`;
  $('#food-probability').textContent=`概率 ${(state.release_food_probability*100).toFixed(1)}%`;
  $('#write-status').textContent=state.write_status;$('#write-probability').textContent=`判断概率 ${(state.write_probability*100).toFixed(1)}%`;
  $('#updates').textContent=`${state.outer_updates} / ${state.self_updates}`;
  $('#fast-values').textContent=`动作 ${state.fast.map(v=>v.toFixed(3)).join(' / ')} · 释放 ${state.release_fast.map(v=>v.toFixed(3)).join(' / ')}`;
  const trainable=state.groups.reduce((sum,group)=>sum+group.trainable.flat().filter(Boolean).length,0);
  $('#training-count').textContent=`可训练 ${trainable}`;
  $('#groups').innerHTML=state.groups.map(group=>{
    const total=group.values.flat().length,active=group.trainable.flat().filter(Boolean).length;
    const maxChange=Math.max(0,...group.changes.flat().map(Math.abs));
    return `<div class="roundtrip-group"><span>${group.label}</span><strong>${active} / ${total}</strong><small>末次最大变化 ${maxChange.toFixed(5)}</small></div>`;
  }).join('');
  const nextHistoryKey=`${state.session}:${state.episode}:${state.history.length}`;
  if(nextHistoryKey!==historyKey){
    $('#history-count').textContent=state.history.length?`最近 ${state.history.length} 回合`:'尚无完整回合';
    $('#history').innerHTML=state.history.slice(-16).reverse().map(record=>`<tr><td>${record.episode}</td><td>${phaseNames[record.phase]}${record.completed?'':'<small>中断</small>'}</td><td>${record.pickups}</td><td>${record.delivered}</td><td>${record.steps}</td><td>${record.writes}</td><td class="${record.reward<0?'negative':'positive'}">${record.reward.toFixed(2)}</td><td><a href="/api/roundtrip/checkpoints/${record.episode}" download>下载</a></td></tr>`).join('');
    historyKey=nextHistoryKey;
  }
  if(state.tick!==lastTick||state.episode!==lastEpisode||state.session!==lastSession){
    scene.update({mode:'neural',tick:state.tick,seconds:state.steps*.1,seed:state.episode,delivered:0,
      field_width:state.field_width,field_height:state.field_height,pheromones:state.pheromones,walls:[],
      foods:[{id:0,x:state.food_x,y:state.food_y,amount:16}],
      ants:[{id:0,x:state.x,y:state.y,heading:state.heading,carrying:state.carrying,
        frozen:state.phase==='autonomous',rays:[],sense_x:state.x,sense_y:state.y,sense_heading:state.heading}]});
    if(state.episode!==lastEpisode||state.session!==lastSession)focus();
    lastTick=state.tick;lastEpisode=state.episode;lastSession=state.session;
  }
  $('#error').textContent=state.error;$('#error').hidden=!state.error;
  document.querySelectorAll<HTMLButtonElement|HTMLSelectElement>('.toolbar button,.toolbar select').forEach(control=>control.disabled=busy||!connected);
  $<HTMLSelectElement>('#write-mode').disabled=busy||!connected||state.phase!=='autonomous';
}

async function send(command:RoundTripCommand):Promise<void> {
  if(busy)return;
  busy=true;revision++;
  try{
    const response=await fetch('/api/roundtrip/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command),signal:AbortSignal.timeout(10000)});
    if(!response.ok){
      const payload:unknown=await response.json().catch(()=>null);
      const detail=typeof payload==='object'&&payload!==null&&'detail' in payload&&typeof payload.detail==='string'?payload.detail:'';
      throw new Error(detail||`操作失败（${response.status}）`);
    }
    render(await response.json() as RoundTripState);
  }catch(error){$('#error').textContent=error instanceof Error?error.message:'操作失败';$('#error').hidden=false;}
  finally{busy=false;}
}

$('#play').addEventListener('click',()=>{if(current)void send({action:current.paused?'play':'pause'});});
$('#step').addEventListener('click',()=>void send({action:'step'}));
$('#reset').addEventListener('click',()=>void send({action:'reset'}));
$('#focus').addEventListener('click',focus);
$<HTMLSelectElement>('#speed').addEventListener('change',()=>void send({action:'speed',speed:Number($<HTMLSelectElement>('#speed').value) as 1|4|16}));
$<HTMLSelectElement>('#write-mode').addEventListener('change',()=>void send({action:'write_mode',write_mode:$<HTMLSelectElement>('#write-mode').value as WriteMode}));
document.querySelectorAll<HTMLButtonElement>('[data-phase]').forEach(control=>control.addEventListener('click',()=>void send({action:'phase',phase:control.dataset.phase as 'memory'|'adaptive'|'autonomous'})));

async function poll():Promise<void> {
  const currentRevision=revision;
  try{
    const response=await fetch('/api/roundtrip/state',{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error(`服务返回 ${response.status}`);
    const next=await response.json() as RoundTripState;
    if(currentRevision===revision&&!busy)render(next);
  }catch{
    connected=false;$('#connection').textContent='连接中断';$('#connection').className='disconnected';
    $('#error').textContent='无法连接往返验收服务，正在重试。';$('#error').hidden=false;
  }finally{setTimeout(()=>void poll(),250);}
}
void poll();
