import {createElement,Activity,Play,Pause,StepForward,RotateCcw,Focus} from 'lucide';
import type {IconNode} from 'lucide';
import * as THREE from 'three';
import {ColonyScene} from '../scene';
import type {RecurrentState,RecurrentCommand,RecurrentParameterHistory,RecurrentPhase,RecurrentTask,WriteMode} from './types';
import './style.css';
import './recurrent-chart-style.css';

const $=<T extends HTMLElement=HTMLElement>(selector:string):T=>document.querySelector<T>(selector)!;
const icon=(node:IconNode)=>createElement(node,{width:18,height:18,'stroke-width':1.7}).outerHTML;
const button=(id:string,title:string,node:IconNode)=>`<button type="button" class="icon-button" id="${id}" title="${title}" aria-label="${title}">${icon(node)}</button>`;
const phases:Record<RecurrentPhase,string>={motor:'基础动作',memory:'循环记忆',adaptive:'条件写入',autonomous:'停止外部训练'};
const tasks:Record<RecurrentTask,string>={normal:'正常',shift:'转向扰动',sensor:'感知偏差',mixed:'混合任务'};

$('#app').innerHTML=`
  <header><div class="identity">${icon(Activity)}<div><span>MathHackson</span><h1>循环记忆试验</h1></div></div>
    <div class="session-status"><a href="/training.html">原训练页</a><span id="connection" role="status">连接中</span><span class="muted" id="session-id"></span></div>
  </header>
  <main>
    <div class="toolbar"><div class="segmented phases" role="group" aria-label="训练阶段">
      <button type="button" data-phase="motor" aria-pressed="true">基础动作</button>
      <button type="button" data-phase="memory" aria-pressed="false">循环记忆</button>
      <button type="button" data-phase="adaptive" aria-pressed="false">条件写入</button>
      <button type="button" data-phase="autonomous" aria-pressed="false">停止外部训练</button>
    </div><div class="run-controls">${button('play','开始训练',Play)}${button('step','单步',StepForward)}${button('reset','重置回合，保留参数',RotateCcw)}
      <label class="speed-label">速度<select id="speed" aria-label="运行速度"><option value="1">1×</option><option value="4">4×</option><option value="16">16×</option></select></label>
    </div></div>
    <div id="error" role="alert" hidden></div>
    <div class="workspace">
      <section class="experiment"><div class="section-heading"><h2>单蚁场景</h2><span id="running" class="state-tag">已暂停</span></div>
        <div class="course-controls"><label>任务<select id="task"><option value="normal">正常</option><option value="shift">转向扰动</option><option value="sensor">感知偏差</option><option value="mixed">混合任务</option></select></label>
          <label>写入对照<select id="write-mode" aria-label="运行时写入方式"><option value="off">关闭</option><option value="learned">模型判断</option><option value="always">始终写入</option></select></label>
          <span class="muted" id="perturbation">本回合无扰动</span></div>
        <div class="scene-wrap"><div id="scene"></div><div class="scene-caption"><span>蚂蚁 01</span><span id="position">x 0.00 · y 0.00</span></div>
          <div class="camera-tools">${button('focus','居中观察蚂蚁与食物',Focus)}<label><input id="follow" type="checkbox" checked>跟随</label></div>
          <div class="scene-bottom"><span id="distance">食物距离</span><span>单只蚂蚁 · 局部反馈</span></div>
        </div>
        <div class="telemetry"><div><span>前进</span><strong id="move">—</strong><small id="move-probability">—</small></div>
          <div><span>本步转向</span><strong id="turn">—</strong><small id="turn-value">—</small></div>
          <div><span>回合奖励</span><strong id="reward">0.000</strong><small id="step-reward">本步 0.000</small></div>
          <div><span>回合内写入</span><strong id="write">尚未推理</strong><small id="write-count">累计 0</small></div>
          <div><span>外部更新</span><strong id="outer-count">0</strong><small id="phase-note">基础动作训练</small></div></div>
        <div class="progress-heading"><h2 id="episode">第 1 回合</h2><span id="progress-text">0 / 96 步</span></div><progress id="progress" value="0" max="96"></progress>
        <div class="history-heading"><h2>回合记录</h2><span class="muted" id="history-count">尚无完整回合</span></div>
        <div class="history-scroll"><table><thead><tr><th>回合</th><th>任务</th><th>结果</th><th>步数</th><th>写入</th><th>奖励</th></tr></thead><tbody id="history"></tbody></table></div>
      </section>
      <section class="network" aria-label="真实循环状态与参数"><div class="section-heading"><h2>隐藏状态</h2><span class="muted" id="hidden-width">8 维</span></div>
        <div id="hidden" class="hidden-values"></div>
        <div class="section-heading"><h2>稀疏记忆</h2><span class="muted">此前 1 / 8 / 12 / 16 步</span></div>
        <div id="memory-taps" class="memory-taps"></div>
        <div class="section-heading"><h2>回合内动作修正</h2><span class="muted">3 个受限参数</span></div>
        <div id="fast-values" class="fast-values"></div>
        <div class="section-heading"><h2>参数与冻结</h2><span class="muted" id="parameter-count">—</span></div>
        <div id="parameters" class="recurrent-parameters"></div>
      </section>
    </div><div id="chart-tooltip" class="chart-tooltip" role="status" hidden></div><footer><span>独立训练会话 · 工程试验</span><span>隐藏状态与运行时写入分别记录；效果尚未验证</span></footer>
  </main>`;

const scene=new ColonyScene($('#scene'),true);
scene.showField=false;
const emptyField=btoa('\0'.repeat(96*64*2));
const trail=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x63bccf,transparent:true,opacity:.65}));
scene.scene.add(trail);
let state:RecurrentState|null=null;
let busy=false,connected=false,revision=0,lastTick=-1,lastEpisode=-1,lastSession='';
let trailPoints:THREE.Vector3[]=[];
let historySignature='';
let parameterHistory:RecurrentParameterHistory|null=null;
let parameterMarkupKey='',requestedHistoryKey='',lastStepChartKey='';
const fastLabels=['前进偏置','转向增益','转向偏置'];

function sparkline(values:number[],frozen=false):string {
  const samples=values.length?values:[0];
  const minimum=Math.min(0,...samples),maximum=Math.max(0,...samples);
  const span=Math.max(maximum-minimum,.02),floor=(minimum+maximum-span)/2;
  const points=samples.map((value,index)=>`${samples.length===1?50:2+96*index/(samples.length-1)},${34-32*(value-floor)/span}`).join(' ');
  const zero=34-32*(0-floor)/span;
  return `<svg class="sparkline${frozen?' is-frozen':''}" viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true"><line class="zero-line" x1="0" y1="${zero}" x2="100" y2="${zero}"/><polyline points="${points}"/><line class="hover-guide" x1="-1" y1="0" x2="-1" y2="36"/></svg>`;
}

function drawParameterCharts():void {
  if(!state)return;
  document.querySelectorAll<HTMLElement>('.parameter-cell').forEach(cell=>{
    const group=Number(cell.dataset.group),row=Number(cell.dataset.row),column=Number(cell.dataset.column);
    const values=parameterHistory?.session===state!.session
      ?parameterHistory.samples.map(sample=>sample.values[group][row][column])
      :[state!.groups[group].values[row][column]];
    cell.querySelector('.sparkline')?.remove();
    cell.insertAdjacentHTML('beforeend',sparkline(values,cell.classList.contains('frozen')));
  });
}

function renderParameterMarkup(next:RecurrentState):void {
  const key=`${next.session}:${next.phase}`;
  if(key===parameterMarkupKey)return;
  $('#parameters').innerHTML=next.groups.map((group,groupIndex)=>{
    const total=group.values.flat().length,trainable=group.trainable.flat().filter(Boolean).length;
    const cells=group.values.flatMap((row,rowIndex)=>row.map((_,columnIndex)=>
      `<div class="parameter-cell ${group.trainable[rowIndex][columnIndex]?'':'frozen'}" tabindex="0" data-kind="parameter" data-group="${groupIndex}" data-row="${rowIndex}" data-column="${columnIndex}" aria-label="${group.label} ${rowIndex},${columnIndex}"><span>${rowIndex}:${columnIndex}</span></div>`)).join('');
    return `<details class="parameter-group" ${groupIndex===2||groupIndex===1?'':'open'}><summary class="parameter-group-heading"><strong>${group.label}</strong><span>${trainable}/${total} 可训练</span></summary><div class="parameter-grid">${cells}</div></details>`;
  }).join('');
  parameterMarkupKey=key;
  drawParameterCharts();
}

async function loadParameterHistory(key:string):Promise<void> {
  try{
    const response=await fetch('/api/recurrent/parameter-history',{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error(`趋势记录返回 ${response.status}`);
    const result=await response.json() as RecurrentParameterHistory;
    if(key!==requestedHistoryKey||result.session!==state?.session)return;
    parameterHistory=result;
    drawParameterCharts();
  }catch{
    if(key===requestedHistoryKey)setTimeout(()=>{requestedHistoryKey='';},2000);
  }
}

function renderStepCharts(next:RecurrentState):void {
  $('#hidden').innerHTML=next.hidden.map((_,index)=>`<div class="trace-cell" tabindex="0" data-kind="hidden" data-index="${index}" aria-label="隐藏状态 h${index} 的趋势"><span>h${index}</span>${sparkline(next.hidden_trace.map(frame=>frame[index]))}</div>`).join('');
  $('#fast-values').innerHTML=next.fast.map((_,index)=>`<div class="trace-cell" tabindex="0" data-kind="fast" data-index="${index}" aria-label="${fastLabels[index]}的趋势"><span>${fastLabels[index]}</span>${sparkline(next.fast_trace.map(frame=>frame[index]))}</div>`).join('');
  $('#memory-taps').innerHTML=next.memory_lags.map((lag,index)=>`<div class="memory-tap" tabindex="0" data-kind="tap" data-index="${index}" aria-label="此前 ${lag} 步的隐藏状态"><span>t-${lag}</span><small>${next.memory_ready[index]?'已接入':'待积累'}</small></div>`).join('');
}

function showTooltip(target:HTMLElement,clientX:number,clientY:number):void {
  if(!state)return;
  const kind=target.dataset.kind,index=Number(target.dataset.index);
  const tooltip=$('#chart-tooltip');
  if(kind==='tap'){
    tooltip.textContent=`t-${state.memory_lags[index]} · ${state.memory_ready[index]?'有效':'填零'} · [${state.memory_taps[index].map(value=>value.toFixed(4)).join(', ')}]`;
  }else{
    const group=Number(target.dataset.group),row=Number(target.dataset.row),column=Number(target.dataset.column);
    const samples=kind==='parameter'
      ?(parameterHistory?.session===state.session?parameterHistory.samples.map(sample=>sample.values[group][row][column]):[state.groups[group].values[row][column]])
      :kind==='hidden'?state.hidden_trace.map(frame=>frame[index]):state.fast_trace.map(frame=>frame[index]);
    const values=samples.length?samples:[kind==='hidden'?state.hidden[index]:state.fast[index]];
    const rect=target.querySelector('svg')?.getBoundingClientRect()??target.getBoundingClientRect();
    const sample=Math.max(0,Math.min(values.length-1,Math.round((clientX-rect.left)/rect.width*(values.length-1))));
    const label=kind==='parameter'?`${state.groups[group].label} [${row},${column}]`:
      kind==='hidden'?`h${index}`:fastLabels[index];
    const time=kind==='parameter'?`第 ${parameterHistory?.samples[sample]?.episode??state.episode} 回合`:
      `第 ${Math.max(1,state.steps-values.length+sample+1)} 步`;
    const delta=kind==='parameter'?` · 末次变化 ${state.groups[group].changes[row][column].toFixed(6)}`:'';
    tooltip.textContent=`${label} · ${time} · ${values[sample].toFixed(6)}${delta}`;
    const guide=target.querySelector<SVGLineElement>('.hover-guide');
    if(guide){const x=values.length===1?50:2+96*sample/(values.length-1);guide.setAttribute('x1',String(x));guide.setAttribute('x2',String(x));}
  }
  tooltip.hidden=false;
  tooltip.style.left=`${Math.max(8,Math.min(clientX+12,innerWidth-tooltip.offsetWidth-8))}px`;
  tooltip.style.top=`${Math.max(8,Math.min(clientY+12,innerHeight-tooltip.offsetHeight-8))}px`;
}

for(const selector of ['#parameters','#hidden','#fast-values','#memory-taps']){
  const container=$(selector);
  container.addEventListener('pointermove',event=>{
    const target=(event.target as Element).closest<HTMLElement>('[data-kind]');
    if(target)showTooltip(target,event.clientX,event.clientY);
  });
  container.addEventListener('pointerleave',()=>{$('#chart-tooltip').hidden=true;});
  container.addEventListener('focusin',event=>{
    const target=(event.target as Element).closest<HTMLElement>('[data-kind]');
    if(target)showTooltip(target,target.getBoundingClientRect().left,target.getBoundingClientRect().bottom);
  });
  container.addEventListener('focusout',()=>{$('#chart-tooltip').hidden=true;});
}

function focus():void {
  if(!state)return;
  const x=(state.x+state.target_x)/2,z=(state.y+state.target_y)/2;
  const distance=Math.max(8,state.distance*1.6);
  scene.controls.target.set(x,0,z);scene.camera.position.set(x,distance,distance*.7+z);scene.camera.lookAt(x,0,z);scene.controls.update();
}

function render(next:RecurrentState):void {
  const legacyService=!Array.isArray(next.hidden_trace);
  if(legacyService){
    next.hidden_trace=[next.hidden];next.fast_trace=[next.fast];
    next.memory_lags=[1,8,12,16];
    next.memory_taps=next.memory_lags.map(()=>Array(next.hidden.length).fill(0) as number[]);
    next.memory_ready=next.memory_lags.map(()=>false);
  }
  state=next;connected=true;
  $('#connection').textContent='已连接';$('#connection').className='connected';
  $('#session-id').textContent=next.motor_source_episode
    ?`${next.session} · 接续第 ${next.motor_source_episode} 回合动作参数`:next.session;
  $('#running').textContent=next.paused?'已暂停':'运行中';$('#running').classList.toggle('live',!next.paused);
  $('#play').innerHTML=icon(next.paused?Play:Pause);$('#play').title=next.paused?'开始训练':'暂停';$('#play').setAttribute('aria-label',$('#play').title);
  document.querySelectorAll<HTMLButtonElement>('[data-phase]').forEach(control=>control.setAttribute('aria-pressed',String(control.dataset.phase===next.phase)));
  $<HTMLSelectElement>('#task').value=next.task;$<HTMLSelectElement>('#speed').value=String(next.speed);
  $<HTMLSelectElement>('#write-mode').value=next.write_mode;
  const kind=next.perturbation_kind==='sensor'?'感知偏差':'转向扰动';
  const unit=next.perturbation_kind==='sensor'?' 弧度':'';
  $('#perturbation').textContent=next.perturbation===0?'本回合无扰动':next.active_perturbation===0?`${kind}尚未生效`:`${kind} ${next.active_perturbation>0?'+':''}${next.active_perturbation.toFixed(2)}${unit}`;
  $('#position').textContent=`x ${next.x.toFixed(2)} · y ${next.y.toFixed(2)}`;
  $('#distance').textContent=`食物距离 ${next.distance.toFixed(2)}`;
  const inferred=next.steps>0;
  $('#move').textContent=inferred?(next.move?'前进':'停止'):'—';
  $('#move-probability').textContent=inferred?`概率 ${(next.move_probability*100).toFixed(1)}%`:'等待推理';
  $('#turn').textContent=inferred?`${(next.turn*next.max_turn).toFixed(2)}°`:'—';
  $('#turn-value').textContent=inferred?`输出 ${next.turn.toFixed(4)}`:'等待推理';
  $('#reward').textContent=next.total_reward.toFixed(3);$('#step-reward').textContent=`本步 ${next.reward.toFixed(3)}`;
  $('#write').textContent=next.write_status;
  $('#write-count').textContent=`累计 ${next.self_updates} · 判断概率 ${(next.write_probability*100).toFixed(1)}%`;
  $('#outer-count').textContent=String(next.outer_updates);
  $('#hidden-width').textContent=`${next.hidden.length} 维`;
  $('#phase-note').textContent=next.phase==='motor'?'仅训练四个动作连接':next.phase==='memory'?'动作锁定 · 训练循环记忆':next.phase==='adaptive'?'动作锁定 · 训练写入判断':'外部训练停止';
  $('#episode').textContent=`第 ${next.episode} 回合`;
  $('#progress-text').textContent=`${next.steps} / ${next.horizon} 步`;
  $<HTMLProgressElement>('#progress').value=next.steps;
  const stepChartKey=`${next.session}:${next.tick}:${next.episode}:${next.phase}:${next.task}:${next.write_mode}`;
  if(stepChartKey!==lastStepChartKey){renderStepCharts(next);lastStepChartKey=stepChartKey;}
  const count=next.groups.reduce((sum,group)=>sum+group.trainable.flat().filter(Boolean).length,0);
  $('#parameter-count').textContent=`当前可训练 ${count}${legacyService?' · 旧服务':''}`;
  renderParameterMarkup(next);
  const traceKey=`${next.session}:${next.episode}`;
  if(!legacyService&&traceKey!==requestedHistoryKey){
    requestedHistoryKey=traceKey;
    void loadParameterHistory(traceKey);
  }
  const history=JSON.stringify(next.history);
  if(history!==historySignature){
    $('#history-count').textContent=next.history.length?`最近 ${next.history.length} 回合`:'尚无完整回合';
    $('#history').innerHTML=next.history.slice(-12).reverse().map(record=>`<tr><td>${record.episode}</td><td>${tasks[record.task]}<small>${phases[record.phase]} · ${record.perturbation_kind==='sensor'?'感知':record.perturbation_kind==='turn'?'转向':'无'} ${record.perturbation.toFixed(2)}</small></td><td>${record.reached?'到达':'未到达'}</td><td>${record.steps}</td><td>${record.writes}</td><td class="${record.reward<0?'negative':'positive'}">${record.reward.toFixed(3)}</td></tr>`).join('');
    historySignature=history;
  }
  if(next.tick!==lastTick||next.episode!==lastEpisode||next.session!==lastSession){
    const changedEpisode=next.episode!==lastEpisode||next.session!==lastSession;
    if(changedEpisode)trailPoints=[];
    trailPoints.push(new THREE.Vector3(next.x,.03,next.y));trailPoints=trailPoints.slice(-300);
    trail.geometry.dispose();trail.geometry=new THREE.BufferGeometry().setFromPoints(trailPoints);
    scene.update({mode:'neural',tick:next.tick,seconds:next.steps*.1,seed:next.episode,delivered:0,
      field_width:96,field_height:64,pheromones:emptyField,walls:[],foods:[{id:0,x:next.target_x,y:next.target_y,amount:8}],
      ants:[{id:0,x:next.x,y:next.y,heading:next.heading,carrying:false,frozen:next.phase==='autonomous',rays:[],sense_x:next.x,sense_y:next.y,sense_heading:next.heading}]});
    if(lastSession!==next.session)focus();
    else if($<HTMLInputElement>('#follow').checked){
      const target=new THREE.Vector3((next.x+next.target_x)/2,0,(next.y+next.target_y)/2);
      const delta=target.clone().sub(scene.controls.target);scene.camera.position.add(delta);scene.controls.target.copy(target);
      if(changedEpisode)focus();
    }
    lastTick=next.tick;lastEpisode=next.episode;lastSession=next.session;
  }
  const statusError=next.error||(legacyService?'当前训练服务仍在使用旧版模型；保留训练现场，趋势记录须在新版服务查看。':'');
  $('#error').textContent=statusError;$('#error').hidden=!statusError;
  document.querySelectorAll<HTMLButtonElement|HTMLSelectElement>('.toolbar button,.toolbar select,.course-controls select').forEach(control=>control.disabled=busy||!connected);
  $<HTMLSelectElement>('#write-mode').disabled=busy||!connected||next.phase!=='autonomous';
  document.querySelector<HTMLButtonElement>('[data-phase="motor"]')!.disabled=busy||!connected||next.phase!=='motor';
  $('#app').dataset.tick=String(next.tick);$('#app').dataset.episode=String(next.episode);
}

async function send(command:RecurrentCommand):Promise<void> {
  if(busy)return;
  busy=true;revision++;
  try{
    const response=await fetch('/api/recurrent/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command),signal:AbortSignal.timeout(10000)});
    if(!response.ok){
      const payload:unknown=await response.json().catch(()=>null);
      const detail=typeof payload==='object'&&payload!==null&&'detail' in payload&&typeof payload.detail==='string'?payload.detail:'';
      throw new Error(detail||`操作失败（${response.status}）`);
    }
    render(await response.json() as RecurrentState);
  }catch(error){$('#error').textContent=error instanceof Error?error.message:'操作失败';$('#error').hidden=false;}
  finally{busy=false;}
}

$('#play').addEventListener('click',()=>{if(state)void send({action:state.paused?'play':'pause'});});
$('#step').addEventListener('click',()=>void send({action:'step'}));
$('#reset').addEventListener('click',()=>void send({action:'reset'}));
$('#focus').addEventListener('click',focus);
$('#task').addEventListener('change',()=>void send({action:'task',task:$<HTMLSelectElement>('#task').value as RecurrentTask}));
$('#write-mode').addEventListener('change',()=>void send({action:'write_mode',write_mode:$<HTMLSelectElement>('#write-mode').value as WriteMode}));
$('#speed').addEventListener('change',()=>void send({action:'speed',speed:Number($<HTMLSelectElement>('#speed').value) as 1|4|16}));
document.querySelectorAll<HTMLButtonElement>('[data-phase]').forEach(control=>control.addEventListener('click',()=>void send({action:'phase',phase:control.dataset.phase as RecurrentPhase})));

async function poll():Promise<void> {
  const currentRevision=revision;
  try{
    const response=await fetch('/api/recurrent/state',{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error(`服务返回 ${response.status}`);
    const next=await response.json() as RecurrentState;
    if(currentRevision===revision&&!busy)render(next);
  }catch{
    connected=false;$('#connection').textContent='连接中断';$('#connection').className='disconnected';
    $('#error').textContent='无法连接循环记忆试验服务，正在重试。';$('#error').hidden=false;
  }finally{setTimeout(()=>void poll(),200);}
}
void poll();
