import {createElement,Play,Pause,StepForward,RotateCcw,Download,Focus,Lock,Unlock,Activity} from 'lucide';
import type {IconNode} from 'lucide';
import * as THREE from 'three';
import {ColonyScene} from '../scene';
import {NetworkView} from './network';
import type {TrainingState,TrainingCommand,Phase,Lesson,GroupId,WeightTrace} from './types';
import './style.css';

const $=<T extends HTMLElement=HTMLElement>(selector:string):T=>document.querySelector<T>(selector)!;
const icon=(node:IconNode)=>createElement(node,{width:18,height:18,'stroke-width':1.7}).outerHTML;
const button=(id:string,title:string,node:IconNode)=>`<button type="button" class="icon-button" id="${id}" title="${title}" aria-label="${title}">${icon(node)}</button>`;
const phases:Record<Phase,string>={motor:'基础动作',meta:'自修改预训练',autonomous:'自主运行'};
const lessons:Record<Lesson,string>={straight:'前方食物',turn:'左右转向',random:'随机食物'};

$('#app').innerHTML=`
  <header><div class="identity">${icon(Activity)}<div><span>MathHackson</span><h1>单蚁训练</h1></div></div>
    <div class="session-status"><span id="connection" role="status">连接中</span><span class="muted" id="session-id"></span></div>
  </header>
  <main>
    <div class="toolbar">
      <div class="segmented phases" role="group" aria-label="训练阶段">
        <button type="button" data-phase="motor" aria-pressed="true">基础动作</button><button type="button" data-phase="meta" aria-pressed="false">自修改预训练</button><button type="button" data-phase="autonomous" aria-pressed="false">自主运行</button>
      </div>
      <div class="run-controls">${button('play','开始训练',Play)}${button('step','单步',StepForward)}${button('reset','重置当前回合，保留参数',RotateCcw)}
        <label class="speed-label">速度<select id="speed" aria-label="运行速度"><option value="1">1×</option><option value="4">4×</option><option value="16">16×</option></select></label>
        ${button('export','下载当前参数与状态',Download)}
      </div>
    </div>
    <div id="error" role="alert" hidden></div>
    <div class="workspace">
      <section class="experiment">
        <div class="section-heading"><h2>单蚁场景</h2><span id="running" class="state-tag">已暂停</span></div>
        <div class="course-controls"><label>课程<select id="lesson"><option value="random" selected>随机食物</option><option value="straight">前方食物</option><option value="turn">左右转向</option></select></label>
          <label>每步转向上限<input id="max-turn" type="number" min="1" max="30" step="1" value="10" required aria-label="每步最大转向角度"><span>°</span></label>
        </div>
        <div class="scene-wrap"><div id="scene"></div><div class="scene-caption"><span>蚂蚁 01</span><span id="position">x 0.00 · y 0.00</span></div>
          <div class="camera-tools">${button('focus','居中观察蚂蚁与食物',Focus)}<label title="镜头随蚂蚁与食物移动"><input id="follow" type="checkbox" checked>跟随</label></div>
          <div class="scene-bottom"><span id="target-distance">食物距离</span><span>可见食物 · 信息素未启用</span></div>
        </div>
        <div class="telemetry"><div><span>前进</span><strong id="move">—</strong><small id="move-probability">—</small></div><div><span>本步转向</span><strong id="turn">0.00°</strong><small id="turn-value">0.0000</small></div>
          <div><span>自写入判断</span><strong id="write">尚未推理</strong><small id="write-probability">—</small></div><div><span>回合累计奖励</span><strong id="reward">0.000</strong><small id="step-reward">本步 0.000</small></div></div>
        <div class="progress-heading"><h2 id="episode">第 1 回合</h2><span id="progress-text">0 / 96 步</span></div><progress id="progress" value="0" max="96"></progress>
        <div class="chart-heading"><h2>回合奖励</h2><div class="update-counts"><span>外部更新 <b id="outer-count">0</b></span><span>自写入 <b id="self-count">0</b></span></div></div>
        <canvas id="reward-chart" width="1000" height="160" aria-label="已结束回合的真实奖励曲线"></canvas>
        <div class="history-heading"><h2>训练记录</h2><span class="muted" id="history-count">尚无完整回合</span></div>
        <div class="history-scroll"><table><thead><tr><th>回合</th><th>阶段 / 结束原因</th><th>奖励</th><th>写入 / 外部</th><th>参数</th></tr></thead><tbody id="history"></tbody></table></div>
      </section>
      <section class="network" aria-label="模型参数与冻结状态">
        <div class="section-heading"><h2>参数与冻结</h2><div class="parameter-actions"><span>39 × 16 · 624 参数</span>${button('freeze-all','冻结全部参数',Lock)}${button('unfreeze-all','解除手动冻结',Unlock)}</div></div>
        <div id="groups" class="groups"></div>
        <div class="weight-trace-heading"><h2>选中权重的变化</h2><span id="weight-trace-status" role="status">等待逐步记录</span></div>
        <canvas id="weight-trace" width="700" height="200" aria-label="选中权重的真实逐步轨迹"></canvas>
        <div class="trace-legend"><span><i class="self-mark"></i>模型自写入</span><span><i class="outer-mark"></i>回合末外部更新</span><span><i class="skip-mark"></i>没有变化</span></div>
        <ol id="weight-events" class="weight-events" aria-label="最近的权重变化"></ol>
        <div class="network-toolbar"><div class="segmented" role="group" aria-label="矩阵视图"><button type="button" data-view="weight" aria-pressed="true">当前权重</button><button type="button" data-view="self" aria-pressed="false">本步自写入 Δ</button><button type="button" data-view="outer" aria-pressed="false">最近外部更新 Δ</button></div><span class="legend"><i></i>负<i></i>正</span></div>
        <canvas id="matrix" tabindex="0" aria-label="完整参数矩阵"></canvas>
        <div class="parameter-picker"><span>参数定位</span><label>行<input id="parameter-row" type="number" min="0" max="38" step="1" value="0" required></label><label>列<input id="parameter-column" type="number" min="0" max="15" step="1" value="0" required></label></div>
        <div id="parameter-detail" class="parameter-detail"></div>
      </section>
    </div>
    <footer><span>开发训练器 · 效果尚未验证</span><span id="phase-note">外部梯度训练 · 自写入关闭</span></footer>
  </main>`;

const scene=new ColonyScene($('#scene'),true);
scene.showField=false;
const network=new NetworkView($('#matrix'),$('#parameter-detail'),$('#weight-trace'),$('#weight-trace-status'),$('#weight-events'));
let traceSelection='';
network.onSelect=(row,column)=>{
  if(document.activeElement!==$('#parameter-row'))$<HTMLInputElement>('#parameter-row').value=String(row);
  if(document.activeElement!==$('#parameter-column'))$<HTMLInputElement>('#parameter-column').value=String(column);
  const selection=`${row}:${column}`;
  if(selection!==traceSelection){traceSelection=selection;void loadTrace();}
};
const emptyField=btoa('\0'.repeat(96*64*2));
const pathMaterial=new THREE.LineBasicMaterial({color:0x63bccf,transparent:true,opacity:.65});
const trail=new THREE.Line(new THREE.BufferGeometry(),pathMaterial);scene.scene.add(trail);
let trailPoints:THREE.Vector3[]=[];
let state:TrainingState|null=null;
let busy=false,connected=false,revision=0,lastTick=-1,lastEpisode=-1,lastSession='';
let groupSignature='';
let historySignature='';
let commandError='';

async function loadTrace():Promise<void> {
  if(!state)return;
  const {row,column}=network,session=state.session;
  try{
    const response=await fetch(`/api/training/trace?row=${row}&column=${column}`,{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error(`轨迹接口返回 ${response.status}`);
    const trace=await response.json() as WeightTrace;
    if(trace.session===session&&state?.session===session&&network.row===row&&network.column===column)network.setTrace(trace);
  }catch{
    if(state?.session===session&&network.row===row&&network.column===column)$('#weight-trace-status').textContent='轨迹暂不可用';
  }
}

function focus():void {
  if(!state)return;
  const x=(state.x+state.target_x)/2,z=(state.y+state.target_y)/2;
  const distance=Math.max(8,state.distance*1.6);
  scene.controls.target.set(x,0,z);scene.camera.position.set(x,distance,distance*.7+z);scene.camera.lookAt(x,0,z);scene.controls.update();
}

function setControlsEnabled(enabled:boolean):void {
  document.querySelectorAll<HTMLButtonElement|HTMLSelectElement|HTMLInputElement>('.toolbar button,.toolbar select,.course-controls input,.course-controls select,[data-freeze],#freeze-all,#unfreeze-all').forEach(control=>control.disabled=!enabled);
  if(enabled&&state?.phase==='motor')document.querySelectorAll<HTMLInputElement>('[data-freeze]:not([data-freeze="action"])').forEach(control=>control.disabled=true);
}

async function send(command:TrainingCommand):Promise<void> {
  if(busy)return;
  busy=true;revision++;commandError='';setControlsEnabled(false);
  try {
    const response=await fetch('/api/training/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command),signal:AbortSignal.timeout(10000)});
    if(!response.ok)throw new Error(`操作失败（${response.status}）`);
    render(await response.json() as TrainingState);
  }catch(error){commandError=error instanceof Error?error.message:'操作失败';showError(commandError);}
  finally{busy=false;setControlsEnabled(connected);}
}

function showError(message:string):void {$('#error').textContent=message;$('#error').hidden=!message;}

function render(next:TrainingState):void {
  connected=true;state=next;
  $('#connection').textContent='已连接';$('#connection').className='connected';
  $('#session-id').textContent=next.session;
  $('#running').textContent=next.paused?'已暂停':'运行中';$('#running').classList.toggle('live',!next.paused);
  $('#play').innerHTML=icon(next.paused?Play:Pause);$('#play').title=next.paused?'开始训练':'暂停';$('#play').setAttribute('aria-label',$('#play').title);
  document.querySelectorAll<HTMLButtonElement>('[data-phase]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.phase===next.phase)));
  if(document.activeElement!==$('#lesson'))$<HTMLSelectElement>('#lesson').value=next.lesson;
  $<HTMLSelectElement>('#speed').value=String(next.speed);
  if(document.activeElement!==$('#max-turn'))$<HTMLInputElement>('#max-turn').value=String(next.max_turn);
  $('#position').textContent=`x ${next.x.toFixed(2)} · y ${next.y.toFixed(2)}`;
  $('#target-distance').textContent=`食物距离 ${next.distance.toFixed(2)}`;
  const inferred=next.steps>0;
  $('#move').textContent=inferred?(next.move?'前进':'停止'):'—';$('#move-probability').textContent=inferred?`概率 ${(next.move_probability*100).toFixed(1)}%`:'等待推理';
  $('#turn').textContent=inferred?`${(next.turn*next.max_turn).toFixed(2)}°`:'—';$('#turn-value').textContent=inferred?`输出 ${next.turn.toFixed(4)}`:'等待推理';
  $('#write').textContent=next.write_status;$('#write-probability').textContent=inferred?`写入概率 ${(next.write_probability*100).toFixed(1)}%`:'等待推理';
  $('#reward').textContent=next.total_reward.toFixed(3);$('#step-reward').textContent=`本步 ${next.reward.toFixed(3)}`;
  $('#episode').textContent=`第 ${next.episode} 回合`;$('#progress-text').textContent=`${next.steps} / ${next.horizon} 步`;
  $<HTMLProgressElement>('#progress').value=next.steps;$<HTMLProgressElement>('#progress').max=next.horizon;
  $('#self-count').textContent=String(next.self_updates);$('#outer-count').textContent=String(next.outer_updates);
  $('#phase-note').textContent=next.phase==='motor'?'外部梯度训练 · 自写入关闭':next.phase==='meta'?'外部梯度训练 + 模型决定自写入':'外部训练已停止 · 模型决定自写入';
  const signature=JSON.stringify(next.groups);
  if(signature!==groupSignature){
    $('#groups').innerHTML=next.groups.map(group=>`<label class="group-row ${group.frozen?'frozen':''}"><span>${icon(group.frozen?Lock:Unlock)}${group.label}</span><span class="group-range">W[${group.start}:${group.end}]</span><span class="freeze-reason">${group.reason}</span><input type="checkbox" data-freeze="${group.id}" aria-label="冻结${group.label}" ${group.frozen?'checked':''}></label>`).join('');
    groupSignature=signature;
  }
  network.update(next);
  const newHistory=JSON.stringify(next.history);
  if(newHistory!==historySignature){
    $('#history-count').textContent=next.history.length?`保留最近 ${next.history.length} 段`:'尚无完整回合';
    $('#history').innerHTML=next.history.slice(-8).reverse().map(record=>`<tr><td>${record.episode}</td><td>${phases[record.phase]}<small>${record.reason==='到达目标'?'触达食物':record.reason} · ${lessons[record.lesson]}</small></td><td class="${record.reward<0?'negative':'positive'}">${record.reward.toFixed(3)}</td><td>${record.self_updates} / ${record.outer_updates}</td><td><a class="icon-button" title="下载第 ${record.episode} 回合的逐步权重" aria-label="下载第 ${record.episode} 回合参数" href="/api/training/checkpoints/${record.episode}" download>${icon(Download)}</a></td></tr>`).join('');
    drawRewards(next);historySignature=newHistory;
  }
  if(next.tick!==lastTick||next.episode!==lastEpisode||next.session!==lastSession){
    const changedEpisode=next.episode!==lastEpisode||next.session!==lastSession;
    if(changedEpisode)trailPoints=[];
    trailPoints.push(new THREE.Vector3(next.x,.03,next.y));trailPoints=trailPoints.slice(-300);
    trail.geometry.dispose();trail.geometry=new THREE.BufferGeometry().setFromPoints(trailPoints);
    scene.update({mode:'neural',tick:next.tick,seconds:next.steps*.1,seed:next.episode,delivered:0,
      field_width:96,field_height:64,pheromones:emptyField,walls:[],foods:[{id:0,x:next.target_x,y:next.target_y,amount:8}],
      ants:[{id:0,x:next.x,y:next.y,heading:next.heading,carrying:false,frozen:next.groups.every(g=>g.frozen),rays:[],sense_x:next.x,sense_y:next.y,sense_heading:next.heading}]});
    if(lastSession!==next.session)focus();
    else if($<HTMLInputElement>('#follow').checked){
      const target=new THREE.Vector3((next.x+next.target_x)/2,0,(next.y+next.target_y)/2);
      const delta=target.clone().sub(scene.controls.target);scene.camera.position.add(delta);scene.controls.target.copy(target);
      if(changedEpisode)focus();
    }
    lastTick=next.tick;lastEpisode=next.episode;lastSession=next.session;
  }
  showError(next.error||commandError);if(!busy)setControlsEnabled(true);
  $('#app').dataset.tick=String(next.tick);$('#app').dataset.episode=String(next.episode);$('#app').dataset.paused=String(next.paused);
}

function drawRewards(next:TrainingState):void {
  const canvas=$<HTMLCanvasElement>('#reward-chart'),ctx=canvas.getContext('2d')!;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const records=next.history.filter(record=>record.reason==='到达目标'||record.reason==='回合结束').slice(-40);
  const rewards=records.map(record=>record.reward),min=Math.min(-1,...rewards),max=Math.max(1,...rewards);
  const y=(value:number)=>18+(max-value)/(max-min)*112;
  ctx.strokeStyle='#41464e';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(55,y(0));ctx.lineTo(980,y(0));ctx.stroke();
  ctx.fillStyle='#a4aab4';ctx.font='20px system-ui';ctx.fillText(max.toFixed(1),0,23);ctx.fillText(min.toFixed(1),0,134);
  if(!rewards.length){ctx.fillText('等待回合结束',70,80);return;}
  ctx.strokeStyle='#6cd9ba';ctx.lineWidth=3;ctx.beginPath();
  rewards.forEach((reward,i)=>{const x=65+i*900/Math.max(1,rewards.length-1);if(i===0)ctx.moveTo(x,y(reward));else ctx.lineTo(x,y(reward));});ctx.stroke();
  rewards.forEach((reward,i)=>{ctx.fillStyle=reward<0?'#f1a372':'#6cd9ba';ctx.beginPath();ctx.arc(65+i*900/Math.max(1,rewards.length-1),y(reward),4,0,Math.PI*2);ctx.fill();});
}

$('#play').addEventListener('click',()=>{if(state)void send({action:state.paused?'play':'pause'});});
$('#step').addEventListener('click',()=>void send({action:'step'}));
$('#reset').addEventListener('click',()=>void send({action:'reset'}));
$('#focus').addEventListener('click',focus);
$('#freeze-all').addEventListener('click',()=>void send({action:'freeze_all',frozen:true}));
$('#unfreeze-all').addEventListener('click',()=>void send({action:'freeze_all',frozen:false}));
$('#lesson').addEventListener('change',()=>void send({action:'lesson',lesson:$<HTMLSelectElement>('#lesson').value as Lesson}));
$('#speed').addEventListener('change',()=>void send({action:'speed',speed:Number($<HTMLSelectElement>('#speed').value) as 1|4|16}));
$('#max-turn').addEventListener('change',()=>{const input=$<HTMLInputElement>('#max-turn');if(input.reportValidity())void send({action:'turn',max_turn:Number(input.value)});});
$('#groups').addEventListener('change',event=>{const input=event.target as HTMLInputElement;void send({action:'freeze',group:input.dataset.freeze as GroupId,frozen:input.checked});});
document.querySelectorAll<HTMLButtonElement>('[data-phase]').forEach(button=>button.addEventListener('click',()=>void send({action:'phase',phase:button.dataset.phase as Phase})));
document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach(button=>button.addEventListener('click',()=>{
  network.setMode(button.dataset.view as 'weight'|'self'|'outer');
  document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach(other=>other.setAttribute('aria-pressed',String(other===button)));
}));
for(const id of ['#parameter-row','#parameter-column'])$(id).addEventListener('change',()=>{
  const row=$<HTMLInputElement>('#parameter-row'),column=$<HTMLInputElement>('#parameter-column');
  if(row.reportValidity()&&column.reportValidity())network.select(Number(row.value),Number(column.value));
});
$('#export').addEventListener('click',()=>{
  if(!state)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=`single-ant-${state.session}-step-${state.tick}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});

async function poll():Promise<void> {
  const requestRevision=revision;
  try{
    const response=await fetch('/api/training/state',{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error(`服务返回 ${response.status}`);
    const next=await response.json() as TrainingState;
    if(requestRevision===revision&&!busy){render(next);await loadTrace();}
  }catch{
    connected=false;$('#connection').textContent='连接中断';$('#connection').className='disconnected';
    showError('无法连接独立训练服务，正在重试。');setControlsEnabled(false);
  }finally{setTimeout(()=>void poll(),150);}
}
setControlsEnabled(false);
void poll();
