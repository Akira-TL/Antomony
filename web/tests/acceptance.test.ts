import {test} from 'node:test';
import assert from 'node:assert/strict';
import {counts,pointAt,visibleIndividualIds} from '../src/acceptance/model.ts';
import type {Ant,Frame,Header} from '../src/acceptance/model.ts';

const ant=(overrides:Partial<Ant>={}):Ant=>({observation:[],active:true,move:true,turn:0,position:[0,0],heading:0,
  carrying:false,exploration_left:160,reserve_left:160,picked_up:false,delivered:false,budget_return:false,
  exhausted:false,killed:false,injury:0,reward:0,writes:0,...overrides});
const frame=(tick:number,ants:Ant[],food_stock:number|null=null):Frame=>({tick,ants,food_stock,source_position:[4,0],source_active:true});

test('复活保留累计死亡而恢复存活，首次等待出巢不算耗尽',()=>{
  const trace=[
    frame(1,[ant({killed:true,exhausted:true,pending:true,cumulative_deaths:1,cumulative_terminations:1}),ant({exhausted:true,pending:true})],384),
    frame(2,[ant({respawned:true,revivals:1,cumulative_deaths:1,cumulative_terminations:1}),ant()],384),
    frame(3,[ant({killed:true,exhausted:true,pending:true,revivals:1,cumulative_deaths:2,cumulative_terminations:2}),ant({delivered:true})],383),
  ];
  const actual=counts(trace,[false,true]);
  assert.equal(actual[0].alive,1);assert.equal(actual[0].waiting,1);
  assert.equal(actual[1].deaths,1);assert.equal(actual[1].exhausted,1);assert.equal(actual[1].waiting,2);
  assert.equal(actual[2].deaths,1);assert.equal(actual[2].alive,2);assert.equal(actual[2].revivals,1);
  assert.equal(actual[3].deaths,2);assert.equal(actual[3].deliveries,1);
});

test('旧记录继续按终止状态计数，提前结束不伪造参数变化',()=>{
  const result=counts([frame(1,[ant({exhausted:true,killed:true}),ant({delivered:true})])]);
  assert.equal(result[1].deaths,1);assert.equal(result[1].exhausted,1);
  assert.equal(result[1].deliveries,1);assert.equal(result[1].revivals,0);
  assert.deepEqual(pointAt([{tick:0,values:[0]},{tick:32,values:[1]}],4096),{tick:32,values:[1]});
});

test('隐藏等待者后仍保留真实个体编号，绘制槽位连续',()=>{
  const header={initial_positions:Array.from({length:32},()=>[0,0]),initial_pending:Array.from({length:32},(_,i)=>i>=8)} as Header;
  assert.deepEqual(visibleIndividualIds(header,null),[0,1,2,3,4,5,6,7]);
  const state=frame(2,Array.from({length:32},(_,i)=>ant({pending:![3,8,31].includes(i)})),384);
  const ids=visibleIndividualIds(header,state);
  assert.deepEqual(ids,[3,8,31]);
  assert.equal(ids[2],31);assert.equal(ids.indexOf(31),2);assert.equal(ids.indexOf(0),-1);
});
