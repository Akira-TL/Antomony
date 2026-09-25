import {test} from 'node:test';
import assert from 'node:assert/strict';
import {AntGait} from '../src/render/gait.ts';

test('静止和暂停不会播放循环腿动画',()=>{
  const gait=new AntGait();const first=structuredClone(gait.update(0,0,0,.016));
  for(let i=0;i<100;i++)assert.deepEqual(gait.update(0,0,0,.016),first);
  assert.deepEqual(gait.update(0,0,0,0),first);
});
test('支撑足保持原来的世界落点，超过阈值后有序移动',()=>{
  const gait=new AntGait();const first=structuredClone(gait.update(0,0,0,.016));
  const small=gait.update(.03,0,0,.016);
  small.forEach((leg,i)=>{assert.equal(leg.x,first[i].x);assert.equal(leg.z,first[i].z);});
  let stepped=false;
  for(let n=1;n<200;n++){
    const feet=gait.update(n*.02,0,0,.016);const moving=feet.map((f,i)=>f.moving?i:-1).filter(i=>i>=0);
    stepped ||= moving.length>0;
    for(const a of moving)for(const b of moving)if(a!==b){assert.notEqual(Math.floor(a/3),Math.floor(b/3));assert.notEqual(a%3,b%3);}
    assert(feet.every(f=>Number.isFinite(f.x)&&Number.isFinite(f.y)));
  }
  assert(stepped);
});
