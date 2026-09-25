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

test('脚端从身后抬起并落在身体前方，落地后保持支撑',()=>{
  const gait=new AntGait();gait.update(0,0,0,.016);
  let x=0;let stepping:number[]=[];
  for(let n=1;n<=12;n++){
    x=n*.02;
    stepping=gait.update(x,0,0,.016).map((foot,i)=>foot.moving?i:-1).filter(i=>i>=0);
    if(stepping.length)break;
  }
  assert(stepping.length>0);
  const landed=gait.update(x,0,0,.2);
  for(const i of stepping){
    const neutral=x+(i%3-1)*.14;
    assert(landed[i].x-neutral>.18,`第 ${i} 条腿没有明显向前落足`);
    assert.equal(landed[i].moving,false);
  }
  const planted=gait.update(x+.08,0,0,.016);
  for(const i of stepping){
    assert.equal(planted[i].x,landed[i].x);
    assert.equal(planted[i].moving,false);
  }
});

test('身体继续前进时，摆动足仍落在前方而非追着身体走',()=>{
  const gait=new AntGait();gait.update(0,0,0,.016);
  let x=0;let stepping:number[]=[];
  for(let n=1;n<=20;n++){
    x+=.03;
    stepping=gait.update(x,0,0,.016).map((foot,i)=>foot.moving?i:-1).filter(i=>i>=0);
    if(stepping.length)break;
  }
  assert(stepping.length>0);
  let feet=gait.update(x,0,0,0);
  for(let n=0;n<8&&stepping.some(i=>feet[i].moving);n++){
    x+=.03;feet=gait.update(x,0,0,.016);
  }
  for(const i of stepping){
    const neutral=x+(i%3-1)*.14;
    assert(feet[i].x-neutral>.18,`第 ${i} 条腿落地时已被身体赶上`);
    assert.equal(feet[i].moving,false);
  }
});

test('直行时同侧三足不互相越位',()=>{
  const gait=new AntGait();
  for(let n=0;n<180;n++){
    const feet=gait.update(n*.025,0,0,.016);
    for(const base of [0,3]){
      assert(feet[base].x+.015<feet[base+1].x,`第 ${n} 帧后足越过中足`);
      assert(feet[base+1].x+.015<feet[base+2].x,`第 ${n} 帧中足越过前足`);
    }
  }
});

test('持续转向时同侧足端仍保持前后顺序',()=>{
  const gait=new AntGait();let x=0,z=0;
  for(let n=0;n<160;n++){
    const heading=n*.012;x+=Math.cos(heading)*.025;z+=Math.sin(heading)*.025;
    const feet=gait.update(x,z,heading,.016),c=Math.cos(heading),s=Math.sin(heading);
    for(const base of [0,3]){
      const along=[0,1,2].map(j=>feet[base+j].x*c+feet[base+j].z*s);
      assert(along[0]+.01<along[1]&&along[1]+.01<along[2],`第 ${n} 帧足端越位`);
    }
  }
});
