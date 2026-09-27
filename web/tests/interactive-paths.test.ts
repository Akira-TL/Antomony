import {test} from 'node:test';
import assert from 'node:assert/strict';
import {api,pageUrl} from '../src/interactive/types.ts';
import config from '../vite.config.ts';

const pages=[
  ['http://localhost:8775/interactive.html','http://localhost:8775/'],
  ['http://localhost:8775/','http://localhost:8775/'],
  ['https://babelbeast.com/Antonomy/interactive.html','https://babelbeast.com/Antonomy/'],
  ['https://babelbeast.com/Antonomy/','https://babelbeast.com/Antonomy/'],
  ['https://babelbeast.com/Antonomy/interactive.html?view=1#parameters','https://babelbeast.com/Antonomy/'],
];

for(const [page,base] of pages){
  test(`接口和导出跟随页面目录：${page}`,()=>{
    assert.equal(pageUrl('/api/state',page),`${base}api/state`);
    assert.equal(pageUrl('/api/parameters?group=adaptive&individual=3',page),`${base}api/parameters?group=adaptive&individual=3`);
    assert.equal(pageUrl('/api/download/record%20one.zip',page),`${base}api/download/record%20one.zip`);
    assert.equal(pageUrl('api/download/run.zip',page),`${base}api/download/run.zip`);
  });
}

test('构建使用相对资源地址以复用同一产物',()=>{
  assert.equal(config.base,'./');
});

test('不将非预期外部地址或目录穿越当作接口资源',()=>{
  for(const path of ['https://example.com/api/file','//example.com/api/file','javascript:alert(1)',
    '/api/../../other','api/%2e%2e/other']){
    assert.throws(()=>pageUrl(path,'https://babelbeast.com/Antonomy/interactive.html'),/接口资源/);
  }
});

for(const [page,base] of [pages[0],pages[2]]){
  test(`请求保留方法、正文和错误处理：${page}`,async context=>{
    const original=Object.getOwnPropertyDescriptor(globalThis,'window');
    Object.defineProperty(globalThis,'window',{configurable:true,value:{location:{href:page}}});
    context.after(()=>{
      if(original)Object.defineProperty(globalThis,'window',original);
      else Reflect.deleteProperty(globalThis,'window');
    });
    const requests:{url:string;options:RequestInit|undefined}[]=[];
    let failed=false;
    context.mock.method(globalThis,'fetch',async (input:RequestInfo|URL,options?:RequestInit)=>{
      requests.push({url:String(input),options});
      return failed?Response.json({detail:'本轮记录不存在'},{status:404}):Response.json({url:'/api/download/run.zip'});
    });
    await api('parameters?group=rules&individual=2');
    assert.equal(requests[0].url,`${base}api/parameters?group=rules&individual=2`);
    assert.equal(requests[0].options?.method,'GET');
    assert.equal(requests[0].options?.body,undefined);
    assert.ok(requests[0].options?.signal instanceof AbortSignal);
    const result=await api<{url:string}>('export',{});
    assert.equal(requests[1].url,`${base}api/export`);
    assert.equal(requests[1].options?.method,'POST');
    assert.equal(requests[1].options?.body,'{}');
    assert.deepEqual(requests[1].options?.headers,{'Content-Type':'application/json'});
    assert.equal(pageUrl(result.url),`${base}api/download/run.zip`);
    failed=true;
    await assert.rejects(api('replay?tick=3'),/本轮记录不存在/);
  });
}
