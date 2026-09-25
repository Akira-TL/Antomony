import {test} from 'node:test';
import assert from 'node:assert/strict';
import {comparisonText,contactRate,ComparisonHistory} from '../src/ui/comparison.ts';
import type {ComparisonFrame} from '../src/types.ts';

test('对比保留失败结果，零分母不产生无穷百分比',()=>{
  assert.equal(comparisonText(8,10).title,'神经组少搬回 2 块');
  assert.match(comparisonText(8,10).detail,/-20\.0%/);
  assert.equal(comparisonText(12,10).title,'神经组多搬回 2 块');
  assert.match(comparisonText(5,0).detail,/不计算/);
  assert.equal(comparisonText(0,0).title,'等待第一批搬运');
  assert.equal(contactRate({contacts:15,samples:100}),15);
});

test('暂停不会制造时间点，重开清除上次曲线，两条曲线同一纵轴',()=>{
  const history=new ComparisonHistory();
  const frame={tick:1,seconds:.1,delivered:1,reference:{delivered:2}} as ComparisonFrame;
  assert.equal(history.observe(frame,1),true);
  assert.equal(history.observe(frame,1),false);
  assert.equal(history.path('neural'),'M4.0,36.0');
  assert.equal(history.path('rules'),'M4.0,6.0');
  const reset={tick:0,seconds:0,delivered:0,reference:{delivered:0}} as ComparisonFrame;
  history.observe(reset,2);
  assert.equal(history.path('neural'),'M4.0,66.0');
});
