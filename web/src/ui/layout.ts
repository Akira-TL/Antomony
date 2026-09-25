/** 演示只保留操作、行为结果和所选个体的真实计算；技术边界放进说明。 */
export const layout = `
<header class="topbar">
  <a class="brand" href="/"><span class="brand-mark">M</span><span>MathHackson<small>独立神经蚁群</small></span></a>
  <div class="view-switch"><button id="compare" class="primary" title="从同一种子重新开始两个独立世界">开启同条件对比</button><button id="fit-view" class="quiet">全景</button><button id="fullscreen" class="quiet">全屏</button></div>
  <div class="connection"><i id="lamp"></i><span id="connection">连接本地仿真…</span><button id="about" class="quiet">说明</button></div>
</header>
<main class="workspace">
<section class="arena-panel">
  <div class="arena-heading"><div><span class="eyebrow">你改变环境，它们各自行动</span><h1 id="heading">一条路，如何被发现。</h1></div><div class="session"><span id="clock">00:00.0</span><small>仿真时间</small></div></div>
  <div id="arenas" class="arenas">
    <section class="world-pane neural-pane"><div class="pane-title"><strong>独立神经模型</strong><span id="neural-state">32 只 · 在线学习</span></div>
      <div class="canvas-wrap"><div id="viewport" class="viewport"></div><div id="toast" class="toast" role="status"></div><div id="hint" class="interaction-hint">拖动旋转 · 滚轮缩放 · 点击观察个体</div></div>
      <div class="pane-metrics"><span>搬回 <b id="delivered">0</b> 块</span><span title="累计发生接触的个体步数 / 累计个体步数 × 100">接触 <b id="contact-rate">0</b> / 百步</span><span title="当前连续低位移超过 4 步的个体数">卡住 <b id="stalled">0</b> 只</span></div>
    </section>
    <section id="reference-pane" class="world-pane rules-pane" hidden><div class="pane-title"><strong>普通规则 · 无神经网络</strong><span>同坐标同步干预</span></div>
      <div class="canvas-wrap"><div id="viewport-reference" class="viewport"></div><div class="reference-note">两边的食物、信息素和个体相互隔离</div></div>
      <div class="pane-metrics"><span>搬回 <b id="reference-delivered">0</b> 块</span><span title="累计发生接触的个体步数 / 累计个体步数 × 100">接触 <b id="reference-contacts">0</b> / 百步</span><span title="当前连续低位移超过 4 步的个体数">卡住 <b id="reference-stalled">0</b> 只</span></div>
    </section>
  </div>
  <div class="toolbar"><div class="tools" role="group" aria-label="场景工具">
    <button class="tool active" data-tool="inspect">⌖ 观察</button><button class="tool" data-tool="wall">▥ 放墙</button><button class="tool" data-tool="erase">⌫ 拆墙</button><button class="tool" data-tool="food">◈ 资源</button><button class="tool" data-tool="scent">∴ 信息素</button>
  </div><div class="play-controls"><button id="pause" class="round" title="暂停 / 继续">Ⅱ</button><button id="step" class="round" title="暂停后单步">▹</button><button id="speed" class="quiet">1×</button><button id="reset" class="quiet">新种子 ↻</button></div></div>
  <div class="arena-foot"><div class="legend"><span><i class="home-dot"></i>回巢</span><span><i class="food-dot"></i>食物</span><label><input type="checkbox" id="field-view" checked>信息素</label></div><span id="edit-scope">点击左侧场地编辑</span></div>
  <section id="comparison-results" class="comparison-results" hidden><div><strong id="difference">等待搬运结果</strong><small id="difference-detail">本次运行的实际差值，不预设胜负</small></div><svg id="comparison-chart" viewBox="0 0 500 70" preserveAspectRatio="none" role="img" aria-label="两组累计搬回资源随仿真时间的变化"><path id="rule-line"/><path id="neural-line"/></svg><div class="chart-legend"><span>绿：神经</span><span>橙：规则</span></div></section>
</section>
<aside class="sidebar">
  <section class="card control-card"><div class="section-title"><h2>改变环境</h2><span id="population" class="small-tag">32 只</span></div>
    <div id="wall-settings" class="wall-settings" hidden><label>墙长 <output id="wall-length-value">4.0</output><input id="wall-length" type="range" min="1" max="8" step=".2" value="4" aria-label="墙体长度"></label><div class="control-row"><span>角度 <output id="wall-angle">0°</output></span><button id="wall-rotate" class="quiet">转 90° / R</button></div><small>滚轮每格转 5°；放在蚂蚁上会就近移开。</small></div>
    <div class="control-row"><span>中央区域横向外力</span><output id="wind-value">0.0</output></div><input id="wind" class="slider" type="range" min="-1.4" max="1.4" step=".2" value="0" aria-label="中央横向外力"><div class="range-caption"><span>反向 ←</span><span>只改变物理，不通知模型</span><span>→ 正向</span></div>
    <div class="dual"><button id="clear-scent">清空信息素</button><button id="freeze-all">暂停学习</button></div>
    <details class="advanced"><summary>复现与导出</summary><label class="seed-row">种子<input id="seed" type="number" value="42" min="0" max="2147483646"><button id="apply-seed">重开</button></label><button id="export" class="quiet">导出当前双侧快照</button><p>开启对比会按当前种子重置场景；两组从同一起点重新开始。单个快照不等于完整回放。</p></details>
  </section>
  <section class="card inspector"><div class="section-title"><h2>看一只蚂蚁的计算</h2><select id="ant-select" aria-label="选择神经蚂蚁"></select></div>
    <div class="ant-id"><span class="ant-icon">✳</span><div><strong id="selected-label">个体 00</strong><small id="ant-state">探索中</small></div><button id="freeze-one" class="quiet">暂停学习</button></div>
    <div class="network"><canvas id="network" width="720" height="290" title="实际网络节点摘录；线宽表示权重绝对值，亮度表示输入乘权重，流光对应新前向计算"></canvas></div>
    <div class="network-key">线宽：权重　亮度：信号　闪光：更新</div>
    <dl class="details"><div><dt>已学习</dt><dd><span id="ant-updates">0</span> 次</dd></div><div><dt>更新前预测误差</dt><dd id="ant-error">—</dd></div></dl>
    <div class="micro-note">显示部分真实连线；没有合并个体的参数。</div>
  </section>
  <section class="card event-card"><div class="section-title"><h2>刚刚发生</h2><span id="trail-count" class="small-tag">循迹 0 只</span></div><div id="events" class="events"></div></section>
</aside></main>
<footer><span id="runtime">本地运行 · 连接中</span><span>局部传感输入 · 规则约束与神经学习分别展示</span></footer>
<dialog id="explanation"><button id="close-about" class="dialog-close">×</button><span class="eyebrow">关于这次现场对比</span><h2>同一个世界条件，两种行动方式。</h2>
<p>左侧每只蚂蚁有独立的 8 → 24 → 16 → 3 网络，分别预热，在线更新最后一层。网络预测局部运动与接触，再参与候选行动评分。</p>
<p>右侧不创建神经网络，依靠相同的局部传感、信息素、探索逻辑和名义运动规则。它不是随机乱走，也不是把一个训练好的网络冻结。</p>
<p>开启对比会重置为同一种子与初始条件。两组同步推进，环境编辑同时生效；但各自搬运食物、留下信息素，不互相借用结果。左侧暂停学习只是额外对照，不会关闭感知和行动。</p>
<p>这里只报告本次运行的真实差异。绕墙、碰撞推开、信息素蒸发和腿部动作不等于神经学习成果；当前没有启用参数回退，也不读取摄像头画面。渲染帧率不是学习频率。</p><button id="understood" class="primary">进入现场</button></dialog>`;
