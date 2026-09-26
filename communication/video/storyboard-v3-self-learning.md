# MathHackson｜三分钟英文概念视频完整分镜

## 片名与表达范围

**Learning While Acting**

180秒数学可视化概念片。英文旁白379词；画面、字幕、公式标签、图例和片尾全部英文。中文仅为制作说明，不进入视频。表达架构提案，不报告实验成绩。

中心问题：

> Can a model learn when to change itself, using local feedback, while it keeps acting?

主线：环境变了 → 常见的权重学习方式 → 小范围快更新 → 局部纠正 → 学习是否更新 → 更新门本身继续学习 → 有限可逆修改 → 不阻塞动作的时间预算 → 蚂蚁验证。

不再出现Transformer、KV cache、压缩历史或无限记忆主线。

## 来源和比赛切片

本轮读取用户指定的 `/home/Akira/Projects/v2v/docs/training-lifecycle/overview.html` 中嵌入的 `intent-and-boundaries.md`、`architecture.md`、`learning-candidate.md`、`growth.md` 和 `maturity.md`；对照比赛仓库 `docs/competition/v2v-extraction-scope.md`、`CONTEXT.md`、主checkout最新 `RESEARCH.md`、`designs/basic-update-learning.md` 与 `docs/engineering/update-decision.md`。

只使用比赛切片的对象：小型个体模型、冻结动作底座、局部反馈、少量快参数、更新接受/跳过及有明确归属的近期参数回退。“更新许可在运行过程中继续学习”是用户本轮指定的构想，不把当前离线拟合实现说成已做到。

局部纠正例子和单个更新门的局部信用式只用于解释想法，不构成新冻结设计；不带入完整V2V的计算块拓扑、分层快照、长期记忆库、需求系统、睡眠、结构扩容或自适应计算招募机制。原V2V只读，比赛研究数据库与模型不变。

“不需要反向传播”的片中表述为：

> No full-network backpropagation during online adaptation.

不等于完全无导数、无计算开销，也不表示所有基础模块的初始训练均不用反向传播。单个局部预测器和门控可以使用局部解析导数；不沿全网或全历史传播。

## 节奏与视觉规则

379词按约140–145词/分钟试录，余下约18–23秒留给公式与观察；尚未实际录音计时。不逐符号朗读公式。

黑色或极深背景；几何图形、连接、矩阵、时间轴构成画面。所有英文旁白原创；不复制3Blue1Brown原句、不冒充其作品或要求模仿其声纹。

同一对象跨镜头延续：控制点的动作箭头 → 网络输出 → 冻结动作通路 → 实时时隙 → 蚂蚁动作。快参数矩阵与更新门反复使用，避免一页一个新框图。

白色表示实际观察；蓝色表示稳定技能；黄色表示候选更新；青色表示局部控制/信用；红色表示误差或撤回。必须辅以文字/形状，不只靠颜色传达判断。

公式固定正对屏幕，不随3D矩阵旋转。每幕最多保留两条主公式；先让图形产生含义，再出现符号。

方案段边角标 `Proposed architecture`；局部算法示意标 `Illustrative local rule`；实验段标 `Proposed experiment`。不使用假性能数字、结果曲线、发光大脑、装饰性粒子或虚构损失地形。

## 总时间表

| 场景 | 时间 | 英文标题 | 旁白词数 |
|---|---|---|---:|
| S01 | 00:00–00:15 | The rule changes | 33 |
| S02 | 00:15–00:32 | How weights usually change | 36 |
| S03 | 00:32–00:50 | Keep the skills. Change a small part. | 37 |
| S04 | 00:50–01:10 | A correction from what actually happened | 42 |
| S05 | 01:10–01:28 | Should this update be applied? | 40 |
| S06 | 01:28–01:53 | Learning how to update | 46 |
| S07 | 01:53–02:12 | A change need not be permanent | 43 |
| S08 | 02:12–02:32 | The next action cannot wait | 45 |
| S09 | 02:32–02:52 | A small world to test it | 42 |
| S10 | 02:52–03:00 | Learning While Acting | 15 |

---

## S01｜00:00–00:15｜The rule changes

**观众问题**：反应很快，为什么仍可能跟不上环境变化？

### 英文旁白

> Watch this controller follow a moving target. Now the controls change. It still responds, but its old rule sends it the wrong way. Can it learn the new rule while it keeps moving?

### 画面文字

`A changing world` / `Reacting is not the same as learning`

### 公式

\[
a_t=\pi_{\theta_t}(o_t,h_t)
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 00:00–00:04 | 白色目标点沿弧线移动，蓝色控制点跟随。只留目标、控制点、动作箭头。 |
| 00:04–00:08 | 动作—位移关系改变；同一个动作箭头产生偏离的实际位移。模型没有获得切换标签。 |
| 00:08–00:12 | 轨迹暂时冻结供观察，底部决策时钟继续走；动作仍在输出。出现 Reacting is not the same as learning。 |
| 00:12–00:15 | 从控制点拉出小型神经网络。原动作箭头接到网络输出。 |

**布局**：主画面占中央70%；标题左上；决策时钟在底部，无加载条或模拟毫秒数字。

**转场**：放大控制点背后的网络，保留同一动作箭头。

**边界**：概念示意；不暗示一切固定参数网络都无法通过内部状态调整行为。本例选择原策略不适用的新动作关系。

---

## S02｜00:15–00:32｜How weights usually change

**观众问题**：我们想替换在线适应里的哪一种计算依赖？

### 英文旁白

> A common way to change a neural network is backpropagation: measure an error, then trace it backward to adjust the weights. Our question is whether adaptation can happen locally, without repeatedly backpropagating through the whole network.

### 画面文字

`Backpropagation` / `A common training method` / `Local adaptation?`

### 公式

\[
\theta^{+}=\theta-\eta\nabla_\theta\mathcal L
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 00:15–00:19 | 活动从左向右穿过三层。输出与实际后果并排，差值亮起。 |
| 00:19–00:24 | 误差信号沿依赖路径向后；相关权重格子高亮。反传只在学习示意中出现。 |
| 00:24–00:28 | 依赖路径压缩成梯度更新式；反向路径与梯度项同色。 |
| 00:28–00:32 | 选中网络的一小块，其余变暗。Local adaptation? 出现在选中部分旁。 |

**布局**：网络居中；公式右下。没有Transformer或长上下文铺垫。

**转场**：局部选框拉开成为快参数矩阵。

**边界**：公式是梯度下降，动画解释梯度的反向计算；反传与梯度下降不是同义词。不画成传统学习必然停机，或每次推理都需要反传。

---

## S03｜00:32–00:50｜Keep the skills. Change a small part.

**观众问题**：新经验具体改变哪里？

### 英文旁白

> Keep the basic controller. Put a small, adjustable correction beside it. The effective weights are the stable weights plus this fast change. New experience can now alter a limited part of the model, instead of rewriting everything.

### 画面文字

`Stable weights` / `Fast changes` / `Effective weights` / `Motor skills: frozen`

### 公式

\[
W_t=S+F_t
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 00:32–00:36 | 轻3D：两张同形矩阵前后分开，S与F；F起初为零。 |
| 00:36–00:41 | 几条反馈进入F，只亮少数元素；S保持不动。 |
| 00:41–00:46 | 相机转向正面，两板逐元素相加成W，不能画成拼接。 |
| 00:46–00:50 | 拉回 Local signals → Adaptive model → Frozen motor → Action；S+F在Adaptive model内部。 |

**布局**：矩阵占左60%，公式右侧，动作路径留在底部。

**转场**：W的一个元素放大为单连接w。

**边界**：S+F只表示选定的小型可适应部分，不让整个动作底座也发生参数变化；不引入自动巩固控制器或长期外存。

---

## S04｜00:50–01:10｜A correction from what actually happened

**观众问题**：不做全网反传，局部更新从哪里来？

### 英文旁白

> Start with one connection. It predicts the effect of an action. The next observation arrives, and the difference gives a local correction. Only this little calculation changes the weight. No error signal has to travel backward through the rest of the network.

### 画面文字

`Local prediction example` / `Prediction` / `Observed outcome` / `Local correction`

### 公式（按顺序出现，不同时铺满）

\[
\hat c=wx,
\qquad e=c-\hat c,
\qquad \Delta w=\eta e x
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 00:50–00:55 | 输入x经过连接w，得到虚线预测c-hat。只显示c-hat=wx。 |
| 00:55–01:00 | 下一时刻实际观测c到达；括号量出差值，折成e。 |
| 01:00–01:05 | 先前保留的x与当前误差e结合，形成Δw，此时才显示局部纠正式。 |
| 01:05–01:10 | 黄色改变量进入待定薄片，暂不写入W；其他模块没有反向路径。 |

**布局**：左侧单连接，右侧预测/实际位置，底部逐项出现公式。

**转场**：待定薄片向矩阵移动，半路出现一道门。

**边界**：这是局部线性例子，不是正式比赛算法声明。Δw=ηex仍是局部平方误差的梯度方向，不能叫完全无梯度。目标必须晚于预测到达，不能把同一目标拟合当作未来测试成绩。

---

## S05｜01:10–01:28｜Should this update be applied?

**观众问题**：有误差，为什么不直接更新？

### 英文旁白

> But an error is not always a reason to learn. A changed rule and random noise can both produce surprises. So we propose a gate that learns when to apply a candidate correction, and when to leave the weights alone.

### 画面文字

`Candidate update` / `Apply` / `Skip` / `Learned update gate`

### 公式

\[
F_{t+1}=F_t+z_t\Delta F_t,
\qquad z_t\in\{0,1\}
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 01:10–01:14 | 两条短输入：一条规律持续改变，一条单次扰动；某一时刻误差大小相近。 |
| 01:14–01:19 | ΔF到达门；z=0时权重保持，z=1时写入。 |
| 01:19–01:24 | 局部观察、状态、已到达反馈进入门；没有规则变了/噪声真值标签输入。 |
| 01:24–01:28 | 门旁出现参数φ；镜头从候选转到门本身。 |

**布局**：门居中，S+F在右；Apply和Skip同时有文字与形状区分。

**转场**：放大φ，紧接着解释这个门怎样学习。

**边界**：不把固定误差阈值冒充学出来的门；部署图不显示先试遍两条未来再选最好结果。

---

## S06｜01:28–01:53｜Learning how to update

**观众问题**：在学习的过程中学习，到底多学了什么？

### 英文旁白

> Now the important part: the gate learns too. When it tries an update, it keeps a short record of that choice. Later consequences provide feedback on the choice. That feedback adjusts its future update decisions. Experience changes both the controller and the way the controller learns.

### 画面文字

`Learning the gate` / `Choice now` / `Consequence later` / `Local credit`

### 公式

\[
\Delta\phi=\beta A E,
\qquad A=R-\widehat R
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 01:28–01:33 | 门做出一次真实试用，保存当时输入、执行选择、局部资格E的小记录；里面没有未来。 |
| 01:33–01:39 | 上方动作继续；下方记录沿时间轴前进，等后果R到达。 |
| 01:39–01:45 | R与试用前预测的R-hat相减成A，再与原资格E结合。 |
| 01:45–01:49 | 形成Δφ，回到同一个门，改变其之后的更新倾向，不修改过去的决定。 |
| 01:49–01:53 | 拉远：F在变化，φ也在变化。屏幕留Learning the gate。 |

**布局**：上方正在行动；下方是更新记录与后到反馈。Δφ是唯一主公式。

**转场**：回到一笔已被接受的快改动，准备说明可以撤回。

**边界**：这是最新讨论中局部控制学习候选的单门解释，不是全系统长期回报的精确梯度保证，不新增永久特殊块类型，不改写正式科研设计。后果提供有噪声的信用，不保证已辨认因果。

---

## S07｜01:53–02:12｜A change need not be permanent

**观众问题**：快速学习怎样保护已有技能？

### 英文旁白

> A bad update should not erase a useful skill. Keeping recent changes separate lets us withdraw a particular change that turns out to be harmful. The world does not rewind. Only that tracked adjustment is removed; the rest of the experience still happened.

### 画面文字

`Tracked recent change` / `Withdraw` / `The world keeps moving`

### 公式

\[
F\leftarrow F-r\delta_j,
\qquad 0\le r\le1
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 01:53–01:58 | 轻3D再次展开S/F，F上某笔快改动带编号j。 |
| 01:58–02:03 | 后续反馈出现，选中仍可归属的δ_j，按r从F减去。 |
| 02:03–02:08 | 重新合成W；S不变，其他改动保留，时间线继续。 |
| 02:08–02:12 | 控制点不回旧位置，实际经历不倒带；出现The world keeps moving。 |

**布局**：矩阵主体，下方实际轨迹持续前进。

**转场**：时间线扩展成实时响应的双轨时间轴。

**边界**：δ_j是仍被记录、未巩固消去身份的剩余改变量。撤回不等于反事实重训，不撤销后续已受影响的经验、状态或世界。没有无限撤回能力。

---

## S08｜02:12–02:32｜The next action cannot wait

**观众问题**：怎样把在线学习与实时动作在架构上分开？

### 英文旁白

> Meanwhile, actions keep coming. The controller uses the latest committed weights. Learning runs within its own budget; an unfinished update waits. Removing global backpropagation is only part of this design. The real requirement is that learning must not make the next action miss its deadline.

### 画面文字

`Action loop` / `Learning budget` / `Commit` / `Next action deadline` / `Design target`

### 公式

\[
T_{\mathrm{sense}\rightarrow\mathrm{act}}\le D
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 02:12–02:17 | 上轨按时隙输出动作，下轨执行有限局部更新。 |
| 02:17–02:22 | 一个更新超出本轮剩余预算，留在待定区；下一动作读取上一完整W继续。 |
| 02:22–02:27 | 已完成更新在下个安全边界一次性Commit；禁止读取半写入的权重。 |
| 02:27–02:32 | 出现时延预算式和Design target；不展示没有测量过的毫秒数或倍数。 |

**布局**：时间向右，上55%为动作、下35%为更新；公式正对屏幕，无需3D。

**转场**：动作时隙的小控制点变成蚂蚁，时间轴下沉成试验场边框。

**边界**：双轨表示职责和提交语义，不是已有硬实时调度能力证明。感知、计算、资源竞争、同步与提交均应计入测量；没有全局反传不自动保证更快。具体调度方法尚需实现。

---

## S09｜02:32–02:52｜A small world to test it

**观众问题**：为什么用蚂蚁，未来怎样检验？

### 英文旁白

> We want to test this in a small ant world: local signals, a few actions, and rules that change without warning. Compare the same starting model with learning disabled, fixed updates, and learned updates. Measure adaptation speed, retained skills, and response time.

### 画面文字

`Local signals` / `Frozen motor skills` / `No updates` / `Fixed updates` / `Learned updates` / `Same starting model` / `Adaptation · Skill retention · Response time`

### 公式

\[
(o_t,h_t)\longrightarrow d_t\longrightarrow a_t
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 02:32–02:37 | 俯视小世界，蚂蚁只看到局部信号，远处压暗；方向模型与冻结动作底座短暂标注。 |
| 02:37–02:43 | 同一初始化复制三个隔离世界：No updates / Fixed updates / Learned updates。 |
| 02:43–02:48 | 配对的外部变化发生，模型没有切换通知。不画胜负轨迹或结果曲线。 |
| 02:48–02:52 | 出现三项测量名称，保留同基础能力与同观察权限的含义。 |

**布局**：三个并排小窗，纯2D；不做复杂游戏UI。右下Proposed experiment。

**转场**：三个窗收回动作—后果—更新闭环。

**边界**：只是比赛切片的验证方案，不宣称优越性。正式实验还应按研究设计保留同能力普通MLP等外部对照，本片用同模型三组先解释核心因果比较。

---

## S10｜02:52–03:00｜Learning While Acting

**观众问题**：最后记住什么？

### 英文旁白

> Can a model learn when to change itself, using local feedback, while it keeps acting?

### 画面文字

`Learning While Acting`

`Local feedback. Adaptive updates. Continuous action.`

`An architecture proposal · MathHackson`

### 公式图

\[
\text{Act}\longrightarrow\text{Observe}\longrightarrow\text{Update}\longrightarrow\text{Act}
\]

### 逐镜头

| 时间 | 画面与运动 |
|---|---|
| 02:52–02:56 | 完整闭环；内部保留F与φ两种调整，动作脉冲仍持续。 |
| 02:56–03:00 | 片名与副行出现，最后1秒不加元素，直接淡出。 |

**边界**：收束为研究问题，不说“重新定义智能”，不回到Transformer或压缩记忆。

---

## 数学和时序附录

### 局部纠正仍可包含导数

\[
\ell(w)=\tfrac12(c-wx)^2,
\qquad -\partial\ell/\partial w=(c-wx)x=ex.
\]

因此Δw=ηex是局部梯度方向；它无需沿其他模块回传，但不是“无梯度”。这里只解释一个可读的局部例子，不声称它能训练任意深层/长时序系统。

### 更新门的局部信用展开（不全部塞进正片）

\xi_t是当时可得的观察、状态与后果摘要；φ是门参数。一个解释性伯努利门可写为：

\[
p_t=\sigma(\phi_t^\top\xi_t),
\qquad z_t\sim\operatorname{Bernoulli}(p_t).
\]

在实际选择发生时保存局部资格：

\[
E_t=\nabla_\phi\log P_{\phi_t}(z_t\mid\xi_t)
=(z_t-p_t)\xi_t.
\]

后果在t+τ到来后，才产生局部控制更新：

\[
\Delta\phi_{t+\tau}
=\beta(R_{t:t+\tau}-\widehat R_t)E_t.
\]

R-hat必须在选择前形成，R在之后真实到达。没有未来分支、正确门标签或隐藏世界状态进入输入。本例一次只画一笔未决控制记录，以免混淆归因。此为最新讨论中局部候选的解释，不是最终新冻结规则，也不是全系统长期梯度保证。局部导数不递归穿过全网历史。

### 参数撤回范围

δ_j仅指仍被记录的剩余快改动；不是原样删除一条经历并重新训练。保存K笔P维完整改动时记录成本随PK增长，本片不宣称常数开销。

### 实时目标范围

D是任务规定的截止时间。两轨图只表示动作读取已提交版本、更新受预算约束的目标。最终必须测感知到动作的完整时延和尾部时延，包括资源争用与提交成本。具体实现可采用分块/抢占/资源隔离等，但本片不假装这些已经完成。

## 3D制作要求

1. S03中同形S/F板前后分开，相机约20–30度轻转；逐元素合成W。符号始终正对屏幕。
2. S07重用同一对象和角度；带j编号的薄片只从F减去，其余元素与时间线不倒退。
3. 不做虚构loss landscape、KV仓库、压缩云团或电影化蚂蚁场景。

## 配音文件

`narration-v3.en.txt`只含与以上十幕一致的英文旁白，379词。

## 公开参考与贡献边界

- 3Blue1Brown, Backpropagation calculus: https://www.3blue1brown.com/lessons/backpropagation-calculus/
- 3Blue1Brown, What is backpropagation really doing?: https://www.3blue1brown.com/lessons/backpropagation/
- Bellec et al. 2020, A solution to the learning dilemma for recurrent networks of spiking neurons: https://www.nature.com/articles/s41467-020-17236-y
- Chen et al. 2016, Learning to Learn without Gradient Descent by Gradient Descent: https://arxiv.org/abs/1611.03824

参考只提供讲述顺序和方法边界，不提供本项目结果证据。本片的辨识点是比赛小型系统中局部快变化、可学习的更新许可、有限可逆修改与动作时限约束的组合，不以“局部学习”或“学习更新”通用概念声称首创。

## 验收和下一阶段

时间连续覆盖180秒；旁白379词；所有视频文本英文；F与φ变化分清；预测/选择/后果/更新不越过真实时间；局部算法图标明示意；无假实验成绩、无绝对无梯度或毫秒性能承诺。

分镜确认前不继续扩写Manim。下一阶段先试录英文旁白，再按实际音频校准停顿和镜头长度。原来的story_prototype.py只是历史工具冒烟样片，不再作为内容方向。
