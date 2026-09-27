# v4 英中旁白与完整大纲

## 00:00 — Training finished. Learning finished?

Training is finished.
训练结束了。

Does learning have to stop?
学习也必须停下来吗？

The weights stay fixed, but the world keeps changing.
权重固定了，世界却还在变化。

What if experience could reshape a few connections?
能不能让经历改变其中几条连接？

Not by retraining everything, but by learning locally.
不重训整个网络，而是在局部学习。

Let's put that question inside a moving ant.
让一只行走的蚂蚁来演示这个问题。

## 00:24 — Same start. Different updates.

Two ants, with the same starting controller.
两只蚂蚁，从相同的控制器出发。

Now give them the same sideways push.
现在给它们相同的侧向扰动。

One keeps its weights fixed.
一只保持权重不变。

The other can change a small steering correction.
另一只可以调整一个小小的转向修正量。

Watch the paths separate.
看，它们的路线开始分开。

What changed inside?
内部究竟改变了什么？

## 00:44 — One number changes the movement

Not the legs.
不是腿变了。

Keep those skills, and add a small correction.
保留已有技能，再加上一点修正。

Slide this number, and her heading changes.
移动这个数，朝向就跟着改变。

The effective weight is the stable part plus the fast part.
实际使用的权重，由稳定部分加上快速变化组成。

## 01:00 — A consequence arrives later

Now suppose the consequence arrives a little later.
假设行动的后果晚了一点才出现。

Which earlier action should it affect?
它应该影响先前的哪次行动？

Each action leaves a fading trace of local activity.
每次行动都留下逐渐衰减的局部活动记录。

New traces join the old ones.
新的记录与尚未消退的旧记录叠加。

This is an eligibility trace.
这就是资格迹。

It preserves a connection to the recent past.
它保留了与近期活动的联系。

## 01:24 — How much should a connection change?

The trace alone doesn't say whether a change is useful.
光有资格迹，还不知道这次改变有没有用。

A small learned modulator reads the feedback.
一个经过训练的小调制器读取反馈。

It can strengthen the candidate, weaken it, or reverse it.
它可以增强候选变化、减弱它，或让它反向。

Multiply the trace by that signed signal.
把资格迹乘上这个有正负的信号。

Now we have a local weight change.
这样就得到了一次局部权重变化。

Bound its size before writing it.
写入之前，还要限制它的大小。

## 01:48 — A decision, not an obligation

But a candidate is not an instruction.
不过，提出候选不等于必须执行。

After every four actions, check again.
每经过四次行动，再检查一次。

Apply this one.
这一次，采用。

Skip the next.
下一次，跳过。

Skipping leaves the weights alone, not the memory.
跳过不改权重，但记忆仍在更新。

The ant keeps moving between decisions.
两次判断之间，蚂蚁照样继续行动。

## 02:08 — Learning the rule, learning with it

Where does this update rule come from?
那这套更新规则从哪里来？

We train it across changing situations.
我们在变化的情境中训练它。

That training can still use backpropagation.
这个训练阶段，仍然可以使用反向传播。

During use, feedback changes the fast connections locally.
运行时，反馈在局部改变快速连接。

Learning the rule, then learning with it.
先学会更新规则，再用这套规则继续学。

## 02:26 — Adaptation is not update selection

Can we tell adaptation from better update selection?
能否区分适应收益与更新选择的收益？

In our completed trials, learned updates beat no updates.
在已完成试验中，学得接受比不更新交付更多。

But they did not beat always accepting.
但它没有超过固定接受。

Those are different tests.
这是两种不同的检验。

## 02:40 — The next action cannot wait

And the next step still has a deadline.
下一次行动，仍然有截止时间。

Use the latest complete weights while a change is prepared.
准备修改时，先用最近一次完整提交的权重。

Commit it when ready.
准备好了，再提交。

Learning must not hold the next action hostage.
不能让学习拖住下一次行动。

## 02:54 — Learning While Acting

Learning While Acting.
边行动，边学习。

Changing through experience, without stopping to learn.
从经历中改变，不为学习停下脚步。
