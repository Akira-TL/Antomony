"""Independent v4 production plan; old videos and research files are read-only."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'communication/video/rendered/v4'
VOICE='en-US-AndrewNeural'
RATE=48000
FPS=60
FILENAME='Learning_While_Acting_v4_1080p60.mp4'
@dataclass(frozen=True)
class Sentence:
    start: float
    en: str
    zh: str
@dataclass(frozen=True)
class Chapter:
    scene: str
    title: str
    seconds: int
    sentences: tuple[Sentence,...]
CHAPTERS=(
Chapter('C00LearningContinues','Training finished. Learning finished?',24,(
Sentence(.4,'Training is finished.','训练结束了。'),
Sentence(2.2,'Does learning have to stop?','学习也必须停下来吗？'),
Sentence(5.,'The weights stay fixed, but the world keeps changing.','权重固定了，世界却还在变化。'),
Sentence(10.,'What if experience could reshape a few connections?','能不能让经历改变其中几条连接？'),
Sentence(14.7,'Not by retraining everything, but by learning locally.','不重训整个网络，而是在局部学习。'),
Sentence(19.,"Let's put that question inside a moving ant.",'让一只行走的蚂蚁来演示这个问题。'))),
Chapter('C01MatchedAnts','Same start. Different updates.',20,(
Sentence(.4,'Two ants, with the same starting controller.','两只蚂蚁，从相同的控制器出发。'),
Sentence(4.,'Now give them the same sideways push.','现在给它们相同的侧向扰动。'),
Sentence(7.6,'One keeps its weights fixed.','一只保持权重不变。'),
Sentence(10.4,'The other can change a small steering correction.','另一只可以调整一个小小的转向修正量。'),
Sentence(14.7,'Watch the paths separate.','看，它们的路线开始分开。'),
Sentence(17.,'What changed inside?','内部究竟改变了什么？'))),
Chapter('C02StableAndFast','One number changes the movement',16,(
Sentence(.4,'Not the legs.','不是腿变了。'),
Sentence(2.1,'Keep those skills, and add a small correction.','保留已有技能，再加上一点修正。'),
Sentence(6.,'Slide this number, and her heading changes.','移动这个数，朝向就跟着改变。'),
Sentence(10.4,'The effective weight is the stable part plus the fast part.','实际使用的权重，由稳定部分加上快速变化组成。'))),
Chapter('C03Eligibility','A consequence arrives later',24,(
Sentence(.4,'Now suppose the consequence arrives a little later.','假设行动的后果晚了一点才出现。'),
Sentence(4.4,'Which earlier action should it affect?','它应该影响先前的哪次行动？'),
Sentence(7.8,'Each action leaves a fading trace of local activity.','每次行动都留下逐渐衰减的局部活动记录。'),
Sentence(12.7,'New traces join the old ones.','新的记录与尚未消退的旧记录叠加。'),
Sentence(16.2,'This is an eligibility trace.','这就是资格迹。'),
Sentence(19.3,'It preserves a connection to the recent past.','它保留了与近期活动的联系。'))),
Chapter('C04ModulatedChange','How much should a connection change?',24,(
Sentence(.4,"The trace alone doesn't say whether a change is useful.",'光有资格迹，还不知道这次改变有没有用。'),
Sentence(4.7,'A small learned modulator reads the feedback.','一个经过训练的小调制器读取反馈。'),
Sentence(8.8,'It can strengthen the candidate, weaken it, or reverse it.','它可以增强候选变化、减弱它，或让它反向。'),
Sentence(13.7,'Multiply the trace by that signed signal.','把资格迹乘上这个有正负的信号。'),
Sentence(17.7,'Now we have a local weight change.','这样就得到了一次局部权重变化。'),
Sentence(21.,'Bound its size before writing it.','写入之前，还要限制它的大小。'))),
Chapter('C05FourSteps','A decision, not an obligation',20,(
Sentence(.4,'But a candidate is not an instruction.','不过，提出候选不等于必须执行。'),
Sentence(3.8,'After every four actions, check again.','每经过四次行动，再检查一次。'),
Sentence(7.8,'Apply this one.','这一次，采用。'),
Sentence(10.,'Skip the next.','下一次，跳过。'),
Sentence(12.5,'Skipping leaves the weights alone, not the memory.','跳过不改权重，但记忆仍在更新。'),
Sentence(16.3,'The ant keeps moving between decisions.','两次判断之间，蚂蚁照样继续行动。'))),
Chapter('C06LearnTheRule','Learning the rule, learning with it',18,(
Sentence(.4,'Where does this update rule come from?','那这套更新规则从哪里来？'),
Sentence(3.6,'We train it across changing situations.','我们在变化的情境中训练它。'),
Sentence(6.8,'That training can still use backpropagation.','这个训练阶段，仍然可以使用反向传播。'),
Sentence(10.2,'During use, feedback changes the fast connections locally.','运行时，反馈在局部改变快速连接。'),
Sentence(14.,'Learning the rule, then learning with it.','先学会更新规则，再用这套规则继续学。'))),
Chapter('C07SeparateTheClaims','Adaptation is not update selection',14,(
Sentence(.4,'Can we tell adaptation from better update selection?','能否区分适应收益与更新选择的收益？'),
Sentence(4.2,'Learned updates delivered more than no updates.','学得接受，比不更新交付更多。'),
Sentence(8.2,'But always accepting delivered slightly more.','不过，固定接受的交付量还略多一些。'),
Sentence(11.6,'Those are different tests.','这是两种不同的检验。'))),
Chapter('C08ActionDeadline','The next action cannot wait',14,(
Sentence(.4,'And the next step still has a deadline.','下一次行动，仍然有截止时间。'),
Sentence(3.8,'Use the latest complete weights while a change is prepared.','准备修改时，先用最近一次完整提交的权重。'),
Sentence(8.5,'Commit it when ready.','准备好了，再提交。'),
Sentence(10.3,'Learning must not hold the next action hostage.','不能让学习拖住下一次行动。'))),
Chapter('C09KeepLearning','Learning While Acting',6,(
Sentence(.3,'Learning While Acting.','边行动，边学习。'),
Sentence(2.1,'Changing through experience, without stopping to learn.','从经历中改变，不为学习停下脚步。'))),
)
assert sum(c.seconds for c in CHAPTERS)==180
for c in CHAPTERS:
    assert all(0<=s.start<c.seconds for s in c.sentences)
    assert all('\n' not in s.en and '\n' not in s.zh for s in c.sentences)
    assert max(map(lambda s:len(s.en),c.sentences))<=80
