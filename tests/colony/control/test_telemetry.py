"""展示摘录来自最近实际前向计算，读取不更新模型。"""
from mathhackson.colony.world import World
from mathhackson.colony.display.telemetry import network_view


def test_network_view_is_read_only_and_uses_executed_inference_weights():
    world=World(73,1,warmup=5)
    world.tick()
    ant=world.ants[0];before=ant.brain.fingerprint()
    view=network_view(world,0)
    assert view is not None and view.widths==[8,24,16,3]
    assert len(view.connections)==102
    edge=next(e for e in view.connections if e.layer==2 and e.source==0 and e.target==0)
    assert edge.weight==float(ant.brain.inference_head[0,0])
    assert ant.brain.fingerprint()==before
    assert view.updates==1
    ant.brain.frozen=True
    assert all(e.update==0 for e in network_view(world,0).connections)


def test_rule_controller_has_no_neural_graph():
    world=World(71,1,mode='rules')
    world.tick()
    assert network_view(world,0) is None
