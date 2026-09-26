from mathhackson.training.comparison.injury_scores import Row, describe


def test_describe_keeps_harmful_and_tied_candidates():
    common = dict(seed=1, tick=1, individual=0, death_window=False, prediction=-1.,
                  delivery_difference=0, death_difference=0, exhaustion_difference=0,
                  injury_difference=0., colony_delivery_difference=0, colony_death_difference=0)
    rows = [Row(**common, originally_accepted=False, reward_difference=2.),
            Row(**common, originally_accepted=True, reward_difference=-2.),
            Row(**common, originally_accepted=False, reward_difference=1e-8)]
    group = describe(1, rows)
    assert group.improved_rejected == group.worse_accepted == group.tied_rejected == 1
    assert group.points == 3 and group.improved_accepted == 0
    assert describe(2, rows).points == 0
