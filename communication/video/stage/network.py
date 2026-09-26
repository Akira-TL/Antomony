"""Original animated network drawing for the question-led opening and closing.

The numerical activity comes from a small feed-forward calculation. Connection
widths encode weights; moving pulses and node fills encode activity. They are
intentionally independent, so a fixed-weight network is not drawn as inactive.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from manimlib import VGroup, Circle, Line, Dot, ORIGIN

from communication.video.scenes.ant_drawing import BG, BLUE, TEAL, GOLD, WHITE, GRAY


@dataclass(frozen=True)
class EdgeIdentity:
    layer: int
    output: int
    input: int


class LivingNetwork(VGroup):
    def __init__(self, center: np.ndarray = ORIGIN, size: float = 1.0):
        super().__init__()
        rng = np.random.default_rng(103)
        counts = (3, 5, 5, 2)
        self.weights = tuple(rng.normal(0, .72, (b, a)) for a, b in zip(counts, counts[1:]))
        self.initial = tuple(w + rng.normal(0, .22, w.shape) for w in self.weights)
        self.biases = tuple(rng.normal(0, .15, b) for b in counts[1:])
        self.layers = tuple(VGroup(*(
            Circle(radius=.15).set_fill(BG, 1).set_stroke(BLUE, 1.8)
            .move_to([x, y, 0])
            for y in np.linspace(1.3, -1.3, count)
        )) for x, count in zip((-2.85, -.95, .95, 2.85), counts))
        self.edge_layers = []
        self.identities: list[EdgeIdentity] = []
        self.edges = VGroup()
        for layer in range(3):
            group = VGroup()
            for row, destination in enumerate(self.layers[layer + 1]):
                for col, source in enumerate(self.layers[layer]):
                    edge = Line(source.get_center(), destination.get_center(), buff=.16)
                    edge.set_stroke(BLUE, 1.2, .38)
                    group.add(edge)
                    self.edges.add(edge)
                    self.identities.append(EdgeIdentity(layer, row, col))
            self.edge_layers.append(group)
        self.pulses = VGroup(*(Dot(radius=.040).set_color(WHITE) for _ in range(6)))
        self.add(self.edges, *self.layers, self.pulses)
        self.scale(size).move_to(center)
        self.update_at(0)

    @staticmethod
    def mix(t: float) -> float:
        fraction = float(np.clip(t / 1.25, 0, 1))
        return fraction * fraction * (3 - 2 * fraction)

    def parameters(self, t: float) -> tuple[np.ndarray, ...]:
        alpha = self.mix(t)
        return tuple(a * (1 - alpha) + b * alpha for a, b in zip(self.initial, self.weights))

    def activations(self, t: float, parameter_time: float | None = None) -> tuple[np.ndarray, ...]:
        inputs = np.array([np.sin(1.25 * t), np.cos(.83 * t + .3), np.sin(.58 * t + 1.)])
        values = [inputs]
        for w, b in zip(self.parameters(t if parameter_time is None else parameter_time), self.biases):
            values.append(np.tanh(w @ values[-1] + b))
        return tuple(values)

    def prediction(self, t: float) -> float:
        return float(self.activations(t, parameter_time=2.)[-1][0])

    def update_at(self, t: float) -> "LivingNetwork":
        weights = self.parameters(t)
        values = self.activations(t)
        for nodes, activation in zip(self.layers, values):
            for node, val in zip(nodes, activation):
                node.set_fill(BLUE if val >= 0 else TEAL, .12 + .75 * abs(float(val)))
        for edge, identity in zip(self.edges, self.identities):
            w = float(weights[identity.layer][identity.output, identity.input])
            edge.set_stroke(BLUE if w >= 0 else GRAY, .6 + 1.6 * min(abs(w), 1.6), .19 + .22 * min(abs(w), 1.))
        for i, pulse in enumerate(self.pulses):
            layer = i // 2
            candidates = self.edge_layers[layer]
            edge = candidates[(i * 7 + 3) % len(candidates)]
            phase = (t * .74 - .2 * layer + .47 * (i % 2)) % 1.
            pulse.move_to(edge.get_start() * (1 - phase) + edge.get_end() * phase)
            pulse.set_opacity(.22 + .76 * np.sin(np.pi * phase) ** 2)
        return self

    def selected_edge(self) -> Line:
        # One last-layer connection, not a fictitious full-network update.
        return self.edge_layers[-1][2]
