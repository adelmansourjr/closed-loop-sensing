#!/usr/bin/env python3
"""Compare exact fixed and adaptive measurement policies on a rotating signal."""

import argparse
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


PHASES = 16
TICKS = 3
AXES = (0, 2, 4, 6)
SECTORS = (np.arange(PHASES) - 2) % PHASES // 4
PHASE_ANGLES = 2 * np.pi * np.arange(PHASES) / PHASES
GUIDE_HORIZONTAL = np.isin(np.arange(PHASES) % 8, [0, 1, 2, 6, 7])
CLASSES = np.eye(4)[np.repeat(SECTORS, 2)]
PRIOR = np.tile(np.array([0.65, 0.35]) / PHASES, PHASES)
TOL = 1e-14


@dataclass(frozen=True)
class Law:
    phase_jitter: tuple[float, float, float] = (0.12, 0.76, 0.12)
    high_to_high: float = 0.86
    low_to_high: float = 0.23
    reliability: tuple[float, float] = (0.82, 0.32)
    erasure: tuple[float, float] = (0.06, 0.24)
    guide_flip: float = 0.10
    reset: bool = False


NOMINAL = Law()
FAMILIES = {
    "nominal": NOMINAL,
    "attenuation": replace(NOMINAL, reliability=(0.44, 0.13)),
    "persistent_loss": replace(NOMINAL, high_to_high=0.94, low_to_high=0.10, erasure=(0.12, 0.65)),
    "clock_drift": replace(NOMINAL, phase_jitter=(0.10, 0.46, 0.44)),
    "guide_drift": replace(NOMINAL, guide_flip=0.40),
    "uniform_final_reset": replace(NOMINAL, reset=True),
}


class RotatingSignal:
    def __init__(self, law):
        self.law = law
        phase_transition = np.zeros((PHASES, PHASES))
        start_phase = np.arange(PHASES)
        for step, probability in enumerate(law.phase_jitter):
            phase_transition[start_phase, (start_phase + step) % PHASES] = probability
        quality_transition = np.array([
            [law.high_to_high, 1 - law.high_to_high],
            [law.low_to_high, 1 - law.low_to_high],
        ])
        transition = np.kron(phase_transition, quality_transition)
        last_phase = np.full((PHASES, PHASES), 1 / PHASES)
        self.transitions = (transition, transition,
                            np.kron(last_phase, quality_transition) if law.reset else transition)

        axis_angle = PHASE_ANGLES[list(AXES)]
        projection = np.cos(PHASE_ANGLES[None, :] - axis_angle[:, None])
        positive = (0.5 + 0.5 * projection[:, :, None] * np.array(law.reliability)).reshape(len(AXES), -1)
        erasure = np.broadcast_to(np.tile(law.erasure, PHASES), positive.shape)
        self.emission = np.stack([
            (1 - positive) * (1 - erasure),
            positive * (1 - erasure),
            erasure,
        ], axis=1)
        horizontal = np.repeat(np.where(GUIDE_HORIZONTAL, 1 - law.guide_flip, law.guide_flip), 2)
        self.guide = np.stack([1 - horizontal, horizontal])
        for array, axis in [*((matrix, 1) for matrix in self.transitions),
                            (self.emission, 1), (self.guide, 0)]:
            assert np.min(array) >= 0
            np.testing.assert_allclose(array.sum(axis=axis), 1, atol=TOL, rtol=0)

    def initial(self, guide):
        return PRIOR * self.guide[guide]

    def step(self, mass, tick, axis=None, outcome=None):
        if axis is not None:
            mass = mass * self.emission[axis, outcome]
        return mass @ self.transitions[tick]

    def forecast(self, mass, start, end=TICKS):
        for tick in range(start, end):
            mass = self.step(mass, tick)
        return mass


@dataclass
class Policy:
    # Axis index, or None when skipping. Terminal nodes have a posterior.
    axis: int | None
    children: tuple["Policy", ...] = ()
    posterior: np.ndarray | None = None
    value: float = 0.0


def terminal(mass):
    class_mass = mass @ CLASSES
    class_posterior = class_mass / class_mass.sum()
    return Policy(None, posterior=class_posterior, value=float(class_mass.max()))


def transition_node(model, mass, tick, axis, next_policy):
    if axis is None:
        children = (next_policy(model.step(mass, tick)),)
    else:
        children = tuple(next_policy(model.step(mass, tick, axis, symbol)) for symbol in range(3))
    return Policy(axis, children=children, value=sum(node.value for node in children))


def adaptive(model, mass, tick, left):
    if tick == TICKS:
        assert left == 0
        return terminal(mass)
    actions = []
    if left < TICKS - tick:
        actions.append(None)
    if left:
        actions.extend(range(len(AXES)))
    best = None
    for axis in actions:
        candidate = transition_node(
            model, mass, tick, axis,
            lambda next_mass: adaptive(model, next_mass, tick + 1, left - (axis is not None))
        )
        if best is None or candidate.value > best.value + TOL:
            best = candidate
    return best


def fixed(model, mass, time_axes, tick=0):
    if tick == TICKS:
        return terminal(mass)
    return transition_node(
        model, mass, tick, time_axes[tick],
        lambda next_mass: fixed(model, next_mass, time_axes, tick + 1)
    )


def best_schedule(model, budget, guides, required_ticks=None):
    best = None
    optimum = -1.0
    tick_options = itertools.combinations(range(TICKS), budget) if required_ticks is None else [required_ticks]
    for ticks in tick_options:
        for axes in itertools.product(range(len(AXES)), repeat=budget):
            schedule = tuple(dict(zip(ticks, axes)).get(tick) for tick in range(TICKS))
            candidate = tuple(fixed(model, model.initial(guide), schedule) for guide in guides)
            gain = sum(node.value for node in candidate)
            if gain > optimum + TOL:
                best = candidate
                optimum = gain
    return best


def single_read_utility(model, mass, tick, sensor, metric):
    scores = []
    for symbol in range(3):
        updated = model.step(mass, tick, sensor, symbol)
        classes = model.forecast(updated, tick + 1) @ CLASSES
        if metric == "risk":
            score = classes.max()
        else:
            score = np.sum(classes * np.log(classes / classes.sum()))
        scores.append(score)
    return sum(scores)


def myopic(model, mass, tick, left, metric):
    if tick == TICKS:
        assert left == 0
        return terminal(mass)
    axis = None
    if left:
        projected = mass
        best = -np.inf
        for next_tick in range(tick, TICKS - left + 1):
            for sensor in range(len(AXES)):
                value = single_read_utility(model, projected, next_tick, sensor, metric)
                if value > best + TOL:
                    best = value
                    axis = sensor if next_tick == tick else None
            projected = model.step(projected, next_tick)
    return transition_node(
        model, mass, tick, axis,
        lambda next_mass: myopic(model, next_mass, tick + 1, left - (axis is not None), metric)
    )


def make_policies(model, budget):
    periodic_ticks = (1,) if budget == 1 else (0, 2)
    return {
        "adaptive": tuple(adaptive(model, model.initial(g), 0, budget) for g in range(2)),
        "fixed_after_guide": tuple(best_schedule(model, budget, (g,))[0] for g in range(2)),
        "fixed_before_guide": best_schedule(model, budget, (0, 1)),
        "periodic": best_schedule(model, budget, (0, 1), periodic_ticks),
        "myopic_risk": tuple(myopic(model, model.initial(g), 0, budget, "risk") for g in range(2)),
        "myopic_entropy": tuple(myopic(model, model.initial(g), 0, budget, "entropy") for g in range(2)),
    }


def independently_score(model, roots, budget):
    # Traverse the frozen policy rather than reoptimising it under the scoring law.
    def visit(plan, mass, tick, reads):
        if plan.posterior is not None:
            assert tick == TICKS and reads == budget
            class_mass = mass @ CLASSES
            assert abs(float(plan.posterior.sum()) - 1) < TOL
            accuracy = class_mass[np.argmax(plan.posterior)]
            brier = class_mass.sum() * (1 + plan.posterior @ plan.posterior) - 2 * (class_mass @ plan.posterior)
            return np.array([class_mass.sum(), accuracy, brier, 0.0])
        if plan.axis is None:
            return visit(plan.children[0], model.step(mass, tick), tick + 1, reads)
        contributions = []
        for outcome, child in enumerate(plan.children):
            next_mass = model.step(mass, tick, plan.axis, outcome)
            contribution = visit(child, next_mass, tick + 1, reads + 1)
            if outcome == 2:
                contribution[-1] += next_mass.sum()
            contributions.append(contribution)
        return np.sum(contributions, axis=0)

    total = sum(visit(plan, model.initial(guide), 0, 0) for guide, plan in enumerate(roots))
    np.testing.assert_allclose(total[0], 1, atol=TOL, rtol=0)
    return {"accuracy": total[1], "brier": total[2], "mean_erased": total[3]}


@dataclass(frozen=True)
class Tape:
    guides: np.ndarray
    observations: np.ndarray
    truth: np.ndarray


def simulate(law, count, seed):
    rng = np.random.default_rng(seed)
    q = rng.integers(0, PHASES, count)
    high = rng.random(count) < 0.65
    guides = GUIDE_HORIZONTAL[q] ^ (rng.random(count) < law.guide_flip)
    observations = np.empty((TICKS, count, len(AXES)), dtype=np.int8)
    for tick in range(TICKS):
        quality = (~high).astype(int)
        coefficients = np.array(law.reliability)[quality]
        erasure = np.array(law.erasure)[quality]
        angle = PHASE_ANGLES[q, None] - PHASE_ANGLES[list(AXES)]
        p = 0.5 + 0.5 * coefficients[:, None] * np.cos(angle)
        positive = (rng.random(p.shape) < p).astype(np.int8)
        observations[tick] = np.where(rng.random(p.shape) < erasure[:, None], 2, positive)
        if tick == TICKS - 1 and law.reset:
            q = rng.integers(0, PHASES, count)
        else:
            q = (q + rng.choice(3, count, p=law.phase_jitter)) % PHASES
        high = rng.random(count) < np.where(high, law.high_to_high, law.low_to_high)
    return Tape(guides, observations, SECTORS[q])


def monte_carlo(roots, tape, budget):
    correct = np.zeros(len(tape.truth), dtype=bool)
    brier = np.zeros(len(tape.truth))
    erased = np.zeros(len(tape.truth), dtype=int)
    reads = np.zeros(len(tape.truth), dtype=int)
    target = np.eye(4)[tape.truth]

    def play(node, paths, tick):
        if node.posterior is not None:
            correct[paths] = np.argmax(node.posterior) == tape.truth[paths]
            brier[paths] = np.sum((target[paths] - node.posterior) ** 2, axis=1)
        elif node.axis is None:
            play(node.children[0], paths, tick + 1)
        else:
            obs = tape.observations[tick, paths, node.axis]
            reads[paths] += 1
            erased[paths] += obs == 2
            for symbol, child in enumerate(node.children):
                play(child, paths[obs == symbol], tick + 1)

    for guide in range(2):
        play(roots[guide], np.flatnonzero(tape.guides == guide), 0)
    assert (reads == budget).all()
    return {"accuracy": correct.mean(), "brier": brier.mean(), "mean_erased": erased.mean()}, correct


def paired_bootstrap(correct_adaptive, correct_fixed, seed):
    difference = correct_adaptive.astype(int) - correct_fixed
    count = len(difference)
    categories = np.bincount(difference + 1, minlength=3)
    distribution = np.random.default_rng(seed).multinomial(count, categories / count, size=4000)
    sampled_means = (distribution[:, 2] - distribution[:, 0]) / count
    return {
        "difference": difference.mean(),
        "paired_95_interval": np.quantile(sampled_means, [0.025, 0.975]).tolist(),
        "adaptive_only": int(categories[2]),
        "fixed_only": int(categories[0]),
    }


def exact_witness(error=0.2):
    # A single sensor is useful conditional on a revealed context. Count the
    # correct MAP mass for each context and binary reading.
    def evaluate(adaptive_choice):
        accuracy = 0
        for context in range(2):
            selected_sensor = context if adaptive_choice else 0
            for symbol in range(2):
                masses = [
                    0.25 * ((1 - error if symbol == truth else error) if selected_sensor == context else 0.5)
                    for truth in range(2)
                ]
                accuracy += max(masses)
        return accuracy

    fixed_value = evaluate(False)
    adaptive_value = evaluate(True)
    np.testing.assert_allclose([fixed_value, adaptive_value], [0.75 - error / 2, 1 - error], atol=TOL)
    return {"error": error, "fixed_before_guide": fixed_value, "adaptive": adaptive_value}


def evaluate():
    model = RotatingSignal(NOMINAL)
    two = make_policies(model, budget=2)
    one = make_policies(model, budget=1)
    known_risk = {
        "guide_only": independently_score(model, tuple(fixed(model, model.initial(g), (None,) * TICKS) for g in range(2)), 0),
        "one_read": {label: independently_score(model, policy, 1) for label, policy in one.items()},
        "three_reads": independently_score(model, tuple(adaptive(model, model.initial(g), 0, 3) for g in range(2)), 3),
    }
    np.testing.assert_allclose(known_risk["one_read"]["adaptive"]["accuracy"],
                               known_risk["one_read"]["fixed_after_guide"]["accuracy"], atol=TOL, rtol=0)
    family_reports = {}
    count = 32768
    for position, (family, law) in enumerate(FAMILIES.items()):
        scoring = RotatingSignal(law)
        tape = simulate(law, count, 544002 + position)
        result = {}
        correct = {}
        for label, roots in two.items():
            exact = independently_score(scoring, roots, 2)
            simulated, correct[label] = monte_carlo(roots, tape, 2)
            if family == "nominal":
                np.testing.assert_allclose(exact["accuracy"], sum(node.value for node in roots), atol=TOL, rtol=0)
            if law.reset:
                np.testing.assert_allclose(exact["accuracy"], 0.25, atol=TOL, rtol=0)
            bound = 5 * np.sqrt(exact["accuracy"] * (1 - exact["accuracy"]) / count)
            assert abs(exact["accuracy"] - simulated["accuracy"]) <= bound
            result[label] = {"exact": exact, "simulation": simulated}
        paired = paired_bootstrap(correct["adaptive"], correct["fixed_after_guide"], 8406 + position)
        paired["exact_difference"] = result["adaptive"]["exact"]["accuracy"] - result["fixed_after_guide"]["exact"]["accuracy"]
        family_reports[family] = {"policies": result, "adaptive_minus_fixed_after_guide": paired}

    primary = family_reports["nominal"]["adaptive_minus_fixed_after_guide"]
    primary_passed = primary["exact_difference"] >= 0.005 and primary["paired_95_interval"][0] > 0
    return {
        "protocol_sha256": hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest(),
        "simulated_windows_per_family": count,
        "primary_passed": bool(primary_passed),
        "budget_references": known_risk,
        "exact_context_witness": exact_witness(),
        "families": family_reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify", type=Path)
    args = parser.parse_args()
    results = json.dumps(evaluate(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(results)
    else:
        assert results == args.verify.read_text(), "Saved results differ from this replay"
        print("Exact-risk, shared-tape and invariant checks passed.")


if __name__ == "__main__":
    main()
