#!/usr/bin/env python3
"""Certify redundant reads in a frozen rotating-signal classification policy."""

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from run import (
    CLASSES, FAMILIES, NOMINAL, TICKS, TOL, Policy, RotatingSignal,
    Tape, best_schedule, simulate, terminal,
)


def labels(plan):
    if plan.posterior is not None:
        return {int(np.argmax(plan.posterior))}
    return set().union(*(labels(child) for child in plan.children))


def prune(model, plan, mass, tick=0):
    possible_labels = labels(plan)
    if len(possible_labels) == 1:
        pruned = terminal(model.forecast(mass, tick))
        assert labels(pruned) == possible_labels
        return pruned
    if plan.axis is None:
        next_mass = model.step(mass, tick)
        children = (prune(model, plan.children[0], next_mass, tick + 1),)
    else:
        children = tuple(
            prune(model, child, model.step(mass, tick, plan.axis, outcome), tick + 1)
            for outcome, child in enumerate(plan.children)
        )
    return Policy(plan.axis, children=children)


def prove_all_paths(reference, pruned):
    """Check that every leaf of the pruned tree agrees with its reference subtree."""
    if pruned.posterior is not None:
        assert labels(pruned) == labels(reference)
        return 1
    assert reference.axis == pruned.axis
    return sum(prove_all_paths(a, b) for a, b in zip(reference.children, pruned.children, strict=True))


def exact_subtree(model, plan, mass, tick=0, reads=0):
    if plan.posterior is not None:
        assert reads <= 2
        class_mass = model.forecast(mass, tick) @ CLASSES
        total = class_mass.sum()
        brier = total * (1 + plan.posterior @ plan.posterior) - 2 * (class_mass @ plan.posterior)
        return np.array([total, total * reads, class_mass[np.argmax(plan.posterior)], brier])
    if plan.axis is None:
        return exact_subtree(model, plan.children[0], model.step(mass, tick), tick + 1, reads)
    return sum(
        exact_subtree(model, child, model.step(mass, tick, plan.axis, outcome), tick + 1, reads + 1)
        for outcome, child in enumerate(plan.children)
    )


def exact_mixture(model, plans, weights):
    totals = sum(
        weight * exact_subtree(model, plans[guide][k], model.initial(guide))
        for guide, mixture in enumerate(weights) for k, weight in enumerate(mixture) if weight > 0
    )
    np.testing.assert_allclose(totals[0], 1, rtol=0, atol=TOL)
    return totals[1:]


def mix_equal_budget(model, exact_schedules, target):
    probabilities = np.array([model.initial(guide).sum() for guide in range(2)])
    acc_mass = np.array([[p.value for p in plans] for plans in exact_schedules])
    best_weights = None
    best_accuracy = -1
    for flexible, fixed_budget, (lo, hi) in itertools.product(range(2), range(3), itertools.combinations(range(3), 2)):
        other = 1 - flexible
        flexible_budget = (target - probabilities[other] * fixed_budget) / probabilities[flexible]
        if lo - TOL <= flexible_budget <= hi + TOL:
            fraction = float(np.clip((flexible_budget - lo) / (hi - lo), 0, 1))
            weight = np.zeros((2, 3))
            weight[other, fixed_budget] = 1
            weight[flexible, lo] = 1 - fraction
            weight[flexible, hi] = fraction
            expected_budget = np.sum(weight * probabilities[:, None] * np.arange(3))
            np.testing.assert_allclose(expected_budget, target, rtol=0, atol=TOL)
            accuracy = np.sum(weight * acc_mass)
            if accuracy > best_accuracy + TOL:
                best_accuracy = accuracy
                best_weights = weight
    assert best_weights is not None
    return best_weights


def play_mixture(tape, plans, weights, seed):
    predictions = np.empty(len(tape.truth), dtype=int)
    probabilities = np.empty((len(tape.truth), 4))
    reads = np.zeros(len(tape.truth), dtype=int)

    def visit(plan, paths, tick):
        if plan.posterior is not None:
            predictions[paths] = np.argmax(plan.posterior)
            probabilities[paths] = plan.posterior
            return
        if plan.axis is None:
            visit(plan.children[0], paths, tick + 1)
            return
        symbols = tape.observations[tick, paths, plan.axis]
        reads[paths] += 1
        for outcome, child in enumerate(plan.children):
            visit(child, paths[symbols == outcome], tick + 1)

    rng = np.random.default_rng(seed)
    for guide in range(2):
        group = np.flatnonzero(tape.guides == guide)
        dispatched = rng.choice(len(weights[guide]), len(group), p=weights[guide])
        for k, weight in enumerate(weights[guide]):
            if weight > 0:
                visit(plans[guide][k], group[dispatched == k], 0)
    assert np.max(reads) <= 2
    brier = np.sum((np.eye(4)[tape.truth] - probabilities) ** 2, axis=1).mean()
    return np.array([reads.mean(), (predictions == tape.truth).mean(), brier]), predictions


def mixture_tape(components, count, seed):
    if len(components) == 1:
        return simulate(components[0][1], count, seed)
    (p, ordinary), (_, inverse) = components
    first = simulate(ordinary, count, seed)
    second = simulate(inverse, count, seed + 100000)
    selector = np.random.default_rng(seed + 200000).random(count) < p
    return Tape(np.where(selector, first.guides, second.guides),
                np.where(selector[None, :, None], first.observations, second.observations),
                np.where(selector, first.truth, second.truth))


def result_dict(scores):
    return dict(zip(("mean_reads", "accuracy", "brier"), map(float, scores), strict=True))


def evaluate():
    nominal = RotatingSignal(NOMINAL)
    exact_schedules = tuple(tuple(best_schedule(nominal, k, (guide,))[0] for k in range(3)) for guide in range(2))
    reference = tuple((plans[2],) for plans in exact_schedules)
    certified = tuple((prune(nominal, original[0], nominal.initial(g)),) for g, original in enumerate(reference))
    leaf_count = sum(prove_all_paths(a[0], b[0]) for a, b in zip(reference, certified, strict=True))
    unit_weight = np.ones((2, 1))
    target = exact_mixture(nominal, certified, unit_weight)[0]
    matched_weights = mix_equal_budget(nominal, exact_schedules, target)
    methods = {
        "fixed_two_reads": (reference, unit_weight),
        "certified_pruning": (certified, unit_weight),
        "best_fixed_matching_mean_budget": (exact_schedules, matched_weights),
    }

    families = {name: ((1, law),) for name, law in FAMILIES.items()}
    families["severe_erasure"] = ((1, replace(NOMINAL, erasure=(0.48, 0.72))),)
    families["polarity_mixture"] = (
        (0.65, NOMINAL), (0.35, replace(NOMINAL, reliability=(-0.82, -0.32)))
    )
    count = 32768
    report = {}
    for position, (family, components) in enumerate(families.items()):
        seed = 764916 + position
        tape = mixture_tape(components, count, seed)
        score_models = [(p, RotatingSignal(law)) for p, law in components]
        exact = {}
        sampled = {}
        predictions = {}
        for name, (plans, weights) in methods.items():
            exact[name] = sum(p * exact_mixture(model, plans, weights) for p, model in score_models)
            sampled[name], predictions[name] = play_mixture(tape, plans, weights, seed + 300000)
            exact_accuracy = exact[name][1]
            bound = 5 * np.sqrt(exact_accuracy * (1 - exact_accuracy) / count)
            assert abs(sampled[name][1] - exact_accuracy) < bound
        class_disagreements = int(np.sum(predictions["fixed_two_reads"] != predictions["certified_pruning"]))
        assert class_disagreements == 0
        np.testing.assert_allclose(exact["fixed_two_reads"][1], exact["certified_pruning"][1], rtol=0, atol=TOL)
        report[family] = {
            "exact": {name: result_dict(values) for name, values in exact.items()},
            "simulation": {name: result_dict(values) for name, values in sampled.items()},
            "class_disagreements": class_disagreements,
            "read_saving_fraction": float((2 - exact["certified_pruning"][0]) / 2),
        }

    nominal_report = report["nominal"]
    np.testing.assert_allclose(nominal_report["exact"]["best_fixed_matching_mean_budget"]["mean_reads"],
                               target, atol=TOL, rtol=0)
    return {
        "protocol_sha256": hashlib.sha256(Path(__file__).with_name("PRUNING_PROTOCOL.md").read_bytes()).hexdigest(),
        "original_protocol_sha256": hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest(),
        "exact_certificate_leaf_count": leaf_count,
        "nominal_matched_mean_mixture_probabilities": matched_weights.tolist(),
        "primary_passed": bool(nominal_report["read_saving_fraction"] >= 0.10),
        "simulated_windows_per_family": count,
        "families": report,
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
        assert results == args.verify.read_text(), "Saved pruning results differ from this replay"
        print("All decision-invariance, equal-mean budget and shared-tape checks passed.")


if __name__ == "__main__":
    main()
