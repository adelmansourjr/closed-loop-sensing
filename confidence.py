#!/usr/bin/env python3
"""Certify the worst Brier excess from stopping a frozen passive sampling tree."""

import argparse
from dataclasses import replace
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from pruning import exact_subtree
from run import AXES, NOMINAL, TICKS, Policy, RotatingSignal, best_schedule, fixed, simulate, terminal

TOL = 1e-12
COUNT = 32768
CAPS = (0.10, 0.20, 0.30)
FAMILIES = {
    "nominal": NOMINAL,
    "attenuation_and_noise": replace(NOMINAL, reliability=(0.58, 0.09), erasure=(0.18, 0.33)),
    "drifting_clock": replace(NOMINAL, phase_jitter=(0.33, 0.47, 0.20), guide_flip=0.27),
    "bursty_erasure": replace(NOMINAL, high_to_high=0.97, low_to_high=0.08, erasure=(0.18, 0.70)),
    "reversed_projection": replace(NOMINAL, reliability=(-0.63, -0.22), guide_flip=0.18),
    "independent_endpoint": replace(NOMINAL, reset=True),
}
FORECASTERS = ("coarse_bayes", "uniform_descendants", "minimax")


def project_simplex(x):
    ordered = np.sort(x)[::-1]
    thresholds = (np.cumsum(ordered) - 1) / np.arange(1, len(x) + 1)
    active = np.flatnonzero(ordered > thresholds)[-1]
    return np.maximum(x - thresholds[active], 0)


def brier_differences(p, q):
    """Broadcast the replacement-minus-full loss over all descendant classes."""
    return (p @ p - np.sum(q**2, axis=1))[:, None] - 2 * (p - q)


def minimax(q):
    coordinate_bounds = np.max(2 * q - np.sum(q**2, axis=1)[:, None], axis=0)
    forecast = project_simplex(coordinate_bounds / 2)
    bound = forecast @ forecast + np.max(coordinate_bounds - 2 * forecast)
    table = brier_differences(forecast, q)
    np.testing.assert_allclose(bound, table.max(), atol=TOL, rtol=0)
    np.testing.assert_allclose(forecast.sum(), 1, atol=TOL, rtol=0)
    return forecast, float(bound)


def audit_projection():
    # Work with exact rational arithmetic in a separate binary check.
    q_binary = (Fraction(3, 5), Fraction(9, 10))

    def binary_excess(p):
        return max(2 * ((p - y)**2 - (q - y)**2) for y, q in product(range(2), q_binary))

    grid = [Fraction(k, 400) for k in range(401)]
    best = min(grid, key=binary_excess)
    assert best == Fraction(27, 40)
    assert binary_excess(best) == Fraction(153, 800)
    forecast, bound = minimax(np.array([[float(q), 1 - float(q)] for q in q_binary]))
    np.testing.assert_allclose([forecast[0], bound], [float(best), float(binary_excess(best))], atol=TOL, rtol=0)
    first, second = float(binary_excess(best)), float(binary_excess(Fraction(3, 4)))
    assert first <= 0.20 < second

    rng = np.random.default_rng(915186)
    largest_slack_violation = 0.0
    for i in range(512):
        draws = rng.dirichlet(np.ones(4) * (0.3 if i % 2 else 3), 1 + i % 9)
        # Preserve within-row confidence variation while rotating the shared winning class.
        shapes = np.sort(draws, axis=1)[:, ::-1]
        q = shapes[:, rng.permutation(4)]
        p, achieved = minimax(q)
        assert np.unique(np.argmax(q, axis=1)).size == 1 and p.argmax() == q[0].argmax()
        for competitor in rng.dirichlet(np.ones(4), 16):
            slack = brier_differences(competitor, q).max() - achieved - ((competitor - p)**2).sum()
            largest_slack_violation = max(largest_slack_violation, -float(slack))
            assert slack >= -TOL
    tied = np.array([[.1, .45, .45, 0], [.05, .45, .45, .05]])
    assert minimax(tied)[0].argmax() == 1
    return {
        "binary_minimax_replacement": str(best),
        "binary_minimax_bound": str(binary_excess(best)),
        "binary_uniform_bound": str(binary_excess(Fraction(3, 4))),
        "random_four_class_trees": 512,
        "strong_convexity_slack_violation": largest_slack_violation,
    }


def terminal_forecasts(plan):
    if plan.posterior is not None:
        return [plan.posterior]
    return [q for child in plan.children for q in terminal_forecasts(child)]


def retains_query(plan):
    return plan.axis is not None or any(retains_query(child) for child in plan.children)


def compress(model, guide, plan, rule, cap=None):
    certificates = []

    def visit(node, mass, tick, history):
        if node.posterior is not None:
            return node
        q = np.asarray(terminal_forecasts(node))
        has_one_class = np.unique(q.argmax(axis=1)).size == 1
        if has_one_class and retains_query(node):
            if rule == "coarse_bayes":
                p = terminal(model.forecast(mass, tick)).posterior
            elif rule == "uniform_descendants":
                p = q.mean(axis=0)
            else:
                p, _ = minimax(q)
            assert p.argmax() == q[0].argmax()
            deltas = brier_differences(p, q)
            worst = float(np.max(deltas))
            if cap is None or worst <= cap + TOL:
                idx, y = np.unravel_index(deltas.argmax(), deltas.shape)
                # The attaining label and descendant are legal under the nominal model.
                certificates.append({
                    "guide": guide, "history": history, "nominal_prefix_probability": float(mass.sum()),
                    "stopped": p.tolist(), "full_forecasts": q.tolist(), "class": int(p.argmax()),
                    "worst_excess": worst, "attaining_descendant_index": int(idx), "attaining_class": int(y),
                })
                return Policy(None, posterior=p)
        if node.axis is None:
            child = visit(node.children[0], model.step(mass, tick), tick + 1, (*history, None))
            return Policy(None, children=(child,))
        children = tuple(
            visit(child, model.step(mass, tick, node.axis, outcome), tick + 1, (*history, (node.axis, outcome)))
            for outcome, child in enumerate(node.children)
        )
        return Policy(node.axis, children=children)

    policy = visit(plan, model.initial(guide), 0, ())
    return policy, certificates


def audit_pair(reference, compressed, reads_reference=0, reads_compressed=0, cap=None):
    """Independently enumerate frozen paths and every possible realised outcome class."""
    if compressed.posterior is not None:
        q = np.asarray(terminal_forecasts(reference))
        assert (q.argmax(axis=1) == compressed.posterior.argmax()).all()
        excess = brier_differences(compressed.posterior, q)
        if cap is not None:
            assert np.max(excess) <= cap + TOL
        extra = count_reads(reference)
        return len(q), float(np.max(excess)), tuple(reads_reference + n - reads_compressed for n in extra)
    assert reference.posterior is None and reference.axis == compressed.axis
    r = int(reference.axis is not None)
    children = [
        audit_pair(a, b, reads_reference + r, reads_compressed + r, cap)
        for a, b in zip(reference.children, compressed.children, strict=True)
    ]
    return sum(x[0] for x in children), max(x[1] for x in children), sum((x[2] for x in children), ())


def count_reads(root):
    if root.posterior is not None:
        return (0,)
    return tuple(int(root.axis is not None) + r for child in root.children for r in count_reads(child))


def best_brier_precommitted(model, guide, budget):
    initial = model.initial(guide)
    best = None
    best_loss = np.inf
    for ticks, axes in product(combinations(range(TICKS), budget), product(range(len(AXES)), repeat=budget)):
        by_tick = dict(zip(ticks, axes))
        plan = fixed(model, initial, tuple(by_tick.get(tick) for tick in range(TICKS)))
        loss = exact_subtree(model, plan, initial)[3]
        if loss < best_loss - TOL:
            best, best_loss = plan, loss
    return best, float(best_loss)


def matched_brier_mixture(model, plans, target):
    probabilities = np.array([model.initial(g).sum() for g in range(2)])
    losses = np.array([[loss for _, loss in group] for group in plans])
    best_weights = None
    best_loss = np.inf
    for flexible, fixed_count, (lo, hi) in product(range(2), range(3), combinations(range(3), 2)):
        other = 1 - flexible
        flexible_count = (target - probabilities[other] * fixed_count) / probabilities[flexible]
        if lo - TOL <= flexible_count <= hi + TOL:
            t = float(np.clip((flexible_count - lo) / (hi - lo), 0, 1))
            weights = np.zeros((2, 3))
            weights[other, fixed_count] = 1
            weights[flexible, lo] = 1 - t
            weights[flexible, hi] = t
            cost = np.sum(weights * probabilities[:, None] * np.arange(3))
            np.testing.assert_allclose(cost, target, atol=TOL, rtol=0)
            loss = float(np.sum(weights * losses))
            if loss < best_loss - TOL:
                best_weights = weights
                best_loss = loss
    assert best_weights is not None
    return best_weights


def exact(model, paths, dispatch_weights):
    totals = sum(
        weight * exact_subtree(model, paths[g][i], model.initial(g))
        for g, weights in enumerate(dispatch_weights) for i, weight in enumerate(weights) if weight
    )
    np.testing.assert_allclose(totals[0], 1.0, atol=TOL, rtol=0)
    return dict(zip(("mean_reads", "accuracy", "brier"), map(float, totals[1:]), strict=True))


def replay_tape(tape, roots, weights, seed):
    probabilities = np.zeros((len(tape.truth), 4))
    reads = np.zeros(len(tape.truth), dtype=int)

    def visit(node, group, tick):
        if node.posterior is not None:
            probabilities[group] = node.posterior
        elif node.axis is None:
            visit(node.children[0], group, tick + 1)
        else:
            symbols = tape.observations[tick, group, node.axis]
            reads[group] += 1
            for outcome, child in enumerate(node.children):
                visit(child, group[symbols == outcome], tick + 1)

    rng = np.random.default_rng(seed)
    for g in range(2):
        group = np.flatnonzero(tape.guides == g)
        chosen = rng.choice(len(weights[g]), len(group), p=weights[g])
        for i, node in enumerate(roots[g]):
            visit(node, group[chosen == i], 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=TOL, rtol=0)
    labels = probabilities.argmax(axis=1)
    loss = np.sum((np.eye(4)[tape.truth] - probabilities)**2, axis=1)
    return labels, loss, reads


def sampled_scores(labels, loss, reads, tape, reference):
    difference = loss - reference[1]
    estimate = difference.mean()
    error = difference.std(ddof=1) / np.sqrt(len(tape.truth))
    return {
        "mean_reads": float(reads.mean()), "accuracy": float((labels == tape.truth).mean()),
        "brier": float(loss.mean()), "brier_excess": float(estimate),
        "brier_excess_ci_95": [float(estimate - 1.96 * error), float(estimate + 1.96 * error)],
        "brier_excess_standard_error": float(error), "largest_realised_excess": float(np.max(difference)),
        "label_disagreements": int((labels != reference[0]).sum()),
    }


def evaluate():
    nominal = RotatingSignal(NOMINAL)
    reference = tuple(best_schedule(nominal, 2, (g,))[0] for g in range(2))
    unit = np.ones((2, 1))
    choices = {"reference": (tuple((root,) for root in reference), unit)}
    certificates = {}
    audits = {}
    for rule in FORECASTERS:
        for cap in (None, *CAPS):
            key = rule if cap is None else f"{rule}_cap_{cap:.2f}"
            compiled = [compress(nominal, guide, root, rule, cap) for guide, root in enumerate(reference)]
            choices[key] = (tuple((p,) for p, _ in compiled), unit)
            certificates[key] = sum((log for _, log in compiled), [])
            exact_audit = [audit_pair(a, b, cap=cap) for a, (b, _) in zip(reference, compiled, strict=True)]
            savings = sum((sav for _, _, sav in exact_audit), ())
            audits[key] = {
                "enumerated_reference_paths": sum(n for n, _, _ in exact_audit),
                "max_pathwise_excess": max(bound for _, bound, _ in exact_audit),
                "per_reference_path_saved": savings,
            }
    old = audits["coarse_bayes"]["max_pathwise_excess"]
    new = audits["minimax"]["max_pathwise_excess"]
    assert audits["minimax"]["per_reference_path_saved"] == audits["coarse_bayes"]["per_reference_path_saved"]
    target = exact(nominal, *choices["minimax"])["mean_reads"]
    fixed_candidates = [
        [best_brier_precommitted(nominal, guide, n) for n in range(3)] for guide in range(2)
    ]
    weights = matched_brier_mixture(nominal, fixed_candidates, target)
    choices["matched_precommit"] = (tuple(tuple(p for p, _ in group) for group in fixed_candidates), weights)
    np.testing.assert_allclose(exact(nominal, *choices["matched_precommit"])["mean_reads"], target, atol=TOL, rtol=0)

    scored = {}
    for family_index, (name, law) in enumerate(FAMILIES.items()):
        model = RotatingSignal(law)
        tape = simulate(law, COUNT, 913004 + family_index)
        reference_tape = replay_tape(tape, *choices["reference"], 915717 + family_index)
        reference_exact = exact(model, *choices["reference"])
        family = {}
        for key, (paths, dispatch) in choices.items():
            run = replay_tape(tape, paths, dispatch, 915717 + family_index)
            simulated = sampled_scores(*run, tape, reference_tape)
            scores = exact(model, paths, dispatch)
            brier_diff = scores["brier"] - reference_exact["brier"]
            scores["brier_excess"] = float(brier_diff)
            difference = abs(simulated["brier_excess"] - brier_diff)
            assert difference <= 5 * simulated["brier_excess_standard_error"] + TOL
            if key != "matched_precommit":
                assert simulated["label_disagreements"] == 0
                np.testing.assert_allclose(scores["accuracy"], reference_exact["accuracy"], atol=TOL, rtol=0)
            if key in audits:
                assert simulated["largest_realised_excess"] <= audits[key]["max_pathwise_excess"] + TOL
            family[key] = {"exact": scores, "simulation": simulated}
        scored[name] = family

    return {
        "protocol_sha256": hashlib.sha256(Path(__file__).with_name("CONFIDENCE_PROTOCOL.md").read_bytes()).hexdigest(),
        "audit": audit_projection(),
        "primary": {
            "bayes_worst_case": old,
            "minimax_worst_case": new,
            "relative_reduction": (old - new) / old,
            "passes": bool((old - new) / old >= 0.15),
            "nominal_matched_reads": target,
        },
        "matching_frontier_and_cap_audits": audits,
        "accepted_stop_certificates": certificates,
        "precommitted_weights_by_guide_and_count": weights.tolist(),
        "independent_paired_windows_per_family": COUNT,
        "families": scored,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()
    results = json.dumps(evaluate(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(results)
    else:
        assert results == args.verify.read_text(), "Confidence replay differs"
        print("Confidence replay matched; projection proof fixtures, frontier parity, caps, and tape checks passed.")


if __name__ == "__main__":
    main()
