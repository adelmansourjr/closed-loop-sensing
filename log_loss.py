#!/usr/bin/env python3
"""Compare optimally class-preserving log-loss forecasts on a frozen signal tree."""

import argparse
from dataclasses import replace
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

from confidence import compress, count_reads, matched_brier_mixture, minimax, terminal_forecasts
from run import AXES, CLASSES, NOMINAL, TICKS, Policy, RotatingSignal, best_schedule, fixed, simulate

TOL = 1e-12
COUNT = 8192
CAPS = (1.25, 1.50)
METHODS = ("bayes", "uniform", "brier_minimax", "log_minimax")
FAMILIES = {
    "nominal": NOMINAL,
    "moderate_attenuation": replace(NOMINAL, reliability=(.69, .22), erasure=(.09, .29)),
    "moderate_guide_drift": replace(NOMINAL, guide_flip=.22),
    "faster_irregular_clock": replace(NOMINAL, phase_jitter=(.06, .61, .33)),
    "lower_quality_persistence": replace(
        NOMINAL, high_to_high=.90, low_to_high=.12, erasure=(.08, .44)
    ),
    "combined_shifts": replace(
        NOMINAL, phase_jitter=(.20, .52, .28), reliability=(.67, .24),
        erasure=(.12, .33), guide_flip=.23
    ),
    "very_noisy_guide_and_sensor": replace(
        NOMINAL, reliability=(.35, .09), erasure=(.20, .52), guide_flip=.43
    ),
    "polarity_and_guide_shift": replace(NOMINAL, reliability=(-.36, -.12), guide_flip=.32),
}


def minimax_log(descendants):
    maxima = np.max(descendants, axis=0)
    return maxima / maxima.sum()


def audit_minimax():
    binary = np.array([[.60, .40], [.90, .10]])
    expected = np.array([9 / 13, 4 / 13])
    np.testing.assert_allclose(minimax_log(binary), expected, atol=TOL, rtol=0)
    np.testing.assert_allclose(np.max(binary / expected), 1.3, atol=TOL, rtol=0)

    rng = np.random.default_rng(606728)
    for index in range(128):
        ordered = np.sort(rng.dirichlet(np.full(4, .4), 2 + index % 7), axis=1)[:, ::-1]
        descendants = ordered[:, rng.permutation(4)]
        p = minimax_log(descendants)
        maxima = descendants.max(axis=0)
        assert p.argmax() == descendants[0].argmax()
        np.testing.assert_allclose(np.max(descendants / p), maxima.sum(), atol=TOL, rtol=0)
        # Below the sharp ratio, the coordinate lower bounds require more than unit mass.
        assert np.sum(maxima / (maxima.sum() - 1e-6)) > 1
        for rival in rng.dirichlet(np.ones(4), size=8):
            assert np.max(descendants / rival) >= maxima.sum() - TOL
    tied = np.array([[.10, .45, .45, .00], [.05, .45, .45, .05]])
    assert minimax_log(tied).argmax() == 1
    return {"binary_sharp_ratio": 1.3, "random_stable_frontiers": 128, "rivals_per_frontier": 8}


def compile_policy(reference, bayes, method, cap=None, history=()):
    """Use the common Bayes stopping frontier and change only its forecasts."""
    if reference.posterior is not None:
        return reference, []
    if bayes.posterior is not None:
        descendants = np.asarray(terminal_forecasts(reference))
        if method == "bayes":
            posterior = bayes.posterior
        elif method == "uniform":
            posterior = descendants.mean(axis=0)
        elif method == "brier_minimax":
            posterior = minimax(descendants)[0]
        else:
            assert method == "log_minimax"
            posterior = minimax_log(descendants)
        assert (posterior > 0).all()
        ratios = descendants / posterior
        descendant_index, endpoint_class = np.unravel_index(ratios.argmax(), ratios.shape)
        ratio = float(ratios[descendant_index, endpoint_class])
        if cap is not None and ratio > cap + TOL:
            return reference, []
        return Policy(None, posterior=posterior), [{
            "history": history,
            "stopped_forecast": posterior.tolist(),
            "full_forecasts": descendants.tolist(),
            "sharp_ratio": ratio,
            "attaining_descendant": int(descendant_index),
            "attaining_endpoint_class": int(endpoint_class),
        }]

    assert reference.axis == bayes.axis
    children = []
    logs = []
    for symbol, pair in enumerate(zip(reference.children, bayes.children, strict=True)):
        step = None if reference.axis is None else (reference.axis, symbol)
        child, records = compile_policy(*pair, method, cap, (*history, step))
        children.append(child)
        logs.extend(records)
    return Policy(reference.axis, children=tuple(children)), logs


def audit_pair(reference, stopped):
    """Enumerate original observational paths independently of recorded certificates."""
    if stopped.posterior is not None:
        full = np.asarray(terminal_forecasts(reference))
        assert (full.argmax(axis=1) == stopped.posterior.argmax()).all()
        return count_reads(reference), float(np.max(full / stopped.posterior))
    assert stopped.axis == reference.axis and reference.posterior is None
    checks = [audit_pair(a, b) for a, b in zip(reference.children, stopped.children, strict=True)]
    return sum((reads for reads, _ in checks), ()), max(ratio for _, ratio in checks)


def exact_plan(model, node, state, tick=0, reads=0):
    if node.posterior is not None:
        terminal = model.forecast(state, tick) @ CLASSES
        mass = terminal.sum()
        return np.array([mass, reads * mass, terminal[node.posterior.argmax()],
                         -terminal @ np.log(node.posterior)])
    if node.axis is None:
        return exact_plan(model, node.children[0], model.step(state, tick), tick + 1, reads)
    return sum(
        exact_plan(model, child, model.step(state, tick, node.axis, symbol), tick + 1, reads + 1)
        for symbol, child in enumerate(node.children)
    )


def exact_mixture(model, roots, weights):
    totals = sum(
        w * exact_plan(model, root, model.initial(g))
        for g, group in enumerate(roots) for root, w in zip(group, weights[g], strict=True) if w
    )
    np.testing.assert_allclose(totals[0], 1, atol=TOL, rtol=0)
    return dict(zip(("mean_reads", "accuracy", "log_loss"), map(float, totals[1:]), strict=True))


def best_fixed_log(model, guide, count):
    candidates = []
    for ticks, axes in product(combinations(range(TICKS), count), product(range(len(AXES)), repeat=count)):
        schedule = dict(zip(ticks, axes, strict=True))
        initial = model.initial(guide)
        root = fixed(model, initial, tuple(schedule.get(tick) for tick in range(TICKS)))
        candidates.append((root, float(exact_plan(model, root, initial)[-1])))
    return min(candidates, key=lambda candidate: candidate[1])


def replay(tape, roots, weights, seed):
    probability = np.zeros(len(tape.truth))
    labels = np.full(len(tape.truth), -1)
    reads = np.zeros(len(tape.truth), dtype=int)

    def visit(node, indexes, tick):
        if node.posterior is not None:
            probability[indexes] = node.posterior[tape.truth[indexes]]
            labels[indexes] = node.posterior.argmax()
        elif node.axis is None:
            visit(node.children[0], indexes, tick + 1)
        else:
            symbols = tape.observations[tick, indexes, node.axis]
            reads[indexes] += 1
            for outcome, child in enumerate(node.children):
                visit(child, indexes[symbols == outcome], tick + 1)

    rng = np.random.default_rng(seed)
    for guide, group in enumerate(roots):
        indexes = np.flatnonzero(tape.guides == guide)
        choices = rng.choice(len(group), size=len(indexes), p=weights[guide])
        for branch, root in enumerate(group):
            visit(root, indexes[choices == branch], 0)
    assert (labels >= 0).all() and (probability > 0).all()
    return labels, -np.log(probability), reads


def sampled_summary(tape, scores, reference):
    labels, losses, reads = scores
    excess = losses - reference[1]
    mean = float(excess.mean())
    se = float(excess.std(ddof=1) / np.sqrt(len(excess)))
    return {
        "mean_reads": float(reads.mean()), "accuracy": float((labels == tape.truth).mean()),
        "log_loss": float(losses.mean()), "excess_log_loss": mean,
        "standard_error": se, "paired_95_pct_ci": [mean - 1.96 * se, mean + 1.96 * se],
        "max_realised_ratio": float(np.exp(excess.max())),
        "label_disagreements": int((labels != reference[0]).sum()),
    }


def evaluate():
    nominal = RotatingSignal(NOMINAL)
    reference = tuple(best_schedule(nominal, 2, (guide,))[0] for guide in range(2))
    baseline = tuple(compress(nominal, g, node, "coarse_bayes")[0] for g, node in enumerate(reference))
    unit = np.ones((2, 1))
    choices = {"full_reference": (tuple((root,) for root in reference), unit)}
    audits = {}
    for method, cap in product(METHODS, (None, *CAPS)):
        key = method if cap is None else f"{method}_ratio_{cap:.2f}"
        compiled = tuple(compile_policy(a, b, method, cap) for a, b in zip(reference, baseline, strict=True))
        roots = tuple((node,) for node, _ in compiled)
        choices[key] = (roots, unit)
        checks = [audit_pair(ref, paths[0]) for ref, paths in zip(reference, roots, strict=True)]
        saved = sum((r for r, _ in checks), ())
        worst = max(ratio for _, ratio in checks)
        if cap is not None:
            assert worst <= cap + TOL
        audits[key] = {
            "reference_paths": len(saved), "attempts_saved_on_each_reference_path": saved,
            "worst_pathwise_probability_ratio": worst, "worst_excess_log_loss": float(np.log(worst)),
            "stops_by_guide": [records for _, records in compiled],
        }
    common_paths = audits["bayes"]["attempts_saved_on_each_reference_path"]
    assert all(audits[key]["attempts_saved_on_each_reference_path"] == common_paths for key in METHODS)
    cost = exact_mixture(nominal, *choices["log_minimax"])["mean_reads"]
    candidates = [[best_fixed_log(nominal, g, count) for count in range(3)] for g in range(2)]
    # The existing mixture search optimises scalar costs. Here the candidate costs are log losses.
    mixture = matched_brier_mixture(nominal, candidates, cost)
    choices["matched_precommit"] = (tuple(tuple(p for p, _ in group) for group in candidates), mixture)
    np.testing.assert_allclose(exact_mixture(nominal, *choices["matched_precommit"])["mean_reads"], cost, atol=TOL, rtol=0)
    best = min(METHODS[:-1], key=lambda key: audits[key]["worst_excess_log_loss"])
    baseline_loss = audits[best]["worst_excess_log_loss"]
    minimax_loss = audits["log_minimax"]["worst_excess_log_loss"]
    improvement = (baseline_loss - minimax_loss) / baseline_loss

    results = {}
    for index, (name, law) in enumerate(FAMILIES.items()):
        model = RotatingSignal(law)
        tape = simulate(law, COUNT, 606821 + index)
        exact_base = exact_mixture(model, *choices["full_reference"])
        sampled_base = replay(tape, *choices["full_reference"], 606921 + index)
        results[name] = {}
        for key, (roots, weights) in choices.items():
            exact = exact_mixture(model, roots, weights)
            exact["excess_log_loss"] = exact["log_loss"] - exact_base["log_loss"]
            summary = sampled_summary(tape, replay(tape, roots, weights, 606921 + index), sampled_base)
            error = abs(summary["excess_log_loss"] - exact["excess_log_loss"])
            assert error <= 5 * summary["standard_error"] + TOL, (name, key)
            if key in audits:
                assert summary["label_disagreements"] == 0
                assert summary["max_realised_ratio"] <= audits[key]["worst_pathwise_probability_ratio"] + TOL
                np.testing.assert_allclose(exact["accuracy"], exact_base["accuracy"], atol=TOL, rtol=0)
            results[name][key] = {"exact": exact, "simulation": summary}

    return {
        "protocol_sha256": hashlib.sha256(Path(__file__).with_name("LOG_LOSS_PROTOCOL.md").read_bytes()).hexdigest(),
        "independent_paired_windows_per_family": COUNT,
        "implementation_audit": audit_minimax(),
        "primary": {
            "strongest_baseline": best, "baseline_worst_excess_log_loss": baseline_loss,
            "minimax_worst_excess_log_loss": minimax_loss, "relative_reduction": improvement,
            "passes": improvement >= .15,
        },
        "pathwise_audits": audits,
        "matched_precommit_dispatch_weights_by_guide_and_read_count": mixture.tolist(),
        "families": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    args = parser.parse_args()
    contents = json.dumps(evaluate(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(contents)
    elif contents != args.verify.read_text():
        raise SystemExit("Results did not reproduce")
    else:
        print(f"Verified {args.verify}")


if __name__ == "__main__":
    main()
