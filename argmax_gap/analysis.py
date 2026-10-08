"""Recompute paper metrics and strata from aligned legal-move distributions."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .metrics import summarize, ece, paired_top1, paired_mean_ci, clustered_ci


def load_predictions(path, batch_size=4096):
    """Validate distributions and recompute ranks, without loading ragged arrays at once."""
    parts = []
    columns = ["row_id", "human_move_uci", "legal_moves_uci", "legal_probs"]
    parquet = pq.ParquetFile(path)
    if "game_id" in parquet.schema_arrow.names:
        columns.append("game_id")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        rows = batch.to_pydict()
        lengths = np.array([len(m) for m in rows["legal_moves_uci"]])
        if (lengths == 0).any():
            raise ValueError(f"Empty legal move list in {path}")
        p = np.zeros((len(lengths), lengths.max()), dtype=np.float64)
        human = np.empty(len(lengths), dtype=int)
        for i, (moves, probs, target) in enumerate(zip(rows["legal_moves_uci"], rows["legal_probs"], rows["human_move_uci"], strict=True)):
            if len(moves) != len(probs) or len(set(moves)) != len(moves) or target not in moves:
                raise ValueError(f"Invalid move alignment at row {rows['row_id'][i]}")
            p[i, :len(probs)] = probs
            human[i] = moves.index(target)
        totals = p.sum(axis=1)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(totals, 1, atol=2e-5, rtol=0):
            raise ValueError(f"Probabilities must be finite, nonnegative, and normalized: {path}")
        p /= totals[:, None]
        ph = p[np.arange(len(p)), human]
        rank = 1 + (p > ph[:, None]).sum(axis=1)
        top = p.argmax(axis=1)
        sorted_p = np.sort(p, axis=1)
        second = sorted_p[:, -2] if p.shape[1] > 1 else np.zeros(len(p))
        top_p = p[np.arange(len(p)), top]
        frame = pd.DataFrame({"row_id": rows['row_id'], "human_move_uci": rows['human_move_uci'],
            "human_rank": rank, "p_human": ph, "p_top1": top_p,
            "top1_move_uci": [moves[t] for moves, t in zip(rows['legal_moves_uci'], top, strict=True)],
            "nll": -np.log(np.maximum(ph, 1e-45)), "entropy": -(p * np.log(np.maximum(p, 1e-45))).sum(axis=1),
            "margin": top_p - second, "num_legal_moves": lengths})
        if "game_id" in rows:
            frame['game_id'] = rows['game_id']
        parts.append(frame)
    if not parts:
        raise ValueError(f"No predictions in {path}")
    result = pd.concat(parts, ignore_index=True)
    if result.row_id.duplicated().any():
        raise ValueError(f"Duplicate row IDs in {path}")
    return result


def check_alignment(base, other):
    if len(base) != len(other) or not np.array_equal(base.row_id, other.row_id):
        raise ValueError("Prediction row IDs or order differ")
    for key in ("human_move_uci", "game_id"):
        if key in base and key in other and not np.array_equal(base[key], other[key]):
            raise ValueError(f"Prediction {key} differs between inputs")


def strata(positions, maia):
    cut = lambda x, edges, labels: pd.cut(x, [-np.inf, *edges, np.inf], labels=labels, right=True)
    t = positions.time_spent_seconds.to_numpy()
    entropy_bins = pd.qcut(maia.entropy, 5, duplicates='drop')
    entropy_labels = entropy_bins.cat.rename_categories([f'Q{i+1}' for i in range(len(entropy_bins.cat.categories))])
    result = {
        "move_time": cut(t, [1, 2, 4, 7], ["[0,1]", "(1,2]", "(2,4]", "(4,7]", "(7,inf)"]),
        "fixed_seconds": cut(t, [2, 5, 10, 20], ["[0,2]", "(2,5]", "(5,10]", "(10,20]", "(20,inf)"]),
        "phase": cut(positions.move_number, [10, 40], ["opening", "middlegame", "endgame"]),
        "legal_moves": cut(maia.num_legal_moves, [20, 30, 40, 60], ["1-20", "21-30", "31-40", "41-60", ">60"]),
        "rating": pd.cut(positions.player_elo, np.arange(600, 3400, 200), right=False),
        "entropy": entropy_labels,
        "margin": cut(maia.margin, [.03, .10, .25], ["<=0.03", "(0.03,0.10]", "(0.10,0.25]", ">0.25"]),
    }
    if "clock_before_seconds" in positions:
        clock = positions.clock_before_seconds.to_numpy()
        if (clock <= 0).any():
            raise ValueError("Clock context requires positive pre-move clocks")
        result['remaining_clock'] = cut(clock, [30, 60, 120, 180], ["<=30", "(30,60]", "(60,120]", "(120,180]", ">180"])
        result['clock_fraction'] = cut(t / clock, [.01, .03, .07, .15], ["<=1%", "(1,3]%", "(3,7]%", "(7,15]%", ">15%"])
    if "increment_seconds" in positions:
        result['increment'] = np.where(positions.increment_seconds > 0, "increment", "no increment")
    return result


def report(positions_path, maia_path, allie_path, output, methods=None, expected_rows=884049, bootstrap_reps=10000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    position_columns = ['game_id', 'human_move_uci', 'time_spent_seconds', 'move_number',
                        'player_elo', 'clock_before_seconds', 'increment_seconds']
    schema = pq.ParquetFile(positions_path).schema_arrow.names
    positions = pq.read_table(positions_path, columns=[c for c in position_columns if c in schema]).to_pandas()
    maia, allie = load_predictions(maia_path), load_predictions(allie_path)
    check_alignment(maia, allie)
    if len(positions) != len(maia) or not np.array_equal(maia.row_id, np.arange(len(positions))):
        raise ValueError("Position table must align with contiguous prediction row IDs")
    if not np.array_equal(positions.human_move_uci, maia.human_move_uci):
        raise ValueError("Position targets differ from model targets")
    if "game_id" in maia and not np.array_equal(positions.game_id.astype(str), maia.game_id.astype(str)):
        raise ValueError("Position games differ from model games")
    if expected_rows is not None and len(maia) != expected_rows:
        raise ValueError(f"Expected {expected_rows} positions, got {len(maia)}")
    base_correct = maia.human_rank.to_numpy() == 1
    allie_correct = allie.human_rank.to_numpy() == 1
    disagreement = maia.top1_move_uci.to_numpy() != allie.top1_move_uci.to_numpy()
    models = {"maia3": maia, "allie": allie}
    if methods is not None:
        for path in sorted((Path(methods) / 'distributions').glob('*.parquet')):
            frame = load_predictions(path)
            check_alignment(maia, frame)
            models[path.stem] = frame
    rows = []
    for name, frame in models.items():
        rows.append({"method": name, **summarize(frame),
            "disagreement_ECE": ece(frame.p_top1.to_numpy()[disagreement], frame.human_rank.to_numpy()[disagreement] == 1)})
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    overlap = {"both_correct": int((base_correct & allie_correct).sum()),
        "maia_only": int((base_correct & ~allie_correct).sum()),
        "allie_only": int((~base_correct & allie_correct).sum()),
        "both_wrong": int((~base_correct & ~allie_correct).sum()),
        "oracle_Top1": float(100 * (base_correct | allie_correct).mean())}
    (output / 'complementarity.json').write_text(json.dumps(overlap, indent=2) + '\n')
    pairs = [{"method": 'allie', **paired_top1(base_correct, allie_correct)},
             {"method": 'diagnostic_oracle', **paired_top1(base_correct, base_correct | allie_correct)}]
    fixed_predictions = {}
    for name, use_allie in {
        'max_probability': allie.p_top1.to_numpy() > maia.p_top1.to_numpy(),
        'min_entropy': allie.entropy.to_numpy() < maia.entropy.to_numpy(),
        'max_margin': allie.margin.to_numpy() > maia.margin.to_numpy(),
    }.items():
        correctness = np.where(use_allie, allie_correct, base_correct)
        fixed_predictions[name] = correctness
        pairs.append({"method": name, "Top1": 100 * correctness.mean(), **paired_top1(base_correct, correctness)})
    changes, rank_changes = [], []
    for name, frame in models.items():
        if name in ('maia3', 'allie'):
            continue
        reference_name = 'allie' if name.startswith('allie') else 'maia3'
        reference = models[reference_name]
        correct = frame.human_rank.to_numpy() == 1
        delta = correct.astype(float) - (reference.human_rank.to_numpy() == 1)
        pair = {"method": name, "reference": reference_name, **paired_top1(reference.human_rank == 1, correct)}
        if bootstrap_reps:
            pair.update({k + '_pp': 100 * v for k, v in clustered_ci(delta, positions.game_id, reps=bootstrap_reps).items()})
        pairs.append(pair)
        rank_changes.append({'method': name, 'reference': reference_name,
            'rank_improved': int((frame.human_rank < reference.human_rank).sum()),
            'rank_worsened': int((frame.human_rank > reference.human_rank).sum()),
            'rank_unchanged': int((frame.human_rank == reference.human_rank).sum()),
            'top1_in': pair['rescues'], 'top1_out': pair['breaks']})
        for metric, difference in {
            'NLL': frame.nll.to_numpy() - reference.nll.to_numpy(),
            'MRR': 1 / frame.human_rank.to_numpy() - 1 / reference.human_rank.to_numpy(),
            'NDCG@5': np.where(frame.human_rank <= 5, 1 / np.log2(frame.human_rank + 1), 0) - np.where(reference.human_rank <= 5, 1 / np.log2(reference.human_rank + 1), 0),
        }.items():
            row = {"method": name, "reference": reference_name, "metric": metric, **paired_mean_ci(difference)}
            if bootstrap_reps:
                row.update(clustered_ci(difference, positions.game_id, reps=bootstrap_reps))
            changes.append(row)
    if methods is not None:
        for path in sorted((Path(methods) / 'predictions').glob('*.parquet')):
            frame = pq.read_table(path).to_pandas()
            check_alignment(maia, frame)
            reference_name = 'allie' if path.stem.startswith('allie') else 'maia3'
            reference = models[reference_name]
            correct = frame.top1_move_uci.to_numpy() == positions.human_move_uci.to_numpy()
            if 'switch' in frame:
                switch = frame['switch'].to_numpy(dtype=bool)
                correct = np.where(switch, correct, reference.human_rank.to_numpy() == 1)
            if 'correct_top1' in frame and not np.array_equal(correct, frame.correct_top1.to_numpy(dtype=bool)):
                raise ValueError(f'Prediction correctness disagrees with targets and gate decisions: {path}')
            pair = {"method": path.stem, "reference": reference_name, "Top1": 100 * correct.mean(),
                    **paired_top1(reference.human_rank == 1, correct)}
            if bootstrap_reps:
                delta = correct.astype(float) - (reference.human_rank.to_numpy() == 1)
                pair.update({k + '_pp': 100 * v for k, v in clustered_ci(delta, positions.game_id, reps=bootstrap_reps).items()})
            pairs.append(pair)
            if reference_name == 'allie':
                pairs.append({'method': path.stem, 'reference': 'maia3', 'Top1': 100 * correct.mean(),
                              **paired_top1(base_correct, correct)})
    pd.DataFrame(pairs).to_csv(output / 'paired_top1.csv', index=False)
    if changes:
        pd.DataFrame(changes).to_csv(output / 'paired_probability_metrics.csv', index=False)
        pd.DataFrame(rank_changes).to_csv(output / 'rank_changes.csv', index=False)
    grouped = []
    for scheme, groups in strata(positions, maia).items():
        labels = groups.categories if isinstance(groups, pd.Categorical) else groups.cat.categories if isinstance(groups.dtype, pd.CategoricalDtype) else pd.unique(groups)
        for group in labels:
            if pd.isna(group):
                continue
            mask = np.asarray(groups == group)
            if not mask.any():
                continue
            for name, frame in [('maia3', maia), ('allie', allie)]:
                grouped.append({"scheme": scheme, "stratum": str(group), "model": name,
                    "percent_rows": 100 * mask.mean(), **summarize(frame.loc[mask]),
                    "oracle_Top1": 100 * (base_correct | allie_correct)[mask].mean()})
    pd.DataFrame(grouped).to_csv(output / 'strata.csv', index=False)
    geometry = []
    for name, frame in [('maia3', maia), ('allie', allie)]:
        wrong = frame.loc[frame.human_rank > 1]
        geometry.append({"model": name, "wrong_rows": len(wrong),
            **{f"within_top{k}_percent": 100 * (wrong.human_rank <= k).mean() for k in (2, 3, 5, 10, 20)},
            "mean_p_human": wrong.p_human.mean(), "median_p_human": wrong.p_human.median(),
            "mean_p_top1": wrong.p_top1.mean(), "mean_entropy": wrong.entropy.mean(),
            "mean_top1_human_margin": (wrong.p_top1 - wrong.p_human).mean(),
            "median_top1_human_margin": (wrong.p_top1 - wrong.p_human).median(),
            "p90_top1_human_margin": (wrong.p_top1 - wrong.p_human).quantile(.9),
            "rank2_margin_below_003_percent": 100 * ((wrong.human_rank == 2) & ((wrong.p_top1 - wrong.p_human) < .03)).mean()})
    pd.DataFrame(geometry).to_csv(output / 'rank_geometry.csv', index=False)
    time_differences = []
    fastest, slowest = positions.time_spent_seconds <= 1, positions.time_spent_seconds > 7
    if fastest.any() and slowest.any():
        for name, frame in [('maia3', maia), ('allie', allie)]:
            for metric, values in [('Top1', (frame.human_rank == 1).astype(float) * 100), ('NLL', frame.nll)]:
                fast, slow = values[fastest].to_numpy(), values[slowest].to_numpy()
                delta = float(slow.mean() - fast.mean())
                se = np.sqrt(slow.var(ddof=1) / len(slow) + fast.var(ddof=1) / len(fast))
                time_differences.append({'model': name, 'metric': metric, 'slow_minus_fast': delta,
                    'ci_low': delta - 1.959963984540054 * se, 'ci_high': delta + 1.959963984540054 * se})
        pd.DataFrame(time_differences).to_csv(output / 'time_differences.csv', index=False)
    for name, frame in models.items():
        frame.to_parquet(output / f'{name}_metrics.parquet', index=False)
    (output / 'run.json').write_text(json.dumps({"rows": len(maia), "bootstrap_reps": bootstrap_reps,
        "full_paper_rows": len(maia) == 884049, "methods": list(models)}, indent=2) + '\n')
    plot_report(output)
    return rows


def plot_report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    frame = pd.read_csv(output / 'metrics.csv').set_index('method')
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ('maia3', 'allie'):
        ax.plot([1, 3, 5, 10, 20], frame.loc[name, ['Top1', 'Top3', 'Top5', 'Top10', 'Top20']].astype(float), marker='o', label=name)
    ax.set(xlabel='k', ylabel='Top-k accuracy (%)', xticks=[1, 3, 5, 10, 20])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'topk.pdf')
    plt.close(fig)
    overlap = json.loads((output / 'complementarity.json').read_text())
    matrix = np.array([[overlap['both_correct'], overlap['maia_only']],
                       [overlap['allie_only'], overlap['both_wrong']]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{matrix[i,j]:,}', ha='center', va='center',
                    color='white' if matrix[i,j] > matrix.max() / 2 else 'black')
    ax.set(xticks=[0, 1], xticklabels=['Correct', 'Wrong'], yticks=[0, 1], yticklabels=['Correct', 'Wrong'],
           xlabel='Allie', ylabel='MAIA3')
    fig.tight_layout()
    fig.savefig(output / 'complementarity.pdf')
    plt.close(fig)
    geometry = pd.read_csv(output / 'rank_geometry.csv').set_index('model')
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ('maia3', 'allie'):
        ax.plot([2, 3, 5, 10, 20], geometry.loc[name, [f'within_top{k}_percent' for k in (2, 3, 5, 10, 20)]], marker='o', label=name)
    ax.set(xlabel='k', ylabel='Top-k among Top1-wrong positions (%)', xticks=[2, 3, 5, 10, 20])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'rank_geometry.pdf')
    plt.close(fig)
    strata_frame = pd.read_csv(output / 'strata.csv')
    for scheme in ('move_time', 'fixed_seconds', 'phase', 'legal_moves'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for name in ('maia3', 'allie'):
            values = strata_frame.loc[(strata_frame.scheme == scheme) & (strata_frame.model == name)]
            for ax, metric in zip(axes, ('Top1', 'NLL') if 'time' in scheme or scheme == 'fixed_seconds' else ('Top1', 'gap_pp')):
                ax.plot(values.stratum, values[metric], marker='o', label=name)
                ax.set(xlabel=scheme.replace('_', ' '), ylabel=metric)
                ax.tick_params(axis='x', rotation=25)
                ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{scheme}.pdf')
        plt.close(fig)
