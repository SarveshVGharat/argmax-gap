"""Frozen MAIA3 and Allie policy inference with full legal-move distributions."""
from __future__ import annotations

import json
import math
import sys
from collections import deque
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from .data import iter_parquet, string_list, validate_position
from .upstream import ASSETS, verify_hash, verify_repository

OUTPUT_SCHEMA = pa.schema([
    ("row_id", pa.int64()), ("game_id", pa.string()), ("human_move_uci", pa.string()),
    ("legal_moves_uci", pa.list_(pa.string())), ("legal_probs", pa.list_(pa.float32())),
    ("legal_logits", pa.list_(pa.float32())), ("p_human", pa.float32()),
    ("p_top1", pa.float32()), ("top1_move_uci", pa.string()), ("human_rank", pa.int32()),
    ("entropy", pa.float32()), ("num_legal_moves", pa.int16()), ("nll", pa.float32()),
    *[(f"is_top{k}", pa.bool_()) for k in (1, 3, 5, 10, 20)],
    ("time_spent_seconds", pa.float32()), ("player_elo", pa.float32()),
    ("opponent_elo", pa.float32()), ("move_number", pa.int32()), ("phase", pa.string()),
])
ALLIE_SCHEMA = pa.schema(list(OUTPUT_SCHEMA) + [
    pa.field("model_think_time", pa.float32()), pa.field("model_value", pa.float32())])


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision != "float32":
        return torch.amp.autocast("cuda", dtype=getattr(torch, precision))
    return nullcontext()


def distribution_record(row: dict[str, Any], moves: list[str], logits: torch.Tensor,
                        probs: torch.Tensor, nll: float, model_name: str) -> dict[str, Any]:
    if not torch.isfinite(logits).all() or not torch.isfinite(probs).all():
        raise ValueError(f"Non-finite distribution at row {row['row_id']}")
    if abs(float(probs.sum()) - 1) > 1e-5:
        raise ValueError(f"Unnormalized distribution at row {row['row_id']}")
    human = moves.index(row["human_move_uci"])
    rank = int((logits > logits[human]).sum()) + 1
    top_scores = logits if model_name == "maia3" else probs
    top = int(torch.argsort(top_scores, descending=True, stable=True)[0])
    if model_name == "maia3":
        p64 = probs.numpy().astype(np.float64)
        entropy = float(-(p64 * np.log(np.clip(p64, 1e-12, 1.0))).sum())
    else:
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-45))).sum())
    return {
        **{key: row[key] for key in ["row_id", "game_id", "human_move_uci", "time_spent_seconds",
                                    "player_elo", "opponent_elo", "move_number", "phase"]},
        "legal_moves_uci": moves, "legal_probs": probs.tolist(), "legal_logits": logits.tolist(),
        "p_human": float(probs[human]), "p_top1": float(probs[top]), "top1_move_uci": moves[top],
        "human_rank": rank, "entropy": entropy, "num_legal_moves": len(moves), "nll": nll,
        **{f"is_top{k}": rank <= k for k in (1, 3, 5, 10, 20)},
    }


class Maia3Policy:
    def __init__(self, root: Path, checkpoint: Path, device: torch.device, precision: str):
        sys.path.insert(0, str(root.resolve()))
        from maia3.uci import parse_args
        from maia3.models import MAIA3Model
        from maia3.utils import get_all_possible_moves

        self.cfg = parse_args(["--model", "maia3-79m", "--checkpoint-path", str(checkpoint),
                               "--device", str(device)])
        self.device, self.precision = device, precision
        self.model = MAIA3Model(self.cfg)
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint_data.get("model_state_dict", checkpoint_data)
        state = {key.replace("smolgen", "gab"): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.move_ids = {move: index for index, move in enumerate(get_all_possible_moves())}

    def tokens(self, row: dict[str, Any], board: chess.Board) -> torch.Tensor:
        from maia3.dataset import tokenize_board, get_historical_tokens

        # Every released position includes explicit past FENs. Fallback to a
        # complete prefix supports custom positions without silently losing history.
        past_fens = string_list(row.get("past_fens"))
        if past_fens:
            boards = [chess.Board(fen) for fen in past_fens[-7:]] + [board]
        else:
            validate_position(row, replay_prefix=True)
            replay = chess.Board()
            history = deque([replay.copy(stack=False)], maxlen=8)
            for move in string_list(row["previous_moves_uci"]):
                replay.push_uci(move)
                history.append(replay.copy(stack=False))
            boards = list(history)
        history = deque((tokenize_board(item) for item in boards), maxlen=8)
        return get_historical_tokens(history, self.cfg,
            base=float(row["initial_time_seconds"]), inc=float(row["increment_seconds"]),
            clk_left_before=float(row["clock_before_seconds"]), clk_ponder=0.0)

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from maia3.utils import mirror_move

        tokens, ids, legal_moves = [], [], []
        mask = torch.zeros((len(rows), len(self.move_ids)), dtype=torch.bool, device=self.device)
        for index, row in enumerate(rows):
            board = validate_position(row)
            moves = string_list(row["legal_moves_uci"])
            legal_ids = [self.move_ids[mirror_move(move) if board.turn == chess.BLACK else move] for move in moves]
            legal_moves.append(moves)
            ids.append(legal_ids)
            mask[index, legal_ids] = True
            tokens.append(self.tokens(row, board))
        tokens_tensor = torch.stack(tokens).to(self.device)
        self_elo = torch.tensor([int(row["player_elo"]) for row in rows], device=self.device)
        oppo_elo = torch.tensor([int(row["opponent_elo"]) for row in rows], device=self.device)
        with torch.inference_mode(), autocast_context(self.device, self.precision):
            logits, _, _ = self.model(tokens_tensor, self_elo, oppo_elo)
        masked = logits.float().masked_fill(~mask, -torch.inf)
        log_probs = torch.log_softmax(masked, dim=-1)
        probs = log_probs.exp()
        records = []
        for index, row in enumerate(rows):
            moves = legal_moves[index]
            human_id = ids[index][moves.index(row["human_move_uci"])]
            records.append(distribution_record(row, moves, masked[index, ids[index]].cpu(),
                probs[index, ids[index]].cpu(), float(-log_probs[index, human_id]), "maia3"))
        return records


class AlliePolicy:
    def __init__(self, root: Path, checkpoint: Path, device: torch.device, precision: str):
        from ._inference.allie import load_tokenizer_and_model

        self.device, self.precision = device, precision
        self.tokenizer, self.model, _ = load_tokenizer_and_model(
            root.resolve(), root.resolve() / "pretrain_config" / "medium.yaml", checkpoint, device)

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from modeling.data import Game, undo_time_normalization

        games, legal_moves = [], []
        for row in rows:
            board = validate_position(row, replay_prefix=True)
            moves = string_list(row["previous_moves_uci"])
            # The target move, its duration, and game result are never inputs.
            games.append(Game(time_control=str(row["time_control"]), white_elo=int(row["white_elo"]),
                              black_elo=int(row["black_elo"]), outcome=None, normal_termination=False,
                              moves=moves, moves_seconds=list(row["previous_move_seconds"]), next_move_seconds=None))
            legal_moves.append([move.uci() for move in board.legal_moves])
        batch = self.tokenizer.pad_and_collate(games, return_labels=False)
        batch = {key: value.to(self.device) for key, value in batch.items()
                 if key in {"input_ids", "attention_mask", "position_ids"}}
        last = batch["attention_mask"].bool().long().sum(dim=1) - 1
        with torch.inference_mode(), autocast_context(self.device, self.precision):
            output = self.model(**batch)
        indices = torch.arange(len(rows), device=self.device)
        logits = output["logits"][indices, last].float().cpu()
        time = output["time_logits"][indices, last].float().cpu().view(-1)
        value = output["value_logits"][indices, last].float().cpu().view(-1)
        records = []
        for index, row in enumerate(rows):
            moves = legal_moves[index]
            legal_ids = [self.tokenizer.token_to_id[move] for move in moves]
            legal_logits = logits[index, legal_ids]
            probs = torch.softmax(legal_logits, dim=0)
            p_human = float(probs[moves.index(row["human_move_uci"])])
            record = distribution_record(row, moves, legal_logits, probs,
                                         -math.log(max(p_human, 1e-45)), "allie")
            record["model_think_time"] = float(undo_time_normalization(time[index]))
            record["model_value"] = float(value[index])
            records.append(record)
        return records


def evaluate(model_name: str, input_path: Path, output_path: Path, upstream_root: Path,
             checkpoint: Path, device_name: str = "cpu", batch_size: int = 128,
             precision: str | None = None, max_rows: int | None = None) -> dict[str, Any]:
    if batch_size < 1 or (max_rows is not None and max_rows < 1):
        raise ValueError("Batch size and max rows must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    verify_repository(upstream_root, model_name)
    asset_name = "maia3-79m.pt" if model_name == "maia3" else "allie-medium.pt"
    verify_hash(checkpoint, ASSETS[asset_name][-1])
    precision = precision or ("float16" if model_name == "maia3" and device.type == "cuda" else "float32")
    if device.type == "cpu" and precision != "float32":
        raise ValueError("CPU inference uses float32")
    if device.type == "cuda" and model_name == "allie":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    policy_type = Maia3Policy if model_name == "maia3" else AlliePolicy
    policy = policy_type(upstream_root, checkpoint, device, precision)
    schema = OUTPUT_SCHEMA if model_name == "maia3" else ALLIE_SCHEMA
    total = pq.ParquetFile(input_path).metadata.num_rows
    expected = min(total, max_rows) if max_rows is not None else total
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    metrics = {"rows_evaluated": 0, "move_nll": 0.0, **{f"top{k}": 0.0 for k in (1, 3, 5, 10, 20)}}
    pending = []

    def write_pending(writer):
        records = policy.predict(pending)
        writer.write_table(pa.Table.from_pylist(records, schema=schema))
        for record in records:
            metrics["rows_evaluated"] += 1
            metrics["move_nll"] += record["nll"]
            for k in (1, 3, 5, 10, 20):
                metrics[f"top{k}"] += record[f"is_top{k}"]
        pending.clear()

    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for index, row in enumerate(tqdm(islice(iter_parquet(input_path), max_rows), total=expected, desc=model_name)):
            if row.get("row_id", index) != index:
                raise ValueError(f"Noncontiguous row IDs at position {index}")
            row["row_id"] = index
            pending.append(row)
            if len(pending) == batch_size:
                write_pending(writer)
        if pending:
            write_pending(writer)
    if metrics["rows_evaluated"] != expected or expected == 0:
        raise ValueError("Incomplete or empty inference output")
    temporary.replace(output_path)
    for key in metrics:
        if key != "rows_evaluated":
            metrics[key] /= expected
    input_summary = input_path.with_suffix(".summary.json")
    input_is_smoke = input_summary.exists() and json.loads(input_summary.read_text()).get("smoke_subset", False)
    metrics.update(model=model_name, precision=precision, device=str(device), smoke_subset=max_rows is not None or input_is_smoke,
                   rows_skipped=0, topk_rule="1 + count(legal_logit > human_logit)")
    output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
