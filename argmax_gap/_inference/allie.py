"""Offline construction of the released Allie medium policy architecture."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

def normalize_config_dict(value: Any) -> dict[str, Any]:
    return dict(OmegaConf.to_container(value, resolve=True))

def build_allie_model(tokenizer: Any, model_config: dict[str, Any]) -> torch.nn.Module:
    """Construct the Allie model architecture without downloading GPT weights."""
    from optimum.bettertransformer import BetterTransformer
    from transformers import AutoModelForCausalLM, GPT2Config

    from modeling.model import CausalLMWithControlToken, CausalLMWithRegressionHead

    config = dict(model_config)
    base_model = str(config.pop("base_model"))
    use_control_token = bool(config.pop("use_control_token", False))
    use_regression_head = bool(config.pop("use_regression_head", False))
    config.pop("use_pretrained", None)
    lm_loss_coef = float(config.pop("lm_loss_coef", 1.0))
    value_loss_coef = float(config.pop("value_loss_coef", 1.0))
    time_loss_coef = float(config.pop("time_loss_coef", 1.0))

    if base_model != "gpt2-medium":
        raise ValueError(
            f"Unsupported Allie base_model={base_model!r}; add an offline config mapping before use."
        )

    hf_config = GPT2Config(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        **config,
    )
    model = AutoModelForCausalLM.from_config(hf_config)
    model = BetterTransformer.transform(model)

    if use_control_token:
        model = CausalLMWithControlToken(model)
    if use_regression_head:
        model = CausalLMWithRegressionHead(
            model,
            lm_loss_coef=lm_loss_coef,
            value_loss_coef=value_loss_coef,
            time_loss_coef=time_loss_coef,
        )
    return model

def load_checkpoint_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    candidates = [state_dict]
    for prefix in ("_orig_mod.", "module."):
        if all(str(key).startswith(prefix) for key in state_dict):
            candidates.append({str(key).removeprefix(prefix): value for key, value in state_dict.items()})
    errors = []
    for candidate in candidates:
        missing, unexpected = model.load_state_dict(candidate, strict=False)
        if not missing and not unexpected:
            return
        errors.append((missing[:10], unexpected[:10]))
    missing, unexpected = errors[-1]
    raise RuntimeError(f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}")

def load_tokenizer_and_model(
    allie_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    allie_src = allie_root / "src"
    if str(allie_src) not in sys.path:
        sys.path.insert(0, str(allie_src))

    from modeling.data import UCITokenizer

    config = OmegaConf.load(config_path)
    tokenizer = UCITokenizer(**normalize_config_dict(config.data_config.tokenizer_config))
    model = build_allie_model(tokenizer, normalize_config_dict(config.model_config))
    load_checkpoint_state(model, checkpoint_path)
    model.to(device)
    model.eval()
    return tokenizer, model, normalize_config_dict(config)
