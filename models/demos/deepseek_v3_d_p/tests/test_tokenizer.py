# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizer-related tests split from test_prefill_transformer.py.

Tests tokenize_prompt_to_isl, tokenize_prompt_to_chat_template and prompt token counts.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    get_4d_causal_mask,
    tokenize_prompt_to_chat_template,
    tokenize_prompt_to_isl,
)


@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
def test_tokenize_prompt_to_isl(tokenizer):
    max_isl = 10
    input_ids, attention_mask, tokens = tokenize_prompt_to_isl(
        tokenizer, max_isl=max_isl, prompt_text="This is a test prompt.", debug=True
    )

    logger.debug(f"Input IDs: {input_ids}")
    logger.debug(f"Attention Mask: {attention_mask}")
    logger.debug(f"Tokens: {tokens}")

    assert input_ids.shape == (1, max_isl), f"Expected input_ids shape (1, {max_isl}), got {input_ids.shape}"

    torch.set_printoptions(threshold=float("inf"), edgeitems=3, precision=2, linewidth=200)
    logger.debug(f"4D Causal Attention Mask shape:\n{get_4d_causal_mask(attention_mask, causal_only=True)}")
    logger.debug(f"4D Causal Attention Mask Paddshape:\n{get_4d_causal_mask(attention_mask, causal_only=False)}")


@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
def test_tokenize_prompt_to_chat_template(tokenizer):
    max_isl = 64
    input_ids, tokens = tokenize_prompt_to_chat_template(
        tokenizer,
        max_isl=max_isl,
        user_prompt="What is the capital of Serbia?",
        system_prompt="You are a helpful assistant.",
        debug=True,
    )
    logger.debug(f"Input IDs: {input_ids}")
    logger.debug(f"Tokens: {tokens}")

    assert input_ids.shape == (1, max_isl), f"Expected input_ids shape (1, {max_isl}), got {input_ids.shape}"


@pytest.mark.parametrize(
    "json_path",
    [
        Path("models/demos/deepseek_v3_d_p/demo/test_prompt_ABC_short.json"),
        Path("models/demos/deepseek_v3_d_p/demo/test_prompt_64tok.json"),
        Path("models/demos/deepseek_v3_d_p/demo/test_prompt_960tok.json"),
        Path("models/demos/deepseek_v3_d_p/demo/test_pie_960tok.json"),
    ],
    ids=["short", "64tok", "960tok", "pie"],
)
@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
def test_token_count(tokenizer, json_path):
    """Tokenize a prompt JSON without padding and report token count."""
    from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

    logger.info(f"{json_path=}")
    prompts = load_prompts_from_json(str(json_path))
    assert prompts, f"No prompts found in {json_path}"
    prompt_text = prompts[0]

    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    logger.info(f"{len(tokens)=}")
    logger.debug(f"Prompt: {repr(prompt_text)}")
    logger.debug(f"Token IDs: {tokens}")

    input_ids, attention_mask, tokens_padded = tokenize_prompt_to_isl(
        tokenizer, max_isl=1024, prompt_text=prompts, debug=True
    )
    number_of_non_padded_tokens = attention_mask.sum().item()  # should be returned by tokenize..
    logger.info(f"{number_of_non_padded_tokens=}")
    logger.debug(f"Prompt: {repr(tokens_padded)}")
    logger.debug(f"Token IDs: {input_ids}")

    # note number_of_non_padded_tokens is len(tokens) + 1 for the added BOS token
