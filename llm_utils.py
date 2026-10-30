"""Loading and prompt helpers shared by LLM fine-tuning and inference (Llama + Gemma)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import CHAT_TEMPLATE_KWARGS, LLM_IDS, llm_messages


def load_llm(name, adapter_dir=None):
    """4-bit NF4 base model (same quantisation for zero-shot and fine-tuned runs)."""
    hf_id = LLM_IDS[name]
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="eager" if name == "gemma" else "sdpa")  # Gemma-2 soft-capping needs eager
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    return model, tok


def prompt_text(tok, src):
    return tok.apply_chat_template(llm_messages(src), tokenize=False, add_generation_prompt=True,
                                   **CHAT_TEMPLATE_KWARGS)


def stop_token_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|eot_id|>", "<end_of_turn>"):
        if t in tok.get_vocab():
            ids.add(tok.convert_tokens_to_ids(t))
    return sorted(ids)


def build_example(tok, src, tgt, max_len=256):
    """input_ids for prompt+answer with labels masked on the prompt (loss on the translation only)."""
    prompt = prompt_text(tok, src)
    full = tok.apply_chat_template(llm_messages(src, tgt), tokenize=False, **CHAT_TEMPLATE_KWARGS)
    assert full.startswith(prompt), "chat template mismatch"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def clean_output(text):
    """Strip chatty wrappers ('Here is the translation:') that instruct models sometimes add."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    lines = [l for l in lines if not l.lower().rstrip(":").startswith(
        ("here is", "here's", "sure", "translation", "english translation", "the translation"))]
    out = lines[0] if lines else ""
    if len(out) > 1 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1]
    return out.strip()
