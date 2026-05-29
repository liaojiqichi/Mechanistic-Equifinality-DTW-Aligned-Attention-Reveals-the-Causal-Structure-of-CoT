import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import warnings
import random
import re

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 200
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Loading {MODEL_ID} on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    output_attentions=True,
    output_hidden_states=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()
NUM_LAYERS = model.config.num_hidden_layers

PREFIX_ALGEBRA = "Solve step by step using algebra. Let x be unknown. "
PREFIX_ARITHMETIC = "Solve step by step using arithmetic only. "
FINAL_PHRASE = "\nFinal answer: "

CACHE_FILE = "cached_cots.csv"
RESULTS_FILE = "results_.csv"

def cosine(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def linear_cka(X, Y):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X - X.mean()
    Y = Y - Y.mean()
    return (np.dot(X, Y) ** 2) / ((np.dot(X, X) * np.dot(Y, Y)) + 1e-12)

def normalize(x):
    x = np.maximum(np.array(x, dtype=np.float64), 0)
    s = x.sum()
    return x / (s + 1e-12)

def jsd(p, q):
    p = normalize(p)
    q = normalize(q)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m + 1e-12)) + np.sum(q * np.log(q / m + 1e-12)))

def extract_signals(full_text):
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    probs = F.softmax(outputs.logits[0, -1], dim=-1).cpu().numpy()
    hs = [outputs.hidden_states[l][0, -1, :].cpu().numpy() for l in range(NUM_LAYERS)]
    seq_len = outputs.attentions[0].shape[-1]
    rollout = np.eye(seq_len)
    rollouts = []
    for layer in outputs.attentions:
        attn = layer[0].mean(dim=0).cpu().numpy()
        attn = attn + np.eye(seq_len)
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-12)
        rollout = attn @ rollout
        rollouts.append(rollout.copy())

    return probs, hs, rollouts

def shuffled_probs(full_text):
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    return F.softmax(out.logits[0, -1], dim=-1).cpu().numpy()


if not os.path.exists(CACHE_FILE):
    print(f"\n Generating CoTs and saving to {CACHE_FILE}...")
    dataset = load_dataset("gsm8k", "main", split="test").shuffle(seed=SEED).select(range(SAMPLE_SIZE))

    cache_data = []
    for i, item in enumerate(tqdm(dataset, desc="Generating")):
        question = item["question"]
        base_prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)

        inputs_A = tokenizer(base_prompt + PREFIX_ALGEBRA, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_A = model.generate(**inputs_A, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        cot_A = tokenizer.decode(out_A[0][inputs_A.input_ids.shape[1]:], skip_special_tokens=True)

        inputs_B = tokenizer(base_prompt + PREFIX_ARITHMETIC, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_B = model.generate(**inputs_B, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        cot_B = tokenizer.decode(out_B[0][inputs_B.input_ids.shape[1]:], skip_special_tokens=True)

        cache_data.append({
            "question": question,
            "base_prompt": base_prompt,
            "cot_A": cot_A,
            "cot_B": cot_B
        })

    pd.DataFrame(cache_data).to_csv(CACHE_FILE, index=False)
    print("Generation complete and cached!")
else:
    print(f"\n Loading {CACHE_FILE}...")

df_cache = pd.read_csv(CACHE_FILE)
results = []

for _, row in tqdm(df_cache.iterrows(), total=len(df_cache), desc="Evaluating"):
    q = row["question"]
    base_prompt = row["base_prompt"]
    cot_A = row["cot_A"]
    cot_B = row["cot_B"]

    ctx_len = len(tokenizer.encode(base_prompt, add_special_tokens=False))

    full_A = base_prompt + PREFIX_ALGEBRA + cot_A + FINAL_PHRASE
    full_B = base_prompt + PREFIX_ARITHMETIC + cot_B + FINAL_PHRASE

    probs_A, hs_A, roll_A = extract_signals(full_A)
    probs_B, hs_B, roll_B = extract_signals(full_B)

    def shuffle_cot_text(cot):
        chunks = [c for c in cot.split("\n") if len(c.strip()) > 0]
        random.shuffle(chunks)
        return "\n".join(chunks)

    probs_A_shuf = shuffled_probs(base_prompt + PREFIX_ALGEBRA + shuffle_cot_text(cot_A) + FINAL_PHRASE)
    probs_B_shuf = shuffled_probs(base_prompt + PREFIX_ARITHMETIC + shuffle_cot_text(cot_B) + FINAL_PHRASE)

    tvd_base = np.sum(np.abs(probs_A - probs_B)) / 2
    tvd_shuf_A = np.sum(np.abs(probs_A - probs_A_shuf)) / 2
    tvd_shuf_B = np.sum(np.abs(probs_B - probs_B_shuf)) / 2

    cos_layers = [cosine(hs_A[l], hs_B[l]) for l in range(NUM_LAYERS)]
    cka_layers = [linear_cka(hs_A[l], hs_B[l]) for l in range(NUM_LAYERS)]

    attn_jsd = [jsd(roll_A[l][-1][:ctx_len], roll_B[l][-1][:ctx_len]) for l in range(NUM_LAYERS)]

    results.append({
        "question": q,
        "tvd_base": float(tvd_base),
        "tvd_shuffle_A": float(tvd_shuf_A),
        "tvd_shuffle_B": float(tvd_shuf_B),
        "cosine_mean": float(np.mean(cos_layers)),
        "cka_mean": float(np.mean(cka_layers)),
        "attn_jsd_mean": float(np.mean(attn_jsd))
    })


df_results = pd.DataFrame(results)
df_results.to_csv(RESULTS_FILE, index=False)

print(f"Mean TVD (Base):             {df_results['tvd_base'].mean():.4f}")
print(f"Mean Cosine (Fixed):         {df_results['cosine_mean'].mean():.4f}")
print(f"Mean CKA (Fixed! No NaNs):   {df_results['cka_mean'].mean():.4f}")
print(f"Mean JSD:                    {df_results['attn_jsd_mean'].mean():.4f}")
print(f"Metrics saved to: {RESULTS_FILE}")
