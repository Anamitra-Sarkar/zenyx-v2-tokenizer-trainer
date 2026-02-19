# ╔══════════════════════════════════════════════════════════════════════╗
# ║   ZENYX-V2 TOKENIZER TRAINER  —  TRUE STREAMING v8 (Fixed)         ║
# ║   Byte-Level BPE | 32k vocab | 1GB Corpus | Direct Stream Train    ║
# ╚══════════════════════════════════════════════════════════════════════╝

!pip install -U -q transformers datasets tokenizers huggingface_hub

# ══════════════════════════════════════════════════════════════════════
# §0  CONFIG
# ══════════════════════════════════════════════════════════════════════
HF_TOKEN = "hf_token"   # <<< Replace with your actual write token in Kaggle
REPO_ID  = "Arko007/zenyx-v2-tokenizer"
SAVE_DIR = "./zenyx_tokenizer"

VOCAB_SIZE          = 32_768
TARGET_CORPUS_BYTES = 1 * 1024 * 1024 * 1024   # 1 GB

MATH_RATIO    = 0.40
CODE_RATIO    = 0.40
ENGLISH_RATIO = 0.20

FINEWEB_MIN_SCORE = 3.0

# Heartbeat: log a progress line every N rows
HEARTBEAT_ROWS = 5_000

CHECKPOINT_FILE = "./zenyx_checkpoint.json"

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|unk|>",
    "<think>",
    "</think>",
    "<scratchpad>",
    "<verify>",
]

# ══════════════════════════════════════════════════════════════════════
# §1  IMPORTS
# ══════════════════════════════════════════════════════════════════════
import os, sys, json, time, logging, unicodedata
from pathlib import Path

# FIX: Force line-buffered stdout so Kaggle shows every log line immediately.
sys.stdout.reconfigure(line_buffering=True)

from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Digits, Split, Sequence as PTSequence
from tokenizers import processors, decoders
from huggingface_hub import HfApi, login, create_repo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ZenyxV2")

# ══════════════════════════════════════════════════════════════════════
# §2  HF SETUP
# ══════════════════════════════════════════════════════════════════════
def setup_hf_repo() -> bool:
    """Returns True if HF is reachable and repo is ready, False otherwise.
    Wrapped in try/except so a bad/revoked token NEVER kills the run
    before training starts.
    """
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        try:
            api.repo_info(repo_id=REPO_ID, repo_type="model", token=HF_TOKEN)
            log.info(f"[HF] Repo '{REPO_ID}' exists.")
        except Exception:
            create_repo(repo_id=REPO_ID, token=HF_TOKEN,
                        repo_type="model", private=False, exist_ok=True)
            log.info(f"[HF] Repo '{REPO_ID}' created.")
        return True
    except Exception as e:
        log.warning(f"[HF] Setup failed ({e}). Training will proceed; upload may fail.")
        return False

# ══════════════════════════════════════════════════════════════════════
# §3  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════
EMPTY_CKPT = {
    "training_complete":   False,
    "vocab_size_achieved": 0,
    "ts_train_start":      None,
    "ts_train_end":        None,
}

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        for k, v in EMPTY_CKPT.items():
            ckpt.setdefault(k, v)
        log.info(f"[CKPT] Loaded: {ckpt}")
        return ckpt
    return dict(EMPTY_CKPT)

def save_checkpoint(ckpt: dict) -> None:
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)

def print_session_status(ckpt: dict) -> None:
    print("\n" + "━" * 60)
    print("  ZENYX-V2  —  SESSION STATUS")
    print("━" * 60)
    print(f"  Mode           : TRUE STREAMING (no spool file)")
    print(f"  Training done  : {'✓' if ckpt['training_complete'] else '✗ (will train)'}")
    if ckpt["vocab_size_achieved"]:
        print(f"  Vocab achieved : {ckpt['vocab_size_achieved']:,}")
    print("━" * 60 + "\n")

# ══════════════════════════════════════════════════════════════════════
# §4  DISK SAFETY
# ══════════════════════════════════════════════════════════════════════
def check_disk_space() -> None:
    stat   = os.statvfs(".")
    free   = stat.f_bavail * stat.f_frsize
    needed = 50 * 1024 * 1024  # 50 MB for output files only
    log.info(f"[DISK] Free: {free/1e9:.2f}GB  Needed: ~{needed/1e6:.0f}MB (output only)")
    if free < needed:
        raise SystemError(
            f"Insufficient disk: {free/1e9:.2f}GB free, {needed/1e6:.0f}MB needed."
        )

# ══════════════════════════════════════════════════════════════════════
# §5  TEXT SANITISATION
# ══════════════════════════════════════════════════════════════════════
def _to_ascii_digit(ch: str) -> str:
    try:
        val = unicodedata.numeric(ch)
        if val == int(val) and 0 <= int(val) <= 9:
            return str(int(val))
    except (ValueError, TypeError):
        pass
    return ch

def sanitise_text(text: str) -> str | None:
    if not text or not text.strip():
        return None
    if "\x00" in text:
        text = text.replace("\x00", "")
        if not text.strip():
            return None
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = "".join(
        _to_ascii_digit(c) if (c.isnumeric() and not ("0" <= c <= "9")) else c
        for c in text
    )
    return text

# ══════════════════════════════════════════════════════════════════════
# §6  STREAMING SOURCES
#
# stack-edu config names are the exact language label used by HuggingFaceTB.
# All 15 are attempted; any that don’t exist on the hub are skipped with a
# warning — the rest are interleaved normally.
# ══════════════════════════════════════════════════════════════════════
def stream_finemath(target_bytes: int):
    log.info(f"[MATH] Loading finemath-4plus (streaming)...")
    try:
        ds = load_dataset("HuggingFaceTB/finemath", name="finemath-4plus",
                          split="train", streaming=True)
    except Exception as e:
        log.error(f"[MATH] Failed to load dataset: {e}")
        return

    log.info(f"[MATH] Dataset ready | target={target_bytes/1e6:.0f}MB")
    consumed  = 0
    row_count = 0
    for row in ds:
        text = sanitise_text(row.get("text", ""))
        if text is None:
            continue
        blen = len(text.encode("utf-8"))
        yield text, blen
        consumed  += blen
        row_count += 1
        if row_count % HEARTBEAT_ROWS == 0:
            log.info(f"[MATH] heartbeat: {row_count:,} rows | {consumed/1e6:.1f}MB streamed")
        if consumed >= target_bytes:
            break


def stream_stack_edu(target_bytes: int):
    # All 15 languages available in HuggingFaceTB/stack-edu.
    # Config name (value) = exact name used by the HuggingFace dataset hub.
    # Any config that doesn’t exist is silently skipped; the rest are interleaved.
    lang_map = {
        # --- confirmed working from prior runs ---
        "Python":     "Python",
        "Java":       "Java",
        "C":          "C",
        "C++":        "Cpp",
        # --- extended set ---
        "JavaScript": "JavaScript",
        "TypeScript": "TypeScript",
        "SQL":        "SQL",
        "Go":         "Go",
        "Rust":       "Rust",
        "PHP":        "PHP",
        "Ruby":       "Ruby",
        "Swift":      "Swift",
        "Kotlin":     "Kotlin",
        "Scala":      "Scala",
        "Shell":      "Shell",
    }

    streams = []
    log.info(f"[CODE] Loading {len(lang_map)} language configs from stack-edu...")
    for lang_name, config in lang_map.items():
        try:
            ds = load_dataset("HuggingFaceTB/stack-edu", name=config,
                              split="train", streaming=True)
            streams.append(ds)
            log.info(f"[CODE]   + {config:<14} ({lang_name}) ✓")
        except Exception as e:
            log.warning(f"[CODE]   - {config:<14} ({lang_name}) ✗  ({e})")

    if not streams:
        log.error("[CODE] No code streams available!")
        return

    combined = interleave_datasets(streams, stopping_strategy="first_exhausted")
    log.info(f"[CODE] {len(streams)}/{len(lang_map)} languages active | "
             f"target={target_bytes/1e6:.0f}MB")

    consumed  = 0
    row_count = 0
    for row in combined:
        raw_text = row.get("content", row.get("text", ""))
        text = sanitise_text(raw_text)
        if text is None or len(text.strip()) < 20:
            continue
        blen = len(text.encode("utf-8"))
        yield text, blen
        consumed  += blen
        row_count += 1
        if row_count % HEARTBEAT_ROWS == 0:
            log.info(f"[CODE] heartbeat: {row_count:,} rows | {consumed/1e6:.1f}MB streamed")
        if consumed >= target_bytes:
            break


def stream_fineweb_edu(target_bytes: int):
    log.info(f"[ENG] Loading fineweb-edu (sample-10BT, streaming)...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
    except Exception as e:
        log.error(f"[ENG] Failed to load dataset: {e}")
        return

    log.info(f"[ENG] Dataset ready | score>{FINEWEB_MIN_SCORE} | target={target_bytes/1e6:.0f}MB")
    consumed  = 0
    row_count = 0
    for row in ds:
        if row.get("score", 0.0) < FINEWEB_MIN_SCORE:
            continue
        text = sanitise_text(row.get("text", ""))
        if text is None:
            continue
        blen = len(text.encode("utf-8"))
        yield text, blen
        consumed  += blen
        row_count += 1
        if row_count % HEARTBEAT_ROWS == 0:
            log.info(f"[ENG] heartbeat: {row_count:,} rows | {consumed/1e6:.1f}MB streamed")
        if consumed >= target_bytes:
            break

# ══════════════════════════════════════════════════════════════════════
# §7  TRUE STREAMING COMBINER
#     Feeds text directly into the BPE trainer — no spool file at all.
# ══════════════════════════════════════════════════════════════════════
def stream_combined_for_training():
    math_target    = int(TARGET_CORPUS_BYTES * MATH_RATIO)
    code_target    = int(TARGET_CORPUS_BYTES * CODE_RATIO)
    english_target = int(TARGET_CORPUS_BYTES * ENGLISH_RATIO)

    log.info("[STREAM] Initialising streaming generators...")
    gen_math    = stream_finemath(math_target)
    gen_code    = stream_stack_edu(code_target)
    gen_english = stream_fineweb_edu(english_target)

    log.info("[STREAM] Generators ready. Streaming directly to BPE trainer.")
    log.info(f"[STREAM] Targets → math={math_target/1e6:.0f}MB  "
             f"code={code_target/1e6:.0f}MB  eng={english_target/1e6:.0f}MB")

    sources_raw = [
        ("math",    gen_math,    math_target,    4),
        ("code",    gen_code,    code_target,    4),
        ("english", gen_english, english_target, 2),
    ]

    consumed       = {"math": 0, "code": 0, "english": 0}
    done           = {name: False for name, *_ in sources_raw}
    weighted_cycle = []
    for name, gen, tgt, weight in sources_raw:
        weighted_cycle.extend([(name, gen, tgt)] * weight)

    total_consumed = 0
    total_rows     = 0
    idx            = 0
    t_start        = time.time()

    while total_consumed < TARGET_CORPUS_BYTES:
        if all(done[name] for name, *_ in sources_raw):
            log.info("[STREAM] All sources exhausted.")
            break

        name, gen, tgt = weighted_cycle[idx % len(weighted_cycle)]
        idx += 1

        if done[name]:
            continue

        try:
            text, blen = next(gen)
        except StopIteration:
            done[name] = True
            log.info(f"[STREAM] '{name}' exhausted at "
                     f"{consumed[name]/1e6:.1f}MB (target={tgt/1e6:.0f}MB).")
            continue

        if consumed[name] + blen > tgt:
            done[name] = True
            continue

        consumed[name]  += blen
        total_consumed  += blen
        total_rows      += 1

        if total_rows % HEARTBEAT_ROWS == 0:
            pct     = 100 * total_consumed / TARGET_CORPUS_BYTES
            elapsed = time.time() - t_start
            speed   = total_consumed / elapsed / 1e6 if elapsed > 0 else 0
            log.info(f"[STREAM] {total_consumed/1e6:.0f}MB ({pct:.1f}%) | "
                     f"math={consumed['math']/1e6:.0f}MB  "
                     f"code={consumed['code']/1e6:.0f}MB  "
                     f"eng={consumed['english']/1e6:.0f}MB | "
                     f"{speed:.2f}MB/s | {total_rows:,} rows")

        yield text

# ══════════════════════════════════════════════════════════════════════
# §8  PRE-TOKENIZER
# ══════════════════════════════════════════════════════════════════════
def build_pretokenizer() -> PTSequence:
    regex = Regex(
        r"""'s|'t|'re|'ve|'m|'ll|'d"""
        r"""| ?\p{L}+"""
        r"""| ?[^\s\p{L}]+[\r\n]*"""
        r"""|\s*[\r\n]+"""
        r"""|\s+"""
    )
    return PTSequence([
        Split(pattern=regex, behavior="isolated"),
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])

# ══════════════════════════════════════════════════════════════════════
# §9  TOKENIZER + TRAINER
# ══════════════════════════════════════════════════════════════════════
def build_tokenizer() -> Tokenizer:
    model = BPE(unk_token="<|unk|>", fuse_unk=False)
    tok   = Tokenizer(model)
    tok.pre_tokenizer  = build_pretokenizer()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.decoder        = decoders.ByteLevel(
        add_prefix_space=False, trim_offsets=False, use_regex=True)
    return tok

def build_trainer() -> BpeTrainer:
    return BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )

# ══════════════════════════════════════════════════════════════════════
# §10  TRAINING
# ══════════════════════════════════════════════════════════════════════
def train_tokenizer(ckpt: dict) -> Tokenizer:
    tok     = build_tokenizer()
    trainer = build_trainer()

    # estimated_rows is a hint to BpeTrainer progress bar only — not a hard limit.
    estimated_rows = max(TARGET_CORPUS_BYTES // 300, 1)
    log.info(f"[TRAIN] est_rows={estimated_rows:,} | True streaming to trainer started.")

    if ckpt["ts_train_start"] is None:
        ckpt["ts_train_start"] = time.time()
        save_checkpoint(ckpt)

    tok.train_from_iterator(
        iterator=stream_combined_for_training(),
        trainer=trainer,
        length=estimated_rows,
    )

    ckpt["ts_train_end"]        = time.time()
    ckpt["training_complete"]   = True
    ckpt["vocab_size_achieved"] = tok.get_vocab_size()
    save_checkpoint(ckpt)

    elapsed = ckpt["ts_train_end"] - ckpt["ts_train_start"]
    log.info(f"[TRAIN] Done in {elapsed/60:.1f}min | vocab={tok.get_vocab_size():,}")
    return tok

# ══════════════════════════════════════════════════════════════════════
# §11  SAVE
# ══════════════════════════════════════════════════════════════════════
def save_tokenizer(tok: Tokenizer) -> dict:
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    vocab = tok.get_vocab()

    def safe_id(token, fallback):
        tid = vocab.get(token)
        if tid is None:
            log.warning(f"[SAVE] '{token}' not in vocab — using fallback={fallback}")
            return fallback
        return tid

    endoftext_id = safe_id("<|endoftext|>", 0)
    pad_id       = safe_id("<|pad|>",       1)
    unk_id       = safe_id("<|unk|>",       2)

    tj_path = os.path.join(SAVE_DIR, "tokenizer.json")
    tok.save(tj_path, pretty=True)

    cfg = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "model_max_length": 131072,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": VOCAB_SIZE,
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "additional_special_tokens": ["<think>", "</think>", "<scratchpad>", "<verify>"],
        "added_tokens_decoder": {
            str(endoftext_id): {"content": "<|endoftext|>", "special": True,
                                "lstrip": False, "rstrip": False,
                                "single_word": False, "normalized": False},
            str(pad_id):       {"content": "<|pad|>", "special": True,
                                "lstrip": False, "rstrip": False,
                                "single_word": False, "normalized": False},
            str(unk_id):       {"content": "<|unk|>", "special": True,
                                "lstrip": False, "rstrip": False,
                                "single_word": False, "normalized": False},
        },
    }
    cfg_path = os.path.join(SAVE_DIR, "tokenizer_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    stm = {
        "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
        "unk_token": "<|unk|>",       "pad_token": "<|pad|>",
        "additional_special_tokens": ["<think>", "</think>", "<scratchpad>", "<verify>"],
    }
    stm_path = os.path.join(SAVE_DIR, "special_tokens_map.json")
    with open(stm_path, "w") as f:
        json.dump(stm, f, indent=2)

    log.info(f"[SAVE] All files written to {SAVE_DIR}/")
    return {"tokenizer_json": tj_path, "config": cfg_path, "stm": stm_path}

# ══════════════════════════════════════════════════════════════════════
# §12  UPLOAD
# ══════════════════════════════════════════════════════════════════════
def upload_to_hub(file_paths: dict) -> None:
    api = HfApi()
    log.info(f"[UPLOAD] Pushing to '{REPO_ID}'...")
    try:
        for _, local_path in file_paths.items():
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=os.path.basename(local_path),
                repo_id=REPO_ID, repo_type="model", token=HF_TOKEN,
            )
            log.info(f"[UPLOAD] ✓ {os.path.basename(local_path)}")
        log.info(f"[UPLOAD] https://huggingface.co/{REPO_ID}")
    except Exception as e:
        log.error(f"[UPLOAD] Failed (files saved locally at {SAVE_DIR}): {e}")

# ══════════════════════════════════════════════════════════════════════
# §13  SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════
def run_smoke_tests() -> None:
    print("\n" + "─" * 60)
    print("  SMOKE TESTS")
    print("─" * 60)

    try:
        Regex(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?[^\s\p{L}]+[\r\n]*|\s*[\r\n]+|\s+""")
        print("  [✓] Regex compiles in Rust engine")
    except Exception as e:
        print(f"  [✗] Regex failed: {e}")
        return

    pre = build_pretokenizer()
    cases = [
        ("4-space indent",   "    def foo():"),
        ("decimal",          "3.14159"),
        ("code expression",  "x += 1"),
        ("snake_case",       "user_id_check"),
        ("#include",         "#include <iostream>"),
    ]
    print()
    for label, text in cases:
        tokens = [chunk for chunk, _ in pre.pre_tokenize_str(text)]
        print(f"  [{label:<18}] {repr(text):<28} → {tokens}")

    assert sanitise_text("") is None
    assert sanitise_text("   ") is None
    assert sanitise_text("abc\x00def") == "abcdef"
    assert sanitise_text("hello") == "hello"
    print("\n  [✓] sanitise_text() assertions pass")

    tokens = [c for c, _ in pre.pre_tokenize_str("3.14159")]
    digits = [t for t in tokens if len(t) == 1 and t.isdigit()]
    assert len(digits) == 6, f"Expected 6 digit tokens, got: {tokens}"
    print("  [✓] '3.14159' → 6 digit tokens ✓")
    print("─" * 60 + "\n")

# ══════════════════════════════════════════════════════════════════════
# §14  VERIFICATION
# ══════════════════════════════════════════════════════════════════════
def run_verification() -> None:
    from transformers import PreTrainedTokenizerFast

    print("\n" + "═" * 68)
    print("  ZENYX-V2  —  VERIFICATION SUITE")
    print("═" * 68)

    tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(SAVE_DIR, "tokenizer.json"))
    tok.add_special_tokens({
        "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
        "unk_token": "<|unk|>",       "pad_token": "<|pad|>",
        "additional_special_tokens": ["<think>", "</think>", "<scratchpad>", "<verify>"],
    })
    vocab = tok.get_vocab()

    cpp = "#include <iostream>\nint main() {\n    for (int i=0;i<4;i++) {\n        std::cout<<i<<\"\\n\";\n    }\n    return 0;\n}"
    cpp_ids = tok.encode(cpp)
    print(f"\n[A] C++ roundtrip      : {'✓' if tok.decode(cpp_ids) == cpp else '✗'}  "
          f"ratio={len(cpp)/len(cpp_ids):.2f} c/tok")

    latex = r"\begin{align}\mathcal{L}&=-\sum y\log\hat{y}\end{align}"
    lat_ids = tok.encode(latex)
    print(f"[B] LaTeX roundtrip    : {'✓' if tok.decode(lat_ids) == latex else '✗'}  "
          f"tokens={len(lat_ids)}")

    print("\n[C] Digit isolation")
    for s in ["$123$", "3.14159", "v2.0.32768", "2024-02-18"]:
        decoded = [tok.decode([i]) for i in tok.encode(s)]
        print(f"    {repr(s):<22} → {decoded}")

    print("\n[D] Special tokens")
    for st in SPECIAL_TOKENS:
        print(f"    {st:<20} → ID {vocab.get(st, 'NOT FOUND')}")

    cot = "<think>\n2+2=4\n</think>\n<verify>\n✓\n</verify>"
    print(f"\n[E] CoT roundtrip      : {'✓' if tok.decode(tok.encode(cot)) == cot else '✗'}")

    combined = cpp + "\n" + latex
    ratio    = len(combined) / len(tok.encode(combined))
    print(f"\n[F] Compression ratio  : {ratio:.2f} c/tok  "
          f"({'✓ ≥5.0' if ratio >= 5.0 else '⚠ <5.0 — normal with small corpus'})")

    print("\n" + "═" * 68 + "\n")

# ══════════════════════════════════════════════════════════════════════
# §15  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 60)
    print(f"  ZENYX-V2  |  vocab={VOCAB_SIZE:,}  |  corpus={TARGET_CORPUS_BYTES/1e9:.1f}GB")
    print("  MODE: TRUE STREAMING  —  No spool file, direct to BPE trainer")
    print("█" * 60 + "\n")

    ckpt = load_checkpoint()
    print_session_status(ckpt)
    check_disk_space()

    # HF setup is best-effort — training proceeds even if token is bad/revoked
    setup_hf_repo()

    if ckpt["training_complete"]:
        log.info("[MAIN] Training already complete — loading saved tokenizer.")
        tj = os.path.join(SAVE_DIR, "tokenizer.json")
        if not os.path.exists(tj):
            raise FileNotFoundError(
                f"training_complete=True but '{tj}' missing. "
                "Delete checkpoint.json and re-run."
            )
        tokenizer = Tokenizer.from_file(tj)
    else:
        tokenizer = train_tokenizer(ckpt)

    file_paths = save_tokenizer(tokenizer)
    upload_to_hub(file_paths)
    run_verification()

    print("\n" + "█" * 60)
    print(f"  COMPLETE  →  https://huggingface.co/{REPO_ID}")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    run_smoke_tests()
    main()
