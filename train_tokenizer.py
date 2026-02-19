# ╔══════════════════════════════════════════════════════════════════════╗
# ║   ZENYX-V2 TOKENIZER TRAINER  —  TRUE STREAMING v7                 ║
# ║   Byte-Level BPE | 32k vocab | 1GB Corpus | Direct Stream Train    ║
# ╚══════════════════════════════════════════════════════════════════════╝

!pip install -U -q transformers datasets tokenizers huggingface_hub

# ══════════════════════════════════════════════════════════════════════
# §0  CONFIG
# ══════════════════════════════════════════════════════════════════════
HF_TOKEN = "hf_toen"
REPO_ID  = "Arko007/zenyx-v2-tokenizer"
SAVE_DIR = "./zenyx_tokenizer"

VOCAB_SIZE          = 32_768
TARGET_CORPUS_BYTES = 1 * 1024 * 1024 * 1024   # 1 GB

MATH_RATIO    = 0.40
CODE_RATIO    = 0.40
ENGLISH_RATIO = 0.20

FINEWEB_MIN_SCORE = 4.8

# Heartbeat: log a progress line every N rows
HEARTBEAT_ROWS = 1_000

CHECKPOINT_FILE = "./zenyx_checkpoint.json"
# NOTE: No spool file — training streams directly from HF datasets.

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
def setup_hf_repo() -> None:
    login(token=HF_TOKEN, add_to_git_credential=False)
    api = HfApi()
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="model", token=HF_TOKEN)
        log.info(f"[HF] Repo '{REPO_ID}' exists.")
    except Exception:
        create_repo(repo_id=REPO_ID, token=HF_TOKEN,
                    repo_type="model", private=False, exist_ok=True)
        log.info(f"[HF] Repo '{REPO_ID}' created.")

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
    print("\n" + "\u2501" * 60)
    print("  ZENYX-V2  \u2014  SESSION STATUS")
    print("\u2501" * 60)
    print(f"  Mode           : TRUE STREAMING (no spool file)")
    print(f"  Training done  : {'\u2713' if ckpt['training_complete'] else '\u2717 (will train)'}")
    if ckpt["vocab_size_achieved"]:
        print(f"  Vocab achieved : {ckpt['vocab_size_achieved']:,}")
    print("\u2501" * 60 + "\n")

# ══════════════════════════════════════════════════════════════════════
# §4  DISK SAFETY
#     Only ~50MB needed for the final tokenizer output files.
# ══════════════════════════════════════════════════════════════════════
def check_disk_space() -> None:
    stat   = os.statvfs(".")
    free   = stat.f_bavail * stat.f_frsize
    needed = 50 * 1024 * 1024  # 50 MB for output files only
    log.info(f"[DISK] Free: {free/1e9:.2f}GB  Needed: ~{needed/1e6:.0f}MB (output only, no spool)")
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
#     Each yields (text, byte_len) with live heartbeat logging.
# ══════════════════════════════════════════════════════════════════════
def stream_finemath(target_bytes: int):
    log.info(f"[MATH] Loading finemath-4plus (streaming)...")
    ds = load_dataset("HuggingFaceTB/finemath", name="finemath-4plus",
                      split="train", streaming=True)
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
    lang_map = {"Python": "Python", "Java": "Java", "C": "C", "C++": "Cpp"}
    streams  = []
    for lang_name, config in lang_map.items():
        try:
            ds = load_dataset("HuggingFaceTB/stack-edu", name=config,
                              split="train", streaming=True)
            streams.append(ds)
            log.info(f"[CODE] Config '{config}' ({lang_name}) \u2713")
        except Exception as e:
            log.warning(f"[CODE] Config '{config}' failed ({e}), trying filter fallback...")
            try:
                ds = load_dataset("HuggingFaceTB/stack-edu", split="train",
                                  streaming=True).filter(
                    lambda x, l=lang_name: x.get("programming_language", "") == l)
                streams.append(ds)
                log.info(f"[CODE] Filter fallback '{lang_name}' \u2713")
            except Exception as e2:
                log.error(f"[CODE] Both attempts failed for '{lang_name}': {e2}")

    if not streams:
        log.error("[CODE] No code streams available.")
        return

    combined  = interleave_datasets(streams, stopping_strategy="first_exhausted")
    log.info(f"[CODE] {len(streams)} stream(s) | target={target_bytes/1e6:.0f}MB")
    consumed  = 0
    row_count = 0
    for row in combined:
        text = sanitise_text(row.get("content", row.get("text", "")))
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
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    log.info(f"[ENG] Dataset ready | score>{FINEWEB_MIN_SCORE} | target={target_bytes/1e6:.0f}MB")
    consumed  = 0
    row_count = 0
    for row in ds:
        if row.get("score", 0.0) <= FINEWEB_MIN_SCORE:
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
#
#     Yields plain text strings directly to the BPE trainer,
#     interleaved across math / code / english in the configured ratio.
#     No spool file, no intermediate disk writes — training starts the
#     moment the first batch of text arrives from the network.
# ══════════════════════════════════════════════════════════════════════
def stream_combined_for_training():
    math_target    = int(TARGET_CORPUS_BYTES * MATH_RATIO)
    code_target    = int(TARGET_CORPUS_BYTES * CODE_RATIO)
    english_target = int(TARGET_CORPUS_BYTES * ENGLISH_RATIO)

    log.info("[STREAM] Initialising streaming generators (may take 1-3 min each)...")
    gen_math    = stream_finemath(math_target)
    gen_code    = stream_stack_edu(code_target)
    gen_english = stream_fineweb_edu(english_target)
    log.info("[STREAM] All generators ready — true streaming begins, training starts immediately.")
    log.info(f"[STREAM] Targets \u2192 math={math_target/1e6:.0f}MB  "
             f"code={code_target/1e6:.0f}MB  eng={english_target/1e6:.0f}MB")

    sources_raw = [
        ("math",    gen_math,    math_target,    4),
        ("code",    gen_code,    code_target,    4),
        ("english", gen_english, english_target, 2),
    ]
    sources = [(n, g, t, w) for n, g, t, w in sources_raw]

    consumed       = {"math": 0, "code": 0, "english": 0}
    done           = {name: False for name, *_ in sources}
    weighted_cycle = []
    for name, gen, tgt, weight in sources:
        weighted_cycle.extend([(name, gen, tgt)] * weight)

    total_consumed = 0
    total_rows     = 0
    idx            = 0

    while total_consumed < TARGET_CORPUS_BYTES:
        if all(done[name] for name, *_ in sources):
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
            pct = 100 * total_consumed / TARGET_CORPUS_BYTES
            log.info(f"[STREAM] {total_consumed/1e6:.0f}MB ({pct:.1f}%) | "
                     f"math={consumed['math']/1e6:.0f}MB  "
                     f"code={consumed['code']/1e6:.0f}MB  "
                     f"eng={consumed['english']/1e6:.0f}MB | "
                     f"rows={total_rows:,}")

        yield text

# ══════════════════════════════════════════════════════════════════════
# §8  PRE-TOKENIZER
#
# Split (canonical GPT-2, Rust-safe, no \p{N}, no lookaheads)
#   → Digits(individual_digits=True)
#   → ByteLevel(add_prefix_space=False, use_regex=False)
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
#      Streams directly from HF datasets via stream_combined_for_training().
#      No spool file — BPE merge learning starts with the very first batch.
#
#      NOTE: train_from_iterator() signature is (iterator, trainer, length).
#            It does NOT accept a batch_size argument — that belongs to the
#            datasets API, not the tokenizers library.
# ══════════════════════════════════════════════════════════════════════
def train_tokenizer(ckpt: dict) -> Tokenizer:
    tok     = build_tokenizer()
    trainer = build_trainer()

    estimated_rows = max(TARGET_CORPUS_BYTES // 200, 1)
    log.info(f"[TRAIN] est_rows={estimated_rows:,} | True streaming — training starts immediately.")

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
    for _, local_path in file_paths.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=REPO_ID, repo_type="model", token=HF_TOKEN,
        )
        log.info(f"[UPLOAD] \u2713 {os.path.basename(local_path)}")
    log.info(f"[UPLOAD] https://huggingface.co/{REPO_ID}")

# ══════════════════════════════════════════════════════════════════════
# §13  SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════
def run_smoke_tests() -> None:
    print("\n" + "\u2500" * 60)
    print("  SMOKE TESTS")
    print("\u2500" * 60)

    try:
        Regex(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?[^\s\p{L}]+[\r\n]*|\s*[\r\n]+|\s+""")
        print("  [\u2713] Regex compiles in Rust engine")
    except Exception as e:
        print(f"  [\u2717] Regex failed: {e}")
        return

    pre = build_pretokenizer()
    cases = [
        ("4-space indent",   "    def foo():"),
        ("8-space indent",   "        return x"),
        ("#include",         "#include <iostream>"),
        ("abc-def",          "abc-def"),
        ("decimal",          "3.14159"),
        ("version string",   "v2.0.32768"),
        ("CJK",              "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8"),
        ("emoji+digits",     "\U0001f680 + 12"),
        ("code expression",  "x += 1"),
    ]
    print()
    for label, text in cases:
        tokens = [chunk for chunk, _ in pre.pre_tokenize_str(text)]
        print(f"  [{label:<18}] {repr(text):<28} \u2192 {tokens}")

    assert sanitise_text("") is None
    assert sanitise_text("   ") is None
    assert sanitise_text("abc\x00def") == "abcdef"
    assert sanitise_text("hello") == "hello"
    print("\n  [\u2713] sanitise_text() assertions pass")

    tokens = [c for c, _ in pre.pre_tokenize_str("3.14159")]
    digits = [t for t in tokens if len(t) == 1 and t.isdigit()]
    assert len(digits) == 6, f"Expected 6 digit tokens, got: {tokens}"
    print("  [\u2713] '3.14159' \u2192 6 digit tokens \u2713")

    tokens = [c for c, _ in pre.pre_tokenize_str("#include")]
    assert not any("#" in t and any(c.isalpha() for c in t) for t in tokens), \
        f"'#include' punct merged into word: {tokens}"
    print(f"  [\u2713] '#include' punct isolated: {tokens}")

    print("\u2500" * 60 + "\n")

# ══════════════════════════════════════════════════════════════════════
# §14  VERIFICATION
# ══════════════════════════════════════════════════════════════════════
def run_verification() -> None:
    from transformers import PreTrainedTokenizerFast

    print("\n" + "\u2550" * 68)
    print("  ZENYX-V2  \u2014  VERIFICATION SUITE")
    print("\u2550" * 68)

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
    print(f"\n[A] C++ roundtrip      : {'\u2713' if tok.decode(cpp_ids) == cpp else '\u2717'}  "
          f"ratio={len(cpp)/len(cpp_ids):.2f} c/tok")

    latex = r"\begin{align}\mathcal{L}&=-\sum y\log\hat{y}\end{align}"
    lat_ids = tok.encode(latex)
    print(f"[B] LaTeX roundtrip    : {'\u2713' if tok.decode(lat_ids) == latex else '\u2717'}  "
          f"tokens={len(lat_ids)}")

    print("\n[C] Digit isolation")
    for s in ["$123$", "3.14159", "v2.0.32768", "2024-02-18"]:
        decoded = [tok.decode([i]) for i in tok.encode(s)]
        print(f"    {repr(s):<22} \u2192 {decoded}")

    print("\n[D] Punctuation separation")
    for s in ["#include", "abc-def", "x+=1"]:
        tokens = [tok.decode([i]) for i in tok.encode(s)]
        bad = any(any(c.isalpha() for c in t) and any(c in "#-+=" for c in t)
                  for t in tokens)
        print(f"    {repr(s):<15} \u2192 {tokens}  {'\u2713' if not bad else '\u26a0 possible merge'}")

    print("\n[E] Indentation")
    for n, label in [(4, "4-space"), (8, "8-space")]:
        ids = tok.encode(" " * n)
        status = f"\u2713 single token (ID={ids[0]})" if len(ids) == 1 else f"\u2717 {len(ids)} tokens"
        print(f"    {label}: {status}")
    py = "def foo():\n    x = 1\n    if x:\n        return x\n"
    print(f"    Python indent roundtrip: {'\u2713' if tok.decode(tok.encode(py)) == py else '\u2717'}")

    print("\n[F] Special tokens")
    for st in SPECIAL_TOKENS:
        print(f"    {st:<20} \u2192 ID {vocab.get(st, 'NOT FOUND')}")

    cot = "<think>\n2+2=4\n</think>\n<verify>\n\u2713\n</verify>"
    print(f"\n[G] CoT roundtrip      : {'\u2713' if tok.decode(tok.encode(cot)) == cot else '\u2717'}")

    print("\n[H] UTF-8 stress")
    for label, s in [("CJK", "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8"),
                     ("Emoji", "\U0001f680\U0001f52c"),
                     ("Math sym", "\u2211\u2202\u2207\u2260")]:
        rt = tok.decode(tok.encode(s)) == s
        print(f"    [{label:<10}] {'\u2713' if rt else '\u2717'}  {repr(s)}")

    cleaned = sanitise_text("abc\x00def")
    print(f"\n[I] Null byte stripped : '{cleaned}'  {'\u2713' if cleaned == 'abcdef' else '\u2717'}")

    combined = cpp + "\n" + latex
    ratio    = len(combined) / len(tok.encode(combined))
    print(f"\n[J] Compression ratio  : {ratio:.2f} c/tok  "
          f"({'\u2713 \u22655.0' if ratio >= 5.0 else '\u26a0 <5.0 \u2014 normal with small corpus'})")

    print("\n" + "\u2550" * 68 + "\n")

# ══════════════════════════════════════════════════════════════════════
# §15  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "\u2588" * 60)
    print(f"  ZENYX-V2  |  vocab={VOCAB_SIZE:,}  |  corpus={TARGET_CORPUS_BYTES/1e9:.1f}GB")
    print("  MODE: TRUE STREAMING  \u2014  no spool, training starts live")
    print("\u2588" * 60 + "\n")

    ckpt = load_checkpoint()
    print_session_status(ckpt)
    check_disk_space()
    setup_hf_repo()

    # ── True streaming: skip spool phase entirely, train directly ─────
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

    print("\n" + "\u2588" * 60)
    print(f"  COMPLETE  \u2192  https://huggingface.co/{REPO_ID}")
    print("\u2588" * 60 + "\n")


if __name__ == "__main__":
    run_smoke_tests()
    main()
