"""All tunable settings in one place. Every other module imports from here,
so a judge can see every knob the prototype has in one file."""
import os

# Language model served by Ollama (local service on localhost:11434).
# NOTE: qwen2.5:3b is under the Qwen Research License (non-commercial). See README.
LLM = "qwen2.5:3b"

# Embedding model served by Ollama. Used with "search_document: " / "search_query: " prefixes.
EMBED = "nomic-embed-text"

# faster-whisper model. small.en is English-only.
WHISPER = "small.en"

# Where faster-whisper runs. Default is CPU, which always works.
# Switch to "cuda" only after the GPU check in the README passes.
# You can also override without editing this file:  $env:MS_DEVICE = "cuda"
DEVICE = os.environ.get("MS_DEVICE", "cpu")

# Number format for faster-whisper. CPU: "int8". CUDA: "int8_float16" (or "float16").
COMPUTE = os.environ.get("MS_COMPUTE", "int8" if DEVICE == "cpu" else "int8_float16")

# Windowing (seconds). Starting points to tune on your own clip, not tested values.
MIN_LEN = 15.0   # a window is never cut before it is this long (unless the clip ends)
MAX_LEN = 35.0   # a window is always cut once it is this long
PAUSE = 1.0      # a silence at least this long counts as a natural cut point
OVERLAP = 5.0    # seconds of the previous window passed as read-only context

# Retrieval
TOP_K = 3

# Cosine score below which the system abstains instead of answering.
# PLACEHOLDER: 0.0 means it never abstains. Set it from eval.py output (README, step 6).
THRESHOLD = 0.643  # tuned by tune.py on this repo's eval data (same questions as the results)

# Cosine score at or above which two gap questions are grouped together.
# PLACEHOLDER: 0.0 groups almost everything. Set it from paired questions (README, step 6).
GAP_SIM = 0.645  # tuned by tune.py on this repo's eval data (same questions as the results)
