from pathlib import Path

ROOT = Path(__file__).parent.parent

LOCAL_TXT_PATH = ROOT / "data" / "the_bible_long.txt"
TXT_PATH = LOCAL_TXT_PATH if LOCAL_TXT_PATH.exists() else ROOT / "data" / "relativity_einstein.txt"
OUTPUT_PATH = ROOT / "data" / "output"
MEDIA_PATH = ROOT / "media"

# Use a bounded text sample for simulations. Loading a multi-GB text in full
# would exhaust memory before the communication-channel simulation starts.
MAX_TEXT_CHARS = 1_000_000
