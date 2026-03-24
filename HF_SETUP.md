# Hugging Face token (`HF_TOKEN`)

Set a [Hugging Face access token](https://huggingface.co/settings/tokens) so Hub downloads use your account quota (higher rate limits than anonymous requests).

## Option A — `.env` file (recommended)

1. Copy `.env.example` to `.env` in the **repository root**.
2. Replace `hf_your_token_here` with your token.
3. Install deps: `pip install python-dotenv` (listed in `requirements.txt`).

On startup, `hf_token.ensure_hf_environment()` loads `.env`, points **`HF_HOME`** at **`./hf_home/`** (unless you set `HF_HOME` yourself), and sets `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` for `transformers` / `huggingface_hub`.

## Local disk cache (no re-download every run)

1. From the repo root run:

   ```bash
   python download_hf_models.py
   ```

   This snapshots every model listed in `hf_token.ALL_PROJECT_HF_REPOS` into `hf_home/` (or your `HF_HOME`).

2. Normal usage loads from that cache; Hub is only contacted again if files are missing or you update models.

3. Optional strict offline mode after the download — in `.env`:<>

   ```env
   HF_OFFLINE=1
   ```

   Then the Hub is not contacted at all; everything must already be in `HF_HOME`.

## Option B — shell / CI

**PowerShell**

```powershell
$env:HF_TOKEN = "hf_..."
```

**bash**

```bash
export HF_TOKEN=hf_...
```

`.env` is gitignored — do not commit tokens.

## Faster large downloads (optional)

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

(See comment in `.env.example`.)
