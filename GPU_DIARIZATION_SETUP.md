# GPU Diarization Setup (Windows + WSL2 + RTX 3060 Ti)

Step-by-step guide to run `diarize.py` on a Windows desktop with NVIDIA GPU via WSL2. On an RTX 3060 Ti, diarization drops from ~30 min/video (CPU) to ~2-3 min/video.

---

## 1. Install WSL2

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

This installs WSL2 with Ubuntu by default. Restart when prompted.

After restart, Ubuntu will launch and ask you to create a username/password.

Verify WSL2 is working:

```powershell
wsl --list --verbose
```

You should see Ubuntu with VERSION 2.

---

## 2. Install NVIDIA GPU Driver (Windows Side)

Download and install the latest **Game Ready** or **Studio** driver from:
https://www.nvidia.com/Download/index.aspx

Select: RTX 3060 Ti, Windows 10/11, Game Ready Driver.

**Important:** Do NOT install CUDA inside WSL2 separately — the Windows driver provides GPU access to WSL2 automatically.

Verify GPU is visible inside WSL2:

```bash
nvidia-smi
```

You should see your RTX 3060 Ti listed with driver version and CUDA version.

---

## 3. Set Up the Project in WSL2

Open Ubuntu (WSL2) terminal:

```bash
# Install system dependencies
sudo apt update && sudo apt install -y python3 python3-pip python3-venv ffmpeg git

# Clone the repo
git clone https://github.com/chrisselig/movie_awards_video_analysis.git
cd movie_awards_video_analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install base requirements
pip install -r requirements.txt
```

---

## 4. Install PyTorch with CUDA Support

This is the key step — you need the CUDA-enabled PyTorch, not the CPU version:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available to PyTorch:

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Ti
```

If CUDA shows `False`, check:
- Windows NVIDIA driver is up to date
- You installed the `cu121` version of PyTorch (not CPU)
- Run `nvidia-smi` to confirm the GPU is visible

---

## 5. Install pyannote.audio

```bash
pip install pyannote.audio
```

---

## 6. Set Up Hugging Face Access

You need a Hugging Face token with access to the pyannote models.

1. Create a token at: https://huggingface.co/settings/tokens
2. Accept the license agreements for these models (click "Agree" on each page):
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create your `.env` file:

```bash
cp .env.example .env  # if .env.example exists, otherwise create manually
```

Edit `.env` with your credentials:

```
TURSO_DATABASE_URL=libsql://your-db.turso.io
TURSO_AUTH_TOKEN=your-turso-token
HF_TOKEN=hf_your_huggingface_token
GOOGLE_API_KEY=your-google-api-key
```

Ask Justin for the Turso credentials if you don't have them.

---

## 6. Install yt-dlp (for audio download)

```bash
pip install yt-dlp
```

Verify:

```bash
yt-dlp --version
```

---

## 7. Run Diarization

```bash
source venv/bin/activate
python -u diarize.py
```

This will:
1. Find all videos missing diarization data
2. Download audio from YouTube
3. Run speaker diarization on GPU
4. Store results in Turso

Useful flags:

```bash
# Process a single video
python -u diarize.py --video VIDEO_ID

# Re-process all videos
python -u diarize.py --full

# Clean up audio files after processing
python -u diarize.py --cleanup-audio
```

---

## 8. After Diarization: Regenerate Analysis

Once diarization is done (or partially done), regenerate the analysis data:

```bash
python -u analyze_data.py
```

This outputs `public/data.json` with updated head-to-head features.

---

## Troubleshooting

### `nvidia-smi` not found in WSL2
- Update your Windows NVIDIA driver to the latest version
- Restart WSL: `wsl --shutdown` in PowerShell, then reopen Ubuntu

### CUDA available but out of memory
The RTX 3060 Ti has 8GB VRAM. pyannote should fit comfortably, but if you hit OOM:
```bash
# Reduce batch size (not directly configurable in pyannote, but closing other GPU apps helps)
# Close any games/browsers using GPU before running
```

### `pyannote` model download fails
- Ensure your HF_TOKEN is set correctly in `.env`
- Ensure you've accepted the model licenses on the Hugging Face pages listed above

### WSL2 runs out of disk space
WSL2 uses a virtual disk. Expand it if needed:
```powershell
# In PowerShell as Admin
wsl --shutdown
# Find the VHD at: %LOCALAPPDATA%\Packages\CanonicalGroupLimited.Ubuntu*\LocalState\ext4.vhdx
```

### Audio download fails
- Ensure `yt-dlp` and `ffmpeg` are installed
- Some videos may be region-locked or private
