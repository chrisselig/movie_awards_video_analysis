# Speaker Diarization on Windows (WSL2 + 3060 Ti)

Step-by-step guide to run speaker diarization on your Windows desktop with GPU acceleration.

## 1. Install WSL2

Open PowerShell as Administrator:

```powershell
wsl --install
```

Restart your computer. After reboot, Ubuntu will finish setting up. Create a username/password.

## 2. Install NVIDIA CUDA Drivers for WSL

On **Windows** (not inside WSL), install the latest NVIDIA Game Ready or Studio driver from:
https://www.nvidia.com/Download/index.aspx

WSL2 automatically exposes the GPU. You do NOT need to install CUDA inside WSL separately.

Verify inside WSL:

```bash
nvidia-smi
```

You should see your 3060 Ti listed.

## 3. Set Up the Project in WSL

```bash
# Install system dependencies
sudo apt update && sudo apt install -y python3 python3-venv python3-pip ffmpeg git

# Clone the repo
git clone <YOUR_REPO_URL> ~/movie_awards_video_analysis
cd ~/movie_awards_video_analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install pyannote
pip install pyannote.audio
```

## 4. Get Hugging Face Access Token

1. Create account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens and create a new token (Read access)
3. Accept the terms for these models (click each link and accept):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

## 5. Configure Environment

```bash
cd ~/movie_awards_video_analysis

# Copy the .env file from the repo or create it:
cat > .env << 'EOF'
TURSO_DATABASE_URL=libsql://movieawards-chrisselig.aws-us-east-1.turso.io
TURSO_AUTH_TOKEN=<your-turso-token>
HF_TOKEN=<your-huggingface-token>
EOF
```

## 6. Run Diarization

```bash
source venv/bin/activate

# Process all videos that don't have diarization yet
python diarize.py

# Or process a single video to test
python diarize.py --video m1kAMUamEMM

# Re-process everything
python diarize.py --full

# Clean up audio files after processing
python diarize.py --cleanup-audio
```

### Expected Performance

| Hardware | Time per 1hr video |
|----------|-------------------|
| CPU only | ~15-20 minutes |
| 3060 Ti (8GB VRAM) | ~1-2 minutes |

For 124 videos, GPU processing should take ~2-4 hours total.

## 7. Verify Results

```bash
# After diarization, re-run analysis to include speaker data
python analyze_data.py

# The data.json in public/ will now include per-speaker breakdowns
```

## 8. Regenerate and Deploy

```bash
# Run the full pipeline
make pipeline

# Or just the analysis
make analyze
```

## Troubleshooting

### "CUDA out of memory"
The 3060 Ti has 8GB VRAM. If you hit memory issues on very long videos:
```bash
# Process one video at a time
python diarize.py --video <VIDEO_ID> --cleanup-audio
```

### "No module named torch"
Make sure you installed PyTorch with CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "Permission denied" for Hugging Face models
You need to accept the model terms on the Hugging Face website (Step 4 above).

### nvidia-smi not found in WSL
Update your Windows NVIDIA driver to the latest version. WSL2 CUDA support requires driver 510+.
