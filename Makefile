.PHONY: extract analyze diarize pipeline deploy serve

# Run the full pipeline: extract new videos, analyze, generate data.json
pipeline: extract analyze

# Extract transcripts for new videos only
extract:
	source venv/bin/activate && python -u extract_data.py

# Full re-extraction (all videos)
extract-full:
	source venv/bin/activate && python -u extract_data.py --full

# Run analysis and generate data.json
analyze:
	source venv/bin/activate && python -u analyze_data.py

# Run speaker diarization (requires ffmpeg + HF_TOKEN)
diarize:
	source venv/bin/activate && python -u diarize.py

# Local preview
serve:
	cd public && python3 -m http.server 8080

# Deploy to Vercel
deploy:
	cd public && npx vercel --prod
