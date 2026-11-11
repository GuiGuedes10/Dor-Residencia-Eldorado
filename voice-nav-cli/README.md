# Voice‑Nav CLI

A tiny Node.js command‑line tool that reproduces the same **voice‑to‑navigation** logic that the Chrome extension uses, but runs entirely in the terminal. It:

1. Captures audio from the default microphone.
2. Sends the audio to a **speech‑to‑text** service (you can use the local Ollama `whisper` model or any HTTP STT endpoint).
3. Passes the transcribed text to a **local LLM** (Ollama, LM Studio, etc.) that extracts the navigation intent.
4. Prints the target URL (or shows the four fallback options).

## Prerequisites
- **Node.js ≥ 20** (download from https://nodejs.org/)
- **Ollama** (or any compatible LLM server) running locally on port 11434. Install with `brew install ollama && ollama serve`.
- **ffmpeg** (used by `node-record-lpcm16` to capture audio). Install with `brew install ffmpeg`.
- The **TensorFlow‑JS page classifier** model file (`model.json` + `weights.bin`). Copy it into `voice‑nav-cli/model/page_classifier/` (see step 3).

## Installation
```bash
# Clone / create the folder (already present in the workspace)
cd /Users/carlosarturguimaraes/Documents/reia/voice-nav-cli
# Install dependencies
npm install
```

## Running
```bash
npm start   # or: node index.js
```
Speak a command such as “quero o calendário de vestibular”. The tool will output the URL it intends to navigate to, e.g.
```
➡️ https://www.puc.edu.br/vestibular
```
If the intent cannot be resolved, the four fallback options are shown.

## How it works (high‑level)
- `index.js` orchestrates the pipeline:
  - `recordAudio()` → audio buffer
  - `transcribe(audio)` → text (calls Ollama `whisper` model)
  - `inferIntent(text)` → intent (calls Ollama LLM with a prompt)
  - `routeTo(intent)` → URL (same mapping as the extension)
  - `classifyCurrentPage()` (optional) – loads the TensorFlow‑JS classifier to know the current page context when you run the CLI inside a browser‑like environment (e.g., using `open -a Safari <url>` before the CLI).

The source code is in the `src/` folder.

---

Feel free to adapt the prompt in `src/llm/prompt.txt` or the model paths.
