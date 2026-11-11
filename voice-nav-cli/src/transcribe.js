import axios from 'axios';
import fs from 'fs';
import path from 'path';

/**
 * Sends raw WAV audio buffer to Ollama's Whisper model and returns the transcribed text.
 * Assumes Ollama is running locally on http://localhost:11434.
 */
export async function transcribeAudio(buffer) {
  // Ollama expects multipart/form-data; however the simple HTTP API accepts raw binary for whisper.
  const response = await axios.post('http://localhost:11434/api/generate', {
    model: 'whisper',
    prompt: '',
    stream: false,
    // base64‑encode the wav data as required by Ollama's API (binary is not supported via JSON)
    // The API also accepts a "raw" field, but to keep it simple we embed as base64.
    // The server decodes internally.
    // If you have a custom endpoint that accepts binary, adjust accordingly.
    // Note: In practice you may use the /api/embeddings endpoint; this is a mock.
    // Using `data` field for base64 string.
    data: buffer.toString('base64')
  }, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (response.data && response.data.response) {
    return response.data.response.trim();
  }
  throw new Error('Transcription failed: no response.body');
}
