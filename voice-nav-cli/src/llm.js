import axios from 'axios';

/**
 * Calls a local LLM (Ollama) to infer the navigation intent from a transcript.
 * Returns one of: 'vestibular', 'cultura', 'ciencias', 'idiomas', or 'NONE'.
 */
export async function inferIntent(transcript) {
  // Load prompt template (same as used in the extension) – keep it inline for simplicity.
  const prompt = `Você é um assistente da PUC. Dado o texto falado abaixo, identifique a intenção de navegação. Responda apenas com uma das palavras: vestibular, cultura, ciencias, idiomas, ou NONE se não houver correlação.\n\nTexto: """${transcript}"""`;

  const response = await axios.post('http://localhost:11434/api/generate', {
    model: 'llama3',
    prompt,
    stream: false
  }, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.data || typeof response.data.response !== 'string') {
    throw new Error('LLM did not return a valid response');
  }
  const answer = response.data.response.trim().toLowerCase();
  const valid = ['vestibular', 'cultura', 'ciencias', 'idiomas'];
  return valid.includes(answer) ? answer : 'NONE';
}
