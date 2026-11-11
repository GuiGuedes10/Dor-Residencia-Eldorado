/**
 * Instituições suportadas. A detecção usa o hostname da URL informada (--site).
 * Para adicionar uma instituição: acrescente um item aqui e rode `npm run map -- <url>`.
 * Máximo de 4 opções por menu (acessibilidade): PUC tem submenu de campus.
 */
export const INSTITUTIONS = [
  {
    id: 'puc',
    nome: 'PUC',
    match: /(^|\.)puc[a-z-]*\.(edu\.)?br$/,
    campi: [
      { id: 'campinas', nome: 'PUC-Campinas', url: 'https://vestibular.puc-campinas.edu.br', neural: true },
      { id: 'rio', nome: 'PUC-Rio', url: 'https://www.puc-rio.br' },
      { id: 'sp', nome: 'PUC-SP', url: 'https://www.pucsp.br' },
      { id: 'rs', nome: 'PUCRS', url: 'https://www.pucrs.br' },
    ],
  },
  { id: 'uf', nome: 'Universidade Federal (UF)', match: /(^|\.)uf[a-z]{1,5}\.(edu\.)?br$/, campi: [{ id: 'ufrj', nome: 'UFRJ', url: 'https://ufrj.br' }] },
  { id: 'if', nome: 'Instituto Federal (IF)', match: /(^|\.)if[a-z]{1,6}\.edu\.br$/, campi: [{ id: 'ifsp', nome: 'IFSP', url: 'https://www.ifsp.edu.br' }] },
  { id: 'fia', nome: 'FIA', match: /(^|\.)fia\.com\.br$/, campi: [{ id: 'fia', nome: 'FIA', url: 'https://www.fia.com.br' }] },
];

export function detectInstitution(url) {
  try {
    const host = new URL(url).hostname;
    return INSTITUTIONS.find((i) => i.match.test(host)) ?? null;
  } catch {
    return null;
  }
}

/** Retorna o campus cujo hostname bate com a URL (para decidir se usa a rede neural da PUC-Campinas). */
export function detectCampus(url) {
  const inst = detectInstitution(url);
  if (!inst) return null;
  const host = new URL(url).hostname;
  return inst.campi.find((c) => new URL(c.url).hostname === host) ?? null;
}
