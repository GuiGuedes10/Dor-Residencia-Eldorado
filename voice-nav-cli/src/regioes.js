/**
 * Regiões, estados e instituições federais (UF = universidades federais, IF = institutos federais).
 * ATENÇÃO: URLs são os domínios institucionais conhecidos; confirme com `npm run map -- <url>`.
 * Estados sem entrada aqui caem em "informe a URL".
 */
export const REGIOES = {
  Norte: ['AM', 'PA', 'AC', 'RO', 'RR', 'AP', 'TO'],
  Nordeste: ['BA', 'CE', 'PE', 'MA', 'PB', 'RN', 'AL', 'SE', 'PI'],
  'Centro-Oeste': ['DF', 'GO', 'MT', 'MS'],
  Sudeste: ['SP', 'RJ', 'MG', 'ES'],
  Sul: ['PR', 'SC', 'RS'],
};

export const FEDERAIS = {
  uf: {
    AM: [{ nome: 'UFAM', url: 'https://ufam.edu.br' }],
    PA: [{ nome: 'UFPA', url: 'https://portal.ufpa.br' }],
    BA: [{ nome: 'UFBA', url: 'https://ufba.br' }],
    CE: [{ nome: 'UFC', url: 'https://www.ufc.br' }],
    PE: [{ nome: 'UFPE', url: 'https://www.ufpe.br' }],
    DF: [{ nome: 'UnB', url: 'https://www.unb.br' }],
    GO: [{ nome: 'UFG', url: 'https://www.ufg.br' }],
    MT: [{ nome: 'UFMT', url: 'https://www.ufmt.br' }],
    MS: [{ nome: 'UFMS', url: 'https://www.ufms.br' }],
    SP: [{ nome: 'UFSCar', url: 'https://www.ufscar.br' }, { nome: 'UNIFESP', url: 'https://www.unifesp.br' }, { nome: 'UFABC', url: 'https://www.ufabc.edu.br' }],
    RJ: [{ nome: 'UFRJ', url: 'https://ufrj.br' }, { nome: 'UFF', url: 'https://www.uff.br' }],
    MG: [{ nome: 'UFMG', url: 'https://ufmg.br' }, { nome: 'UFOP', url: 'https://ufop.br' }],
    ES: [{ nome: 'UFES', url: 'https://www.ufes.br' }],
    PR: [{ nome: 'UFPR', url: 'https://www.ufpr.br' }],
    SC: [{ nome: 'UFSC', url: 'https://ufsc.br' }],
    RS: [{ nome: 'UFRGS', url: 'https://www.ufrgs.br' }],
  },
  if: {
    AM: [{ nome: 'IFAM', url: 'https://www.ifam.edu.br' }],
    PA: [{ nome: 'IFPA', url: 'https://ifpa.edu.br' }],
    BA: [{ nome: 'IFBA', url: 'https://portal.ifba.edu.br' }],
    CE: [{ nome: 'IFCE', url: 'https://ifce.edu.br' }],
    PE: [{ nome: 'IFPE', url: 'https://www.ifpe.edu.br' }],
    DF: [{ nome: 'IFB', url: 'https://www.ifb.edu.br' }],
    GO: [{ nome: 'IFG', url: 'https://www.ifg.edu.br' }],
    MT: [{ nome: 'IFMT', url: 'https://www.ifmt.edu.br' }],
    MS: [{ nome: 'IFMS', url: 'https://www.ifms.edu.br' }],
    SP: [{ nome: 'IFSP', url: 'https://www.ifsp.edu.br' }],
    RJ: [{ nome: 'IFRJ', url: 'https://www.ifrj.edu.br' }],
    MG: [{ nome: 'IFMG', url: 'https://www.ifmg.edu.br' }],
    ES: [{ nome: 'IFES', url: 'https://www.ifes.edu.br' }],
    PR: [{ nome: 'IFPR', url: 'https://ifpr.edu.br' }],
    SC: [{ nome: 'IFSC', url: 'https://www.ifsc.edu.br' }],
    RS: [{ nome: 'IFRS', url: 'https://ifrs.edu.br' }],
  },
};
