# Checklist de segurança e privacidade

## 1. Natureza dos dados

| Item | Situação |
|---|---|
| Base | Microdados públicos da PNS 2013 (IBGE), módulo laboratorial, anonimizados na origem |
| Dados pessoais identificáveis | Nenhum: o extrato não contém nome, CPF, endereço, UPA, domicílio, ordem do morador nem município |
| Dados sensíveis (LGPD art. 5º, II) | Sim: saúde (diagnósticos autorreferidos, exames, pressão arterial), cor ou raça. São dados públicos e anonimizados, usados exclusivamente para pesquisa acadêmica |
| Base legal | Tratamento de dados anonimizados de fonte pública para fins acadêmicos (LGPD art. 7º, IV e art. 12); citação obrigatória da fonte |

## 2. Avaliação de risco de reidentificação

| Vetor | Análise | Risco |
|---|---|---|
| Quase-identificadores presentes | região (5 níveis), idade (anos), sexo, cor ou raça, número de moradores, rendas | |
| Unicidade no extrato | A combinação região × idade × sexo × cor produz classes com dezenas de pessoas em quase todo o espaço; casos raros (80+ em regiões menores) ainda somam mais de uma pessoa por classe no extrato de 8.952 linhas e não há chave que ligue o registro a outra base | baixo |
| Ligação externa | Sem UPA, município ou data de entrevista não há junção determinística com outras bases públicas do IBGE | baixo |
| Valores de renda | Valores exatos de renda poderiam ser distintivos; permanecem no extrato porque o modelo os consome apenas em faixas de salário mínimo, e nenhuma saída do sistema expõe registros individuais do treino | baixo |
| Artefatos publicados | `drift_reference.json` guarda, para as variáveis numéricas, os valores do treino ordenados sem qualquer outra coluna: não permite reconstruir linhas | baixo |
| Modelo | Regressão logística com 86 coeficientes; sem memorização de registros | desprezível |

Conclusão: risco de reidentificação baixo; compatível com a publicação do extrato no
repositório, decisão registrada no README com atribuição ao IBGE.

## 3. Dados em produção (API e interface)

| Controle | Implementação |
|---|---|
| Persistência de dados de entrada | Nenhuma em disco. A API mantém uma janela em memória (`monitoring.window_size` observações, sem identificador) para PSI/KS, descartada no reinício |
| Logs | JSON estruturado com correlation id, endpoint, status e duração; **o payload não é logado** |
| Métricas | Agregados Prometheus (contadores, histogramas, PSI/KS por variável); nenhum valor individual |
| Interface | Estado em `st.session_state` da sessão do navegador; uploads de CSV processados em memória e devolvidos como download, sem gravação no servidor |
| Transporte | HTTP local no compose; em qualquer exposição externa é obrigatório TLS e autenticação (fora do escopo local) |
| Credenciais | Nenhuma no código; senha do Grafana por variável de ambiente; GitHub Secrets no CI |
| Imagens | Usuário não root; sem bibliotecas de treino na API; sem modelo na interface |
| Dependências | `pip-audit` bloqueante nos locks da API e da interface; `bandit` no código |

## 4. Política de retenção e descarte

| Ativo | Retenção | Descarte |
|---|---|---|
| Extrato bruto (`data/raw`) | Enquanto o modelo derivado dele for mantido; versionado por hash | Remoção do repositório e do histórico (`git filter-repo`) se o IBGE alterar os termos de uso |
| `data/interim`, `data/processed` | Regenerados a cada execução; ignorados pelo Git | `make clean` ou remoção da pasta |
| Runs do MLflow e `artifacts/` | Enquanto houver versão referenciada no registry; runs arquivadas podem ser apagadas após um ciclo de retreino | `mlflow gc` após arquivar |
| Janela de observações da API | Somente memória; no máximo `window_size` registros | Automático no reinício ou por eviction |
| Uploads na interface | Só durante a sessão | Automático ao fechar a sessão |
| Logs de container | Retenção do Docker (`json-file`); recomendado limitar por `max-size` no compose em uso prolongado | `docker compose down` |
| Séries do Prometheus | 7 dias (`--storage.tsdb.retention.time=7d`) | Automático |

## 5. Ética e equidade

Desempenho avaliado por sexo e faixa etária (model card, seção 5) com disparidade
relevante de sensibilidade; estratégia de mitigação documentada. Disclaimer clínico em
todas as saídas. O sistema não toma decisão automática nem envia dados a terceiros.
