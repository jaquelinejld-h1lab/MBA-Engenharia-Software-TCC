"""Executa o notebook de relatorio sem Jupyter e, opcionalmente, gera um HTML.

Le `notebooks/relatorio_resultados.ipynb`, executa as celulas de codigo em ordem
no mesmo espaco de nomes e grava as saidas em `evidencias/relatorio_resultados/`.
Com `--html`, produz tambem um arquivo unico e autocontido, com as figuras
embutidas, que abre em qualquer navegador sem servidor e sem autenticacao.

Motivacao
---------
A imagem de treino nao inclui `jupyterlab` nem `nbconvert`, e acrescenta-los exige
alterar o lock de treino, o que muda o ambiente que produz as metricas de
fidelidade. Este script depende apenas da biblioteca padrao mais o que o proprio
notebook ja usa (pandas, numpy, scikit-learn, matplotlib, joblib), de modo que a
geracao das evidencias nao altera o ambiente travado.

Uso
---
    python scripts/run_relatorio.py                  # executa e grava as evidencias
    python scripts/run_relatorio.py --html           # tambem gera Relatorio.html
    python scripts/run_relatorio.py --ate 12         # para na celula 12, para depurar
    python scripts/run_relatorio.py --sem-codigo     # HTML sem os blocos de codigo

Codigo de saida
---------------
0 quando todas as celulas executam, 1 quando alguma levanta excecao nao tratada.
As pendencias registradas pelo proprio notebook nao alteram o codigo de saida:
sao resultado esperado, nao falha de execucao.
"""

from __future__ import annotations

import argparse
import base64
import builtins
import html
import io
import json
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Localizacao do repositorio
# --------------------------------------------------------------------------- #


def localizar_raiz(inicio: Path | None = None) -> Path:
    """Localiza a raiz do repositorio subindo a arvore de diretorios.

    Parameters
    ----------
    inicio : Path or None, optional
        Diretorio de partida. Quando ausente, usa o diretorio deste arquivo.

    Returns
    -------
    Path
        Primeiro ancestral que contenha `configs/` e `src/`.

    Raises
    ------
    FileNotFoundError
        Quando nenhum ancestral satisfaz o criterio.
    """
    atual = (inicio or Path(__file__).resolve().parent).resolve()
    for candidato in [atual, *atual.parents]:
        if (candidato / "configs").is_dir() and (candidato / "src").is_dir():
            return candidato
    raise FileNotFoundError(f"Raiz do repositorio nao encontrada a partir de {atual}.")


# --------------------------------------------------------------------------- #
# Conversao de Markdown, subconjunto suficiente para as celulas deste notebook
# --------------------------------------------------------------------------- #


def _inline(texto: str) -> str:
    """Converte marcacao de linha: negrito, enfase e codigo literal."""
    escapado = html.escape(texto)
    escapado = re.sub(r"`([^`]+)`", r"<code>\1</code>", escapado)
    escapado = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escapado)
    escapado = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escapado)
    return escapado


def _tabela(linhas: list[str]) -> str:
    """Converte um bloco de tabela em canos para HTML."""
    def celulas(linha: str) -> list[str]:
        return [c.strip() for c in linha.strip().strip("|").split("|")]

    cabecalho = celulas(linhas[0])
    corpo = [celulas(linha) for linha in linhas[2:]]
    partes = ["<table><thead><tr>"]
    partes += [f"<th>{_inline(c)}</th>" for c in cabecalho]
    partes.append("</tr></thead><tbody>")
    for linha in corpo:
        partes.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in linha) + "</tr>")
    partes.append("</tbody></table>")
    return "".join(partes)


def markdown_para_html(texto: str) -> str:
    """Converte o subconjunto de Markdown usado no notebook para HTML.

    Suporta titulos de nivel 1 a 4, tabelas em canos, listas nao ordenadas e
    ordenadas, blocos de codigo cercados, negrito, enfase e codigo literal.
    Qualquer outra construcao cai em paragrafo.

    Parameters
    ----------
    texto : str
        Conteudo da celula de Markdown.

    Returns
    -------
    str
        Fragmento HTML.
    """
    saida: list[str] = []
    linhas = texto.split("\n")
    indice = 0
    while indice < len(linhas):
        linha = linhas[indice]
        despida = linha.strip()

        if not despida:
            indice += 1
            continue

        if despida.startswith("```"):
            bloco: list[str] = []
            indice += 1
            while indice < len(linhas) and not linhas[indice].strip().startswith("```"):
                bloco.append(linhas[indice])
                indice += 1
            indice += 1
            saida.append(f"<pre class='codigo'>{html.escape(chr(10).join(bloco))}</pre>")
            continue

        titulo = re.match(r"^(#{1,4})\s+(.*)$", despida)
        if titulo:
            nivel = len(titulo.group(1))
            saida.append(f"<h{nivel}>{_inline(titulo.group(2))}</h{nivel}>")
            indice += 1
            continue

        proxima = linhas[indice + 1].strip() if indice + 1 < len(linhas) else ""
        if despida.startswith("|") and set(proxima) <= set("|-: "):
            bloco = []
            while indice < len(linhas) and linhas[indice].strip().startswith("|"):
                bloco.append(linhas[indice])
                indice += 1
            saida.append(_tabela(bloco))
            continue

        if re.match(r"^[-*]\s+", despida) or re.match(r"^\d+\.\s+", despida):
            ordenada = bool(re.match(r"^\d+\.\s+", despida))
            itens: list[str] = []
            while indice < len(linhas):
                item = linhas[indice].strip()
                achado = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", item)
                if not achado:
                    break
                itens.append(f"<li>{_inline(achado.group(1))}</li>")
                indice += 1
            etiqueta = "ol" if ordenada else "ul"
            saida.append(f"<{etiqueta}>{''.join(itens)}</{etiqueta}>")
            continue

        paragrafo: list[str] = []
        while indice < len(linhas) and linhas[indice].strip():
            paragrafo.append(linhas[indice].strip())
            indice += 1
        saida.append(f"<p>{_inline(' '.join(paragrafo))}</p>")

    return "\n".join(saida)


# --------------------------------------------------------------------------- #
# Coleta das saidas de cada celula
# --------------------------------------------------------------------------- #


class Coletor:
    """Acumula as saidas exibidas por celula, para montagem posterior do HTML."""

    def __init__(self) -> None:
        self.atual: list[tuple[str, str]] = []

    def novo_bloco(self) -> None:
        """Inicia a coleta de uma nova celula."""
        self.atual = []

    def adicionar(self, tipo: str, conteudo: str) -> None:
        """Registra uma saida da celula corrente.

        Parameters
        ----------
        tipo : {'html', 'texto', 'imagem'}
            Natureza do conteudo, usada na montagem do HTML.
        conteudo : str
            Fragmento HTML, texto puro, ou PNG em base64.
        """
        self.atual.append((tipo, conteudo))


class Espelho(io.TextIOBase):
    """Duplica a escrita em dois destinos, para manter o log no terminal."""

    def __init__(self, primario: Any, secundario: Any) -> None:
        self._primario = primario
        self._secundario = secundario

    def write(self, texto: str) -> int:
        """Escreve nos dois destinos e devolve o numero de caracteres."""
        self._primario.write(texto)
        self._secundario.write(texto)
        return len(texto)

    def flush(self) -> None:
        """Descarrega os dois destinos."""
        self._primario.flush()
        self._secundario.flush()


def construir_exibidor(coletor: Coletor) -> Any:
    """Cria o substituto de `IPython.display.display` ligado a um coletor.

    DataFrames viram tabela HTML, figuras do matplotlib viram PNG embutido em
    base64, objetos `Markdown` do IPython sao convertidos, e o restante cai em
    texto pre-formatado.
    """
    def exibir(*objetos: Any) -> None:
        for objeto in objetos:
            tipo = type(objeto).__name__

            if tipo == "Figure":
                reserva = io.BytesIO()
                objeto.savefig(reserva, format="png", dpi=110,
                               bbox_inches="tight", facecolor="white")
                coletor.adicionar(
                    "imagem", base64.b64encode(reserva.getvalue()).decode("ascii")
                )
                continue

            if tipo in {"DataFrame", "Series"}:
                quadro = objeto.to_frame() if tipo == "Series" else objeto
                coletor.adicionar(
                    "html", quadro.to_html(index=False, border=0, na_rep="")
                )
                print(quadro.to_string(index=False))
                continue

            dados = getattr(objeto, "data", None)
            if isinstance(dados, str):
                coletor.adicionar("html", markdown_para_html(dados))
                print(dados)
                continue

            coletor.adicionar("texto", str(objeto))
            print(objeto)

    return exibir


# --------------------------------------------------------------------------- #
# Montagem do HTML
# --------------------------------------------------------------------------- #

ESTILO = """
:root { color-scheme: light; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       max-width: 62rem; margin: 0 auto; padding: 2rem 1.25rem 6rem;
       line-height: 1.55; color: #1b1b1f; background: #fff; }
h1 { font-size: 1.9rem; border-bottom: 2px solid #e3e3e8; padding-bottom: .4rem; }
h2 { font-size: 1.4rem; margin-top: 2.4rem; border-bottom: 1px solid #ececf1;
     padding-bottom: .3rem; }
h3 { font-size: 1.12rem; margin-top: 1.6rem; }
table { border-collapse: collapse; margin: .9rem 0; font-size: .86rem;
        display: block; overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid #dcdce3; padding: .35rem .6rem; text-align: left;
         white-space: nowrap; }
th { background: #f5f5f8; font-weight: 600; }
tr:nth-child(even) td { background: #fbfbfd; }
pre { background: #f7f7fa; border: 1px solid #e6e6ec; border-radius: 6px;
      padding: .75rem .9rem; overflow-x: auto; font-size: .8rem; line-height: 1.45; }
pre.saida { background: #fcfcfd; color: #333; }
code { background: #f2f2f6; padding: .1rem .3rem; border-radius: 3px;
       font-size: .86em; }
pre code { background: none; padding: 0; }
img { max-width: 100%; height: auto; display: block; margin: 1rem 0;
      border: 1px solid #ececf1; border-radius: 4px; }
details { margin: .6rem 0; }
summary { cursor: pointer; color: #555; font-size: .82rem; user-select: none; }
.celula { margin: 1.1rem 0 1.8rem; }
.aviso { background: #fff6e5; border-left: 4px solid #e0a03a; padding: .7rem .9rem;
         border-radius: 0 4px 4px 0; margin: 1rem 0; font-size: .9rem; }
"""


def montar_html(blocos: list[dict[str, Any]], titulo: str, com_codigo: bool) -> str:
    """Monta o documento HTML autocontido a partir dos blocos coletados.

    Parameters
    ----------
    blocos : list of dict
        Sequencia de blocos, cada um com 'tipo' e o conteudo correspondente.
    titulo : str
        Titulo da pagina.
    com_codigo : bool
        Quando verdadeiro, inclui o codigo de cada celula em bloco recolhivel.

    Returns
    -------
    str
        Documento HTML completo.
    """
    corpo: list[str] = []
    for bloco in blocos:
        if bloco["tipo"] == "markdown":
            corpo.append(markdown_para_html(bloco["fonte"]))
            continue

        partes = ["<div class='celula'>"]
        if com_codigo:
            partes.append(
                "<details><summary>codigo</summary>"
                f"<pre class='codigo'>{html.escape(bloco['fonte'])}</pre></details>"
            )
        texto = bloco["stdout"].strip()
        if texto:
            partes.append(f"<pre class='saida'>{html.escape(texto)}</pre>")
        for tipo, conteudo in bloco["saidas"]:
            if tipo == "imagem":
                partes.append(f"<img alt='figura' src='data:image/png;base64,{conteudo}'>")
            elif tipo == "html":
                partes.append(conteudo)
            else:
                partes.append(f"<pre class='saida'>{html.escape(conteudo)}</pre>")
        if bloco.get("erro"):
            partes.append(
                f"<div class='aviso'><strong>Falha nesta celula</strong>"
                f"<pre>{html.escape(bloco['erro'])}</pre></div>"
            )
        partes.append("</div>")
        corpo.append("".join(partes))

    return (
        "<!doctype html>\n<html lang='pt-BR'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{html.escape(titulo)}</title>\n<style>{ESTILO}</style>\n"
        "</head>\n<body>\n" + "\n".join(corpo) + "\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# Execucao
# --------------------------------------------------------------------------- #


def main() -> int:
    """Ponto de entrada. Devolve o codigo de saida do processo."""
    analisador = argparse.ArgumentParser(
        description="Executa o notebook de resultados sem Jupyter."
    )
    analisador.add_argument("--notebook", type=Path, default=None,
                            help="Padrao: notebooks/relatorio_resultados.ipynb")
    analisador.add_argument("--html", action="store_true",
                            help="Gera Relatorio.html autocontido, sem servidor.")
    analisador.add_argument("--sem-codigo", action="store_true",
                            help="Omite os blocos de codigo no HTML.")
    analisador.add_argument("--ate", type=int, default=None,
                            help="Executa apenas ate esta celula de codigo.")
    argumentos = analisador.parse_args()

    raiz = localizar_raiz()
    caminho = argumentos.notebook or (raiz / "notebooks" / "relatorio_resultados.ipynb")
    if not caminho.is_file():
        print(f"ERRO: notebook nao encontrado em {caminho}", file=sys.stderr)
        return 1

    notebook = json.loads(caminho.read_text(encoding="utf-8"))
    coletor = Coletor()
    exibir = construir_exibidor(coletor)
    builtins.display = exibir  # type: ignore[attr-defined]
    escopo: dict[str, Any] = {"__name__": "__main__", "display": exibir}

    blocos: list[dict[str, Any]] = []
    falhas = 0
    executadas = 0
    total = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")

    for celula in notebook["cells"]:
        fonte = "".join(celula["source"])

        if celula["cell_type"] == "markdown":
            blocos.append({"tipo": "markdown", "fonte": fonte})
            continue

        if argumentos.ate is not None and executadas >= argumentos.ate:
            break

        executadas += 1
        print(f"\n{'=' * 72}\nCELULA {executadas} de {total}\n{'=' * 72}")
        coletor.novo_bloco()
        capturado = io.StringIO()
        erro = ""
        try:
            with redirect_stdout(Espelho(sys.__stdout__, capturado)):
                exec(compile(fonte, f"<celula {executadas}>", "exec"), escopo)
        except Exception:
            falhas += 1
            erro = traceback.format_exc()
            print(f"### FALHA NA CELULA {executadas}", file=sys.stderr)
            print(erro, file=sys.stderr)

        blocos.append({"tipo": "codigo", "fonte": fonte,
                       "stdout": capturado.getvalue(),
                       "saidas": list(coletor.atual), "erro": erro})

    if argumentos.html:
        destino = raiz / "evidencias" / "relatorio_resultados" / "Relatorio.html"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            montar_html(blocos, "Relatorio de Resultados do TCC",
                        com_codigo=not argumentos.sem_codigo),
            encoding="utf-8",
        )
        tamanho = destino.stat().st_size / 1024
        print(f"\nHTML autocontido: {destino} ({tamanho:.0f} KB)")
        print("Abra com duplo clique. Nao precisa de servidor nem de token.")

    print(f"\n{'=' * 72}")
    print(f"{executadas - falhas} de {executadas} celulas executadas.")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
