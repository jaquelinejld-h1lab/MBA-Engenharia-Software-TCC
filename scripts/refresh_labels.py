"""Reescreve os rotulos em portugues dos artefatos a partir do mapeamento atual.

Por que existe
--------------
`artifacts/<run>/interpretation.json` e `evidencias/global_importance.json` gravam o
campo `label_pt` no momento do treino. A API serve esses arquivos como estao, de modo
que alterar `configs/variable_mapping.yaml` nao muda os rotulos exibidos na pagina do
modelo ate que o treino seja reexecutado.

Este script e o caminho curto: atualiza apenas os rotulos de exibicao nos artefatos ja
gravados, sem tocar em nenhum numero. O caminho duravel continua sendo reexecutar o
treino, que regera os artefatos inteiros.

`[REPRODUCIBILIDADE]` O script altera artefatos versionados. Rode com a arvore de
trabalho limpa e revise o diff: nenhuma metrica, nenhum coeficiente e nenhum hash pode
aparecer nele.

Uso
---
    python scripts/refresh_labels.py            # aplica
    python scripts/refresh_labels.py --dry-run  # so relata o que mudaria
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mapping import get_mapping  # noqa: E402
from src.paths import PROJECT_ROOT  # noqa: E402

# Campos de exibicao que podem ser reescritos. Qualquer outra chave e preservada.
CAMPOS_ROTULO = ("label_pt",)


def construir_indice() -> tuple[dict[str, str], dict[str, str]]:
    """Monta os mapas de rotulo a partir de `configs/variable_mapping.yaml`.

    Returns
    -------
    por_variavel : dict of str to str
        Rotulo por nome de variavel do modelo, em ingles.
    por_dummy : dict of str to str
        Rotulo por nome de coluna binaria.
    """
    mapping = get_mapping()
    por_variavel: dict[str, str] = {}
    por_dummy: dict[str, str] = {}
    for variavel in mapping.model_variables:
        por_variavel[variavel.en] = variavel.label_pt
        for dummy in getattr(variavel, "dummies", []) or []:
            por_dummy[dummy.en] = variavel.label_pt
    return por_variavel, por_dummy


def atualizar(
    payload: Any, por_variavel: dict[str, str], por_dummy: dict[str, str]
) -> int:
    """Atualiza os rotulos no lugar e devolve quantos foram alterados.

    Percorre a estrutura recursivamente e reescreve `label_pt` sempre que o registro
    identificar a variavel de origem, por `model_variable`, `variable` ou `dummy`.
    Nenhum outro campo e tocado.
    """
    alterados = 0
    if isinstance(payload, dict):
        chave = (
            payload.get("model_variable")
            or payload.get("variable")
            or payload.get("dummy")
        )
        if isinstance(chave, str):
            novo = por_variavel.get(chave) or por_dummy.get(chave)
            for campo in CAMPOS_ROTULO:
                if novo and payload.get(campo) not in (None, novo):
                    payload[campo] = novo
                    alterados += 1
        for valor in payload.values():
            alterados += atualizar(valor, por_variavel, por_dummy)
    elif isinstance(payload, list):
        for item in payload:
            alterados += atualizar(item, por_variavel, por_dummy)
    return alterados


def alvos() -> list[Path]:
    """Lista os artefatos que carregam rotulos de exibicao."""
    encontrados = sorted((PROJECT_ROOT / "artifacts").glob("*/interpretation.json"))
    importancia = PROJECT_ROOT / "evidencias" / "global_importance.json"
    if importancia.is_file():
        encontrados.append(importancia)
    return encontrados


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="apenas relata")
    args = parser.parse_args(argv)

    por_variavel, por_dummy = construir_indice()
    print(f"{len(por_variavel)} variaveis e {len(por_dummy)} colunas binarias no mapeamento")

    total = 0
    arquivos = alvos()
    if not arquivos:
        print("Nenhum artefato com rotulos encontrado. Execute o treino primeiro.")
        return 1

    for caminho in arquivos:
        payload = json.loads(caminho.read_text(encoding="utf-8"))
        alterados = atualizar(payload, por_variavel, por_dummy)
        total += alterados
        relativo = caminho.relative_to(PROJECT_ROOT)
        if alterados and not args.dry_run:
            caminho.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        print(f"  {relativo}: {alterados} rotulos"
              f"{' (simulacao)' if args.dry_run and alterados else ''}")

    print(f"\n{total} rotulos {'seriam alterados' if args.dry_run else 'alterados'} "
          f"em {len(arquivos)} arquivos")
    if total and not args.dry_run:
        print("Revise o diff: nenhuma metrica pode aparecer nele.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
