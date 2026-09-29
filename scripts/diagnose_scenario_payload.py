"""Diz por que a API rejeita o lote do cenario de desvio, sem subir a API.

Valida localmente cada registro da janela do cenario contra o mesmo contrato que a
API usa (``src.data.contract.RawRecord``) e reporta, campo a campo, quantos registros
reprovam e por qual regra. Substitui a leitura do corpo do 422, que o cliente do
cenario descarta ao chamar ``raise_for_status``.

Tambem informa se a correcao de recorte ao dominio (``clip_to_domain``) esta presente
no modulo instalado, o que distingue "a correcao nao foi aplicada" de "a correcao foi
aplicada e ainda ha campo fora do contrato".

Uso
---
    python scripts/diagnose_scenario_payload.py
    python scripts/diagnose_scenario_payload.py --scenario baseline
    python scripts/diagnose_scenario_payload.py --fixture      # sem o dado real
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pydantic import ValidationError

from src.config import get_config
from src.data.contract import RawRecord
from src.data.mapping import get_mapping
from src.monitoring import scenarios as modulo_cenarios
from src.monitoring.scenarios import prepare_windows


def validar(registros: list[dict[str, Any]]) -> tuple[int, Counter, dict[str, Any]]:
    """Valida os registros contra o contrato da API.

    Returns
    -------
    reprovados : int
        Numero de registros com ao menos um erro.
    motivos : collections.Counter
        Contagem por par (campo, regra violada).
    exemplo : dict
        Primeiro erro encontrado, com campo, regra, mensagem e valor recebido.
    """
    reprovados = 0
    motivos: Counter = Counter()
    exemplo: dict[str, Any] = {}
    for registro in registros:
        try:
            RawRecord(**registro)
        except ValidationError as erro:
            reprovados += 1
            for detalhe in erro.errors():
                campo = ".".join(str(p) for p in detalhe["loc"]) or "(raiz)"
                motivos[(campo, detalhe["type"])] += 1
                if not exemplo:
                    exemplo = {
                        "campo": campo,
                        "regra": detalhe["type"],
                        "mensagem": detalhe["msg"],
                        "valor_recebido": registro.get(campo),
                    }
    return reprovados, motivos, exemplo


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="simulated")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--limite", type=int, default=12, help="motivos exibidos")
    args = parser.parse_args(argv)

    tem_recorte = hasattr(modulo_cenarios, "clip_to_domain")
    print(f"Correcao de recorte presente no modulo: {'sim' if tem_recorte else 'NAO'}")
    if not tem_recorte:
        print("  A versao instalada de src/monitoring/scenarios.py e a anterior a "
              "correcao. Descompacte correcao_cenario_online.zip sobre a raiz.")

    cfg = get_config()
    janelas = prepare_windows(
        cfg, args.scenario,
        use_fixture=args.fixture,
        materialize=not args.no_materialize,
    )
    atual: pd.DataFrame = janelas.current
    print(f"\nJanela do cenario '{args.scenario}': {len(atual)} linhas, "
          f"{atual.shape[1]} colunas")

    def relatar(quadro: pd.DataFrame, rotulo: str) -> int:
        registros = [
            {k: (None if pd.isna(v) else float(v)) for k, v in linha.items()}
            for linha in quadro.to_dict("records")
        ]
        reprovados, motivos, exemplo = validar(registros)
        print(f"\n{rotulo}: {reprovados} de {len(registros)} registros reprovam")
        if not reprovados:
            print("  Nenhum campo fora do contrato.")
            return 0
        print(f"  {'campo':28s} {'regra':32s} registros")
        for (campo, regra), total in motivos.most_common(args.limite):
            print(f"  {campo:28s} {regra:32s} {total:>9d}")
        print(f"  primeiro erro: {exemplo}")
        return reprovados

    antes = relatar(atual, "Janela sem recorte")

    if tem_recorte:
        recortada, contagem = modulo_cenarios.clip_to_domain(atual, get_mapping())
        print(f"\nValores recortados ao dominio declarado: {contagem or 'nenhum'}")
        depois = relatar(recortada, "Janela apos o recorte")
        if antes and not depois:
            print("\nO recorte resolve: o envio a API deve passar.")
        elif depois:
            print("\nO recorte NAO resolve sozinho. Os campos acima estao fora do "
                  "contrato por outra razao; me envie esta saida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
