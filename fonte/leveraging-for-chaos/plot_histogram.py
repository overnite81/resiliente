#!/usr/bin/env python3
"""Plota o histograma CSV produzido pelo HistogramCalculator.

Uso:
    python3 plot_histogram.py histograma.csv
    python3 plot_histogram.py histograma.csv --output histograma.png --title "Histograma"

O CSV pode conter os valores em uma única linha ou em várias linhas,
separados por vírgula, espaço ou ponto e vírgula.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_histogram(path: Path | str) -> list[int]:
    """Lê os valores inteiros de um CSV ou da entrada padrão quando path == "-"."""
    values: list[int] = []
    stream = sys.stdin if path == "-" else None

    try:
        if stream is None:
            stream = Path(path).open("r", encoding="utf-8", newline="")
        for line_number, line in enumerate(stream, start=1):
                normalized = line.replace(";", ",").replace(" ", ",")
                for row in csv.reader([normalized], skipinitialspace=True):
                    for token in row:
                        token = token.strip()
                        if not token:
                            continue
                        try:
                            value = int(token)
                        except ValueError as error:
                            raise ValueError(
                                f"valor invalido na linha {line_number}: {token!r}"
                            ) from error
                        if value < 0:
                            raise ValueError(
                                f"o histograma nao pode conter valores negativos: {value}"
                            )
                        values.append(value)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"arquivo nao encontrado: {path}") from error
    finally:
        if stream is not None and stream is not sys.stdin:
            stream.close()

    if not values:
        raise ValueError("o CSV esta vazio")

    return values


def plot_histogram(values: list[int], output: Path, title: str, show: bool) -> None:
    """Gera e salva o grafico do histograma."""
    positions = list(range(len(values)))

    figure_width = max(8.0, min(20.0, len(values) * 0.08))
    figure, axis = plt.subplots(figsize=(figure_width, 5.5))
    axis.bar(positions, values, width=1.0, edgecolor="black", linewidth=0.25)
    axis.set_title(title)
    axis.set_xlabel("Posicao na reta real")
    axis.set_ylabel("Contagem")
    axis.set_xlim(-0.5, len(values) - 0.5)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    print(f"Grafico salvo em: {output}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plota o CSV produzido pelo HistogramCalculator."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="arquivo CSV do histograma; use '-' para ler da entrada padrao",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="arquivo de imagem de saida (padrao: mesmo nome do CSV com .png)",
    )
    parser.add_argument(
        "--title",
        default="Histograma",
        help="titulo do grafico",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="tambem abre uma janela com o grafico",
    )
    args = parser.parse_args()

    values = read_histogram(args.csv_file)
    output = args.output or Path("histograma.png") if args.csv_file == "-" else args.output or Path(args.csv_file).with_suffix(".png")
    plot_histogram(values, output, args.title, args.show)


if __name__ == "__main__":
    main()

