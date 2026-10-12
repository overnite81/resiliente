#!/usr/bin/env python3

import argparse
import csv
from bisect import bisect_right
from pathlib import Path

from PIL import Image


DEFAULT_OUTPUT = "ctm_output.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous Triangular Model (CTM) plotter for CSV points."
    )
    parser.add_argument("csv_path", help="CSV file with columns: x,y")
    parser.add_argument("span", type=int, help="grid size, e.g. 256")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output image file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the generated image in the default viewer",
    )
    return parser.parse_args()


def load_points(csv_path):
    points = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row_index, row in enumerate(reader, start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 2:
                raise ValueError(f"Linha {row_index}: espera 2 colunas (x,y), recebeu {len(row)}.")
            try:
                x = float(row[0])
                y = float(row[1])
            except ValueError as exc:
                raise ValueError(f"Linha {row_index}: valores numéricos inválidos: {row}") from exc
            points.append((x, y))

    if not points:
        raise ValueError("CSV vazio ou sem pontos válidos.")

    points.sort()
    for index in range(1, len(points)):
        if points[index][0] == points[index - 1][0]:
            raise ValueError("As coordenadas x devem ser únicas.")

    return points


class PiecewiseLinearIntegral:
    def __init__(self, points):
        self.x_values = [x for x, _ in points]
        self.y_values = [y for _, y in points]
        self.prefix_areas = [0.0]

        for index in range(len(points) - 1):
            width = self.x_values[index + 1] - self.x_values[index]
            area = width * (self.y_values[index] + self.y_values[index + 1]) / 2.0
            self.prefix_areas.append(self.prefix_areas[-1] + area)

    def _value_at(self, x):
        if x <= self.x_values[0]:
            return self.y_values[0]
        if x >= self.x_values[-1]:
            return self.y_values[-1]

        right = bisect_right(self.x_values, x)
        left = right - 1
        fraction = (x - self.x_values[left]) / (self.x_values[right] - self.x_values[left])
        return self.y_values[left] + fraction * (self.y_values[right] - self.y_values[left])

    def area_until(self, x):
        if x <= self.x_values[0]:
            return 0.0
        if x >= self.x_values[-1]:
            return self.prefix_areas[-1]

        right = bisect_right(self.x_values, x)
        left = right - 1
        width = x - self.x_values[left]
        start_value = self.y_values[left]
        end_value = self._value_at(x)
        partial_area = width * (start_value + end_value) / 2.0
        return self.prefix_areas[left] + partial_area

    def area(self, start, end):
        return self.area_until(end) - self.area_until(start)


def build_ctm_grid(points, span):
    if span <= 0:
        raise ValueError("span deve ser maior que zero.")

    xs = [x for x, _ in points]
    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        raise ValueError("O domínio x deve possuir mais de um valor.")

    integral = PiecewiseLinearIntegral(points)
    x_step = (x_max - x_min) / span
    grid = [[None for _ in range(span)] for _ in range(span)]

    # Row i is scale i. It contains i windows, each with span - i base intervals.
    for scale in range(1, span):
        window_count = span - scale
        for position in range(scale):
            start = x_min + position * x_step
            end = x_min + (position + window_count) * x_step
            grid[scale][position] = integral.area(start, end)

    return grid


def normalize_grid(grid):
    flat_values = [value for row in grid for value in row if value is not None]
    min_value = min(flat_values)
    max_value = max(flat_values)
    if max_value == min_value:
        return [[0.0 if value is not None else None for value in row] for row in grid]

    normalized = []
    for row in grid:
        normalized_row = []
        for value in row:
            if value is None:
                normalized_row.append(None)
            else:
                normalized_row.append((value - min_value) / (max_value - min_value) * 255.0)
        normalized.append(normalized_row)

    return normalized


def grid_to_image(grid):
    height = len(grid)
    width = len(grid[0]) if height else 0
    image = Image.new("L", (width, height), color=0)
    pixels = image.load()

    for scale in range(1, height):
        first_column = (width - scale) // 2
        for position in range(scale):
            value = grid[scale][position]
            pixels[first_column + position, scale] = int(max(0, min(255, value)))

    return image


def main():
    args = parse_args()

    points = load_points(args.csv_path)
    grid = build_ctm_grid(points, args.span)
    normalized = normalize_grid(grid)
    image = grid_to_image(normalized)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Imagem salva em: {output_path.resolve()}")

    if args.show:
        image.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Erro: {exc}")
