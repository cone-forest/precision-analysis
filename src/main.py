#!/usr/bin/env python3
import argparse
import sys
import numpy as np

from utils import load_poses_csv, df_to_Ts, summarize_errors, print_T
from tsai_lenz import tsai_lenz
from park_martin import park_martin
from daniilidis import daniilidis
from li_wang_wu import li_wang_wu
from shah import shah


DEFAULT_A = "data/calibF/MeasuredPositionsLeica.txt"
DEFAULT_B = "data/calibF/MeasuredPositionsTS_ModelLines.txt"


def run_method(name, As, Bs):
    if name == "tsai-lenz":
        X, Y = tsai_lenz(As, Bs)
    elif name == "park-martin":
        X, Y = park_martin(As, Bs)
    elif name == "daniilidis":
        X, Y = daniilidis(As, Bs)
    elif name == "li-wang-wu":
        X, Y = li_wang_wu(As, Bs)
    elif name == "shah":
        X, Y = shah(As, Bs)
    else:
        raise ValueError(f"Неизвестный метод: {name}")

    t_stats, r_stats = summarize_errors(As, Bs, X, Y)
    return X, Y, t_stats, r_stats


def load_inputs(file_A, file_B):
    dfA = load_poses_csv(file_A)
    dfB = load_poses_csv(file_B)

    As = df_to_Ts(dfA)
    Bs = df_to_Ts(dfB)
    return As, Bs


def print_table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    sep = "+".join("-" * (w + 2) for w in widths)
    line_sep = f"+{sep}+"

    def fmt_row(values):
        return "|" + "|".join(f" {v:<{w}} " for v, w in zip(values, widths)) + "|"

    print(line_sep)
    print(fmt_row(headers))
    print(line_sep)
    for r in rows:
        print(fmt_row(r))
    print(line_sep)


def metric_row(name, stats):
    return [
        name,
        f"{stats['mean']:.4f}",
        f"{stats['median']:.4f}",
        f"{stats['rmse']:.4f}",
        f"{stats['p95']:.4f}",
        f"{stats['max']:.4f}",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение методов AX=YB / AX=XB."
    )
    parser.add_argument(
        "--method",
        "-m",
        default="all",
        help="Метод: tsai-lenz | park-martin | daniilidis | li-wang-wu | shah | all (по умолчанию all)",
    )
    parser.add_argument("--file-a", "-a", default=DEFAULT_A, help=f"Путь к файлу A (по умолчанию {DEFAULT_A})")
    parser.add_argument("--file-b", "-b", default=DEFAULT_B, help=f"Путь к файлу B (по умолчанию {DEFAULT_B})")
    parser.add_argument("--print-xy", action="store_true", help="Печатать матриц X и Y(Z) для выбранного метода")
    args = parser.parse_args()

    method_map = {
        "tsai-lenz": "tsai-lenz",
        "tsai": "tsai-lenz",
        "park-martin": "park-martin",
        "park": "park-martin",
        "daniilidis": "daniilidis",
        "li-wang-wu": "li-wang-wu",
        "li": "li-wang-wu",
        "shah": "shah",
        "all": "all",
        "--all": "all",
    }

    key = method_map.get(args.method.lower(), None)
    if key is None:
        print("Неизвестный метод. Доступно: tsai-lenz | park-martin | daniilidis | li-wang-wu | shah | all")
        sys.exit(1)

    As, Bs = load_inputs(args.file_a, args.file_b)

    headers = ["method", "mean", "median", "rmse", "p95", "max"]

    if key == "all":
        methods = ["tsai-lenz", "park-martin", "daniilidis", "li-wang-wu", "shah"]

        t_rows = []
        r_rows = []
        for name in methods:
            try:
                _, _, t_stats, r_stats = run_method(name, As, Bs)
                t_rows.append(metric_row(name, t_stats))
                r_rows.append(metric_row(name, r_stats))
            except Exception:
                t_rows.append([name, "ERR", "ERR", "ERR", "ERR", "ERR"])
                r_rows.append([name, "ERR", "ERR", "ERR", "ERR", "ERR"])

        print("\nTranslation errors (mm):")
        print_table(t_rows, headers)

        print("\nRotation errors (deg):")
        print_table(r_rows, headers)

    else:
        X, Y, t_stats, r_stats = run_method(key, As, Bs)

        print("\nTranslation errors (mm):")
        print_table([metric_row(key, t_stats)], headers)

        print("\nRotation errors (deg):")
        print_table([metric_row(key, r_stats)], headers)

        if args.print_xy:
            print("")
            print_T("X", X)
            print_T("Y", Y)
            print("")


if __name__ == "__main__":
    main()
