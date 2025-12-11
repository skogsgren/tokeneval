import argparse
import json
from pathlib import Path

from adjustText import adjust_text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FRIENDLY_METHOD = {
    "fvt": r"\textsc{FVT}",
    "omp": r"\textsc{OMP}",
    "100mb_src": r"100\,MB$_\text{src}$",
    "10mb_tgt": r"10\,MB$_\text{tgt}$",
}

FRIENDLY_LABEL = {
    "no-finetuning": r"\emph{No FT}",
    "mono-finetuning": r"\emph{Mono FT}",
    "mixed-finetuning": r"\emph{Mixed FT}",
}

FRIENDLY_ASCII_LABEL = {
    "no-finetuning": "No FT",
    "mono-finetuning": "Mono FT",
    "mixed-finetuning": "Mixed FT",
}

FRIENDLY_METRIC = {
    "ppl": "Perplexity",
    "mbp": "MultiBlimp",
}


def aggregate_group_table(group_tables, decimals=3):
    methods = set()
    langs = []
    for lang, df in group_tables.items():
        langs.append(lang)
        methods.update(df["Method"].tolist())
    methods = sorted(methods)
    langs_sorted = sorted(langs)
    data = {m: {} for m in methods}
    for lang in langs_sorted:
        df = group_tables[lang].copy()
        df = df.set_index("Method")
        df = df.replace("N/A", np.nan)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for m in methods:
            if m not in df.index:
                data[m][lang.upper()] = "N/A"
                continue
            row = df.loc[m]
            med = row.median(skipna=True)
            if pd.isna(med):
                data[m][lang.upper()] = "N/A"
            else:
                data[m][lang.upper()] = f"{med:.{decimals}f}"
    agg_df = (
        pd.DataFrame.from_dict(data, orient="index")
        .reset_index()
        .rename(columns={"index": "Method"})
    )
    ordered_cols = ["Method"] + [i.upper() for i in langs_sorted]
    agg_df = agg_df.loc[:, ordered_cols]

    return agg_df


def order_bold_table(df, baselines, bold_type="table"):
    mask = df["Method"].apply(lambda x: any(b in x for b in baselines))
    baseline_rows = df[mask]
    other_rows = df[~mask]
    df = pd.concat([baseline_rows, other_rows], ignore_index=True)

    if bold_type == "column":
        new_data = df.copy().astype(str)
        for col in df.columns:
            if col == "Method":
                continue
            if col == "TGT NormPPL" or col == "SRC+TGT NormPPL":
                numeric_values = df[col].apply(
                    lambda x: float(x.split()[0])
                    if isinstance(x, str) and x != "N/A"
                    else (np.inf if x == "N/A" else x)
                )

                min_val = numeric_values.min()
                new_data[col] = [
                    f"\\textbf{{{v}}}"
                    if (
                        isinstance(v, str)
                        and v != "N/A"
                        and float(v.split()[0]) == min_val
                    )
                    else v
                    for v in df[col]
                ]
            if col == "TGT MultiBlimp" or col == "SRC+TGT MultiBlimp":
                numeric_values = df[col].apply(
                    lambda x: float(x.split()[0])
                    if isinstance(x, str) and x != "N/A"
                    else (np.inf if x == "N/A" else x)
                )

                min_val = numeric_values.max()
                new_data[col] = [
                    f"\\textbf{{{v}}}"
                    if (
                        isinstance(v, str)
                        and v != "N/A"
                        and float(v.split()[0]) == min_val
                    )
                    else v
                    for v in df[col]
                ]

    elif bold_type == "table":
        numeric_map = np.full(df.shape, np.inf)
        for i, row in enumerate(df.values):
            for j, val in enumerate(row):
                if j == 0:
                    continue
                if isinstance(val, str) and val != "N/A":
                    head = val.split()[0]
                    numeric_map[i, j] = float(head)
                elif isinstance(val, (float, int)):
                    numeric_map[i, j] = float(val)
        min_idx = np.unravel_index(np.argmin(numeric_map), numeric_map.shape)

        new_data = df.copy().astype(str)
        new_data.iloc[min_idx] = f"\\textbf{{{new_data.iloc[min_idx]}}}"

    return new_data


def to_latex(df: pd.DataFrame, label: str, caption: str, columns: str, n_base: int = 0):
    df = df.rename(columns=lambda c: f"\\textbf{{{c}}}")
    latex = df.style.hide(axis=0).to_latex(
        caption=caption,
        label=f"tab:{label}",
        hrules=True,
        column_format=columns,
    )

    lines = latex.splitlines()
    lines.insert(1, r"\footnotesize")
    block_start = None
    block_end = None

    for i, line in enumerate(lines):
        if "\\caption" in line:
            block_start = i
            break

    brace_count = lines[block_start].count("{") - lines[block_start].count("}")
    j = block_start + 1
    while j < len(lines) and brace_count > 0:
        brace_count += lines[j].count("{") - lines[j].count("}")
        j += 1
    block_end = j
    assert "\\label" in lines[block_end]
    block_end += 1
    block = lines[block_start:block_end]
    del lines[block_start:block_end]

    # Move below the tabular environment
    tab_end = next(i for i, line in enumerate(lines) if "\\end{tabular}" in line)
    for k, bline in enumerate(block):
        lines.insert(tab_end + 1 + k, bline)
    latex = "\n".join(lines)

    if n_base > 0:
        lines = latex.splitlines()
        start_idx = next(i for i, line in enumerate(lines) if "\\midrule" in line) + 1
        midrule_target = start_idx + n_base
        lines.insert(midrule_target, r"\midrule")
        latex = "\n".join(lines)
    return latex


def latex_escape(text):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


def pretty(method, label):
    method = FRIENDLY_METHOD.get(method, method)
    label = FRIENDLY_LABEL.get(label, label)
    return rf"{method} ({label})"


def pretty_ascii(method, label):
    label = FRIENDLY_ASCII_LABEL.get(label, label)
    return rf"{method.upper()} ({label})"


def pretty_pair(src_lang, tgt_lang):
    return rf"\texttt{{{src_lang[:3].upper()}$\rightarrow${tgt_lang[:3].upper()}}}"


def build_tables(data, allowed_methods, uid_threshold=0.05, excluded=None):
    excluded = excluded or {}

    def tgt_ok(metrics):
        return metrics.get("tgt_uid_prop", 1) < uid_threshold

    def src_ok(metrics):
        return metrics.get("src_uid_prop", 1) < uid_threshold

    def pair_ok(metrics):
        return (
            metrics.get("tgt_uid_prop", 1) < uid_threshold
            and metrics.get("src_uid_prop", 1) < uid_threshold
        )

    rows_t1 = []
    rows_t4 = []
    scatter_data = {}

    for label, src_block in data.items():
        for method in allowed_methods:
            if method in excluded.get(label, []):
                continue
            collected, bpt_coll, log_ppl, sum_ppl = [], [], [], []
            src_ppl_list, tgt_ppl_list = [], []
            sum_mbp, src_mbp, tgt_mbp = [], [], []

            for src_lang, tgt_block in src_block.items():
                for tgt_lang, method_block in tgt_block.items():
                    metrics = method_block.get(method)
                    if metrics and tgt_ok(metrics):
                        collected.append(metrics["tgt_norm_ppl"])
                        log_ppl.append(metrics["tgt_log_ppl"])
                        bpt_coll.append(metrics["tgt_bpt"])
                        tgt_mbp.append(metrics["tgt_mbp"])
                    if metrics and pair_ok(metrics):
                        src_ppl_list.append(metrics["src_norm_ppl"])
                        tgt_ppl_list.append(metrics["tgt_norm_ppl"])
                        sum_ppl.append(metrics["sum_norm_ppl"])
                        src_mbp.append(metrics["src_mbp"])
                        sum_mbp.append(metrics["src_mbp"] + metrics["tgt_mbp"])

            method_label = pretty(method, label)
            method_label_ascii = pretty_ascii(method, label)

            scatter_data[method_label_ascii] = {
                "src_ppl": np.median(src_ppl_list),
                "tgt_ppl": np.median(tgt_ppl_list),
                "src_mbp": np.median(src_mbp),
                "tgt_mbp": np.median(tgt_mbp),
            }

            arr = np.array(collected)
            log_ppl = np.array(log_ppl)
            sum_ppl = np.array(sum_ppl)
            bpt = np.array(bpt_coll)
            tgt_mbp = np.array(tgt_mbp)
            sum_mbp = np.array(sum_mbp)

            rows_t1.append(
                {
                    "Method": method_label,
                    "TGT NormPPL": f"{np.median(arr):.3f} (std={np.std(arr):.3f})",
                    "TGT MultiBlimp": f"{np.median(tgt_mbp):.3f} (std={np.std(tgt_mbp):.3f})",
                    "Bytes/Token": f"{np.mean(bpt):.2f}",
                }
            )
            rows_t4.append(
                {
                    "Method": method_label,
                    "SRC+TGT NormPPL": f"{np.median(sum_ppl):.3f} (std={np.std(sum_ppl):.3f})",
                    "SRC+TGT MultiBlimp": f"{np.median(sum_mbp):.3f} (std={np.std(sum_mbp):.3f})",
                }
            )
    df_t1 = pd.DataFrame(rows_t1)
    df_t4 = pd.DataFrame(rows_t4)

    per_src_tables = {}

    for label, src_block in data.items():
        for src_lang, tgt_block in src_block.items():
            if src_lang not in per_src_tables:
                per_src_tables[src_lang] = {}

            for tgt_lang, method_block in tgt_block.items():
                for method in allowed_methods:
                    if method in excluded.get(label, []):
                        continue
                    metrics = method_block.get(method)
                    if not metrics:
                        continue

                    method_label = pretty(method, label)
                    pair_key = pretty_pair(src_lang, tgt_lang)

                    if method_label not in per_src_tables[src_lang]:
                        per_src_tables[src_lang][method_label] = {}

                    per_src_tables[src_lang][method_label][pair_key] = (
                        f"{metrics['sum_norm_ppl']:.3f}" if pair_ok(metrics) else "N/A"
                    )

    df_t2 = {}
    for src_lang, rows in per_src_tables.items():
        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.reset_index().rename(columns={"index": "Method"})
        df = df[sorted(df.columns)]
        df_t2[src_lang] = df

    per_tgt_tables = {}

    for label, src_block in data.items():
        for src_lang, tgt_block in src_block.items():
            for tgt_lang, method_block in tgt_block.items():
                if tgt_lang not in per_tgt_tables:
                    per_tgt_tables[tgt_lang] = {}

                for method in allowed_methods:
                    if method in excluded.get(label, []):
                        continue
                    metrics = method_block.get(method)
                    if not metrics:
                        continue

                    method_label = pretty(method, label)
                    pair_key = pretty_pair(src_lang, tgt_lang)

                    if method_label not in per_tgt_tables[tgt_lang]:
                        per_tgt_tables[tgt_lang][method_label] = {}

                    per_tgt_tables[tgt_lang][method_label][pair_key] = (
                        f"{metrics['sum_norm_ppl']:.3f}" if tgt_ok(metrics) else "N/A"
                    )

    df_t3 = {}
    for tgt_lang, rows in per_tgt_tables.items():
        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.reset_index().rename(columns={"index": "Method"})
        df = df[sorted(df.columns)]
        df_t3[tgt_lang] = df

    return df_t1, df_t2, df_t3, scatter_data, df_t4


def strip_latex(text):
    replacements = [
        (r"\textsc{", ""),
        (r"}", ""),
        (r"\emph{", ""),
        (r"\,M$_\text{src}$", "M_src"),
        (r"\,M$_\text{tgt}$", "M_tgt"),
        (r"\\", ""),
    ]
    for pattern, repl in replacements:
        text = text.replace(pattern, repl)
    return text


def create_scatter(scatter_data, metric="ppl", output_path="./plots/scatter.pdf"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    texts = []
    for method_label, values in scatter_data.items():
        ax.scatter(values[f"tgt_{metric}"], values[f"src_{metric}"], s=100, alpha=0.6)
        txt = ax.annotate(
            method_label,
            (values[f"tgt_{metric}"], values[f"src_{metric}"]),
            fontsize=7,
            alpha=0.8,
        )
        texts.append(txt)

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel(f"Normalized Target {FRIENDLY_METRIC[metric]} (median)")
    ax.set_ylabel(f"Normalized Source {FRIENDLY_METRIC[metric]} (median)")
    ax.set_title(f"Source vs. Target {FRIENDLY_METRIC[metric]}")
    ax.grid(True, alpha=0.3)

    all_src = [v[f"src_{metric}"] for v in scatter_data.values()]
    all_tgt = [v[f"tgt_{metric}"] for v in scatter_data.values()]
    max_val = max(max(all_src), max(all_tgt))
    min_val = min(min(all_src), min(all_tgt))
    # ax.set_xlim(min_val * 0.9, max_val * 1.05)
    # ax.set_ylim(min_val * 0.9, max_val * 1.05)

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--mono_json", type=Path, required=True)
    parser.add_argument("--mixed_json", type=Path, required=True)
    args = parser.parse_args()

    out_data = json.loads(args.out_json.read_text())
    mono_data = json.loads(args.mono_json.read_text())
    mixed_data = json.loads(args.mixed_json.read_text())
    data = {
        "no-finetuning": out_data,
        "mono-finetuning": mono_data,
        "mixed-finetuning": mixed_data,
    }

    baselines = [FRIENDLY_METHOD["100mb_src"], FRIENDLY_METHOD["10mb_tgt"]]
    uid_threshold = 0.2
    tables = build_tables(
        data,
        ["fvt", "omp", "100mb_src", "10mb_tgt"],
        uid_threshold=uid_threshold,
        excluded={
            "mono-finetuning": ["10mb_tgt"],
            "mixed-finetuning": ["10mb_tgt"],
        },
    )

    first_caption = """
Median and standard deviation of mean normalized target perplexity (smaller is better) and MultiBlimp (larger is better) scores across all language pairs (n=30), alongside average bytes per token for each method.
"""
    first_table = to_latex(
        df=order_bold_table(tables[0], baselines, "column"),
        label="aggregated_tgt_ppl",
        caption=first_caption,
        columns="llll",
        n_base=len(baselines),
    )
    print(first_table)

    second_caption = """
Median and standard deviation of mean normalized source + target perplexity (smaller is better) and MultiBlimp (larger is better) scores across all language pairs (n=30)
"""
    second_table = to_latex(
        df=order_bold_table(tables[4], baselines, "column"),
        label="aggregated_sum_ppl",
        caption=second_caption,
        columns="lll",
        n_base=len(baselines),
    )
    print(second_table)

    create_scatter(tables[3], "ppl", Path("./plots/bilingual_ppl.pdf"))
    create_scatter(tables[3], "mbp", Path("./plots/bilingual_mbp.pdf"))
