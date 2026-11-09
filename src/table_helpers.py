from pathlib import Path
import csv
MODES=("r0_linear","r0_mlp","r1_linear","r1_mlp","r2_linear","r2_mlp")
def index(rows, keys):
    result = {tuple(r[k] for k in keys): r for r in rows}
    assert len(result) == len(rows), "Duplicate CSV keys"
    return result

def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(row) + " |" for row in rows])

def pair(row, metric="mae_bpm", signed=False, digits=2, tex=True):
    value = number(row[metric + "_mean"], signed, digits)
    sd = number(row[metric + "_sd_ddof0"], digits=digits)
    return f"${value}\\pm{sd}$" if tex else f"{value} ± {sd}"

def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def number(value, signed=False, digits=2):
    return format(float(value), f"{'+' if signed else ''}.{digits}f")
