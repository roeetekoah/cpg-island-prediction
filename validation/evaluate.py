from dataclasses import dataclass

@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision_c: float
    recall_c: float
    f1_c: float
    tp: int
    fp: int
    fn: int
    tn: int

def evaluate_predictions(true_lbl: str, pred_lbl: str):
    if len(true_lbl) != len(pred_lbl):
        raise ValueError(f"Label length mismatch: true={len(true_lbl)} pred={len(pred_lbl)}")

    tp = fp = fn = tn = 0
    for t, p in zip(true_lbl, pred_lbl):
        if p == "C":
            if t == "C":
                tp += 1
            else:
                fp += 1
        else:
            if t == "C":
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn

def metrics_from_counts(tp, fp, fn, tn) -> Metrics:
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return Metrics(acc, prec, rec, f1, tp, fp, fn, tn)
