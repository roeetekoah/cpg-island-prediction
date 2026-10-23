from __future__ import annotations

def remove_short_cpg_segments(labels: str, min_len: int) -> str:
    """
    Post-process a base-level label string ('C'/'N') by removing CpG segments shorter than min_len.
    Any run of 'C' with length < min_len is converted to 'N'.

    If min_len <= 1, returns labels unchanged.
    """
    if min_len <= 1:
        return labels
    n = len(labels)
    if n == 0:
        return labels

    out = list(labels)
    i = 0
    while i < n:
        if out[i] != 'C':
            i += 1
            continue
        j = i + 1
        while j < n and out[j] == 'C':
            j += 1
        if (j - i) < min_len:
            for k in range(i, j):
                out[k] = 'N'
        i = j
    return ''.join(out)
