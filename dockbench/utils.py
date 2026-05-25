import argparse
import sys
from typing import Dict, Optional, Tuple

import molgrid
import torch

# Last printed metrics per split ("Train" / "Test") for console Δ lines (reset when epoch == 1).
_prev_metrics_console: Dict[Tuple[str], Dict[str, float]] = {}


def _metric_to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except (TypeError, ValueError):
        return None


def print_args(
    args: argparse.Namespace, header: Optional[str] = None, stream=sys.stdout
):
    """
    Print command line arguments to stream.

    Parameters
    ----------
    args: argparse.Namespace
        Command line arguments
    header: str
        Header string
    stream:
        Output stream
    """
    if header is not None:
        print(header, file=stream)
    for name, value in vars(args).items():
        if type(value) is float:
            print(f"{name}: {value:.5E}", file=stream)
        else:
            print(f"{name} = {value!r}", file=stream)

    # Flush stream
    print("", end="", file=stream, flush=True)


def log_print(
    metrics,
    title: Optional[str] = None,
    epoch: Optional[int] = None,
    epoch_time: Optional[float] = None,
    elapsed_time: Optional[float] = None,
    stream=sys.stdout,
):
    """
    Print metrics to a stream. Uses compact format for stdout, full detail for files.

    Parameters
    ----------
    metrics:
        Dictionary of metrics
    title: str
        Title to print
    epoch: int
        Epoch number
    epoch_time: float
        Time for this epoch
    elapsed_time: float
        Total elapsed time
    stream:
        Output stream
    """
    is_console = (stream is sys.stdout or stream is sys.stderr)

    if is_console:
        # --- Readable multi-line format: values + Δ vs previous epoch (same run) ---
        tkey = title or "_"
        if epoch == 1:
            _prev_metrics_console.pop((tkey,), None)

        prev = _prev_metrics_console.get((tkey,))

        order = [
            ("Pose Loss", "Pose"),
            ("Affinity Loss", "Aff"),
            ("Balanced Accuracy", "BalAcc"),
            ("Pose Recall Neg", "Rneg"),
            ("Pose Recall Pos", "Rpos"),
            ("PR AUC", "PR"),
            ("MCC", "MCC"),
            ("MAE", "MAE"),
            ("RMSE", "RMSE"),
            ("Pearson R", "r"),
            ("Spearman Rho", "rho"),
            ("C-index", "Cidx"),
            ("EF_1pct", "EF1"),
            ("EF_5pct", "EF5"),
            ("Success_at_1", "S@1"),
            ("Success_at_5", "S@5"),
            ("Success_at_10", "S@10"),
            ("ROC AUC", "AUC"),
            ("Accuracy", "Acc"),
        ]

        sep = "-" * 78
        print(sep, file=stream, flush=True)
        head = []
        if title is not None and epoch is not None:
            head.append(f"[{title}  epoch {epoch}]")
        if epoch_time is not None:
            head.append(f"epoch_t={epoch_time:.2f}s")
        if elapsed_time is not None:
            head.append(f"total_t={elapsed_time:.0f}s")
        if head:
            print("  " + "  ".join(head), file=stream, flush=True)

        vals = []
        for full, short in order:
            if full not in metrics:
                continue
            fv = _metric_to_float(metrics[full])
            if fv is None:
                continue
            vals.append(f"{short}={fv:.4f}")
        if vals:
            print("  " + " | ".join(vals), file=stream, flush=True)

        if prev:
            deltas = []
            for full, short in order:
                if full not in metrics or full not in prev:
                    continue
                cur = _metric_to_float(metrics[full])
                old = prev.get(full)
                if cur is None or old is None:
                    continue
                d = cur - old
                if abs(d) < 1e-7:
                    continue
                sign = "+" if d >= 0 else ""
                deltas.append(f"Δ{short}={sign}{d:.4f}")
            if deltas:
                print("  " + " | ".join(deltas), file=stream, flush=True)

        flat: Dict[str, float] = {}
        for k, v in metrics.items():
            fv = _metric_to_float(v)
            if fv is not None:
                flat[k] = fv
        _prev_metrics_console[(tkey,)] = flat
    else:
        # --- Full detail format for log file ---
        if title is not None and epoch is not None:
            print(f">>> {title} - Epoch[{epoch}] <<<", file=stream)
            indent = "    "
        else:
            indent = ""

        loss: float = 0.0
        for name, value in metrics.items():
            print(f"{indent}{name}: {value:.5f}", file=stream)
            if "loss" in name.lower():
                loss += value

        if loss > 0:
            print(f"    Loss: {loss:.5f}", file=stream)

        if epoch_time is not None:
            print(f"{indent}Epoch Time: {epoch_time:.5f}", file=stream, flush=True)

        if elapsed_time is not None:
            print(f"{indent}Elapsed Time: {elapsed_time:.5f}", file=stream, flush=True)

        # Flush stream
        print("", end="", file=stream, flush=True)


def set_device(device_name: str) -> torch.device:
    """
    Set the device to use.

    Parameters
    ----------
    device_name: str
        Name of the device to use (:code:`"cpu"`, :code:`"cuda"`, :code:`"cuda:0"`, ...)

    Returns
    -------
    torch.device
        PyTorch device

    Notes
    -----
    This function also set the global device for :code:`molgrid` so that the
    :code:`molgrid.ExampoleProvider` works on the correct device.

    https://github.com/gnina/libmolgrid/issues/43
    """
    # TODO: Set global PyTorch device?

    device = torch.device(device_name)
    if "cuda" in device_name:
        try:  # cuda:IDX
            idx = int(device_name[-1])
            molgrid.set_gpu_device(idx)
        except ValueError:  # cuda
            # Set device 0 by default
            molgrid.set_gpu_device(0)

    return device
