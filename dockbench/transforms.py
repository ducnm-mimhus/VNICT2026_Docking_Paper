"""
PyTorch-Ignite output transformations.

Note
----
PyTorch-Ignite :code:`output_transform` arguments allow to transform the
:code:`Engine.state.output` for the intendend use (by `ignite.metrics` and
`ignite.handlers`).
"""

from typing import Dict, Tuple

import torch


def output_transform_select_log_pose(
    output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select pose :code:`log_softmax` output and labels from output dictionary.

    Parameters
    ----------
    output: Dict[str, ignite.metrics.Metric]
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Logarithm of the pose class probabilities (:code:`log_softmax`) and class label

    Notes
    -----
    This function is used as :code:`output_transform` in
    :class:`ignite.metrics.metric.Metric` and allow to select pose results from
    the dictionary that the evaluator returns.

    The output is not activated, i.e. the :code:`log_softmax` output is returned
    unchanged
    """
    # Return pose log class probabilities and true labels
    return output["pose_log"], output["labels"]


def output_transform_select_pose(
    output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select pose :code:`softmax` output and labels from output dictionary.

    Parameters
    ----------
    output: Dict[str, ignite.metrics.Metric]
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Class probabilities and class labels

    Notes
    -----
    This function is used as :code:`output_transform` in
    :class:`ignite.metrics.metric.Metric` and allow to select pose results from
    the dictionary that the evaluator returns.

    The output is activated, i.e. the :code:`log_softmax` output is transformed into
    :code:`softmax`.
    """
    # Return pose class probabilities and true labels
    # log_softmax is transformed into softmax to get the class probabilities
    return torch.exp(output["pose_log"]), output["labels"]


def output_transform_select_affinity(
    output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select predicted affinities output and experimental (target) affinities from output
    dictionary.

    Parameters
    ----------
    output: Dict[str, ignite.metrics.Metric]
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Predicted binding affinity and experimental (target) binding affinity

    Notes
    -----
    This function is used as :code:`output_transform` in
    :class:`ignite.metrics.metric.Metric` and allow to select affinity predictions from
    the dictionary that the evaluator returns.
    """
    # Return pose class probabilities and true labels
    return output["affinities_pred"], output["affinities"]


def output_transform_select_affinity_abs(
    output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select predicted and target affinities, using absolute values for both.

    Parameters
    ----------
    output: Dict[str, ignite.metrics.Metric]
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        |predicted affinity| and |target affinity|

    Notes
    -----
    GNINA convention (original Caffe code): good poses have positive affinity,
    bad poses have negative affinity (= negated real affinity). The sign encodes
    pose quality for the loss function. For regression metrics (MAE, R2, RMSE),
    the original implementation often works on magnitudes.

    This transform is kept for backward compatibility but is NOT used in the
    main evaluation pipeline, which focuses on good poses only.
    """
    return output["affinities_pred"], torch.abs(output["affinities"])


def output_transform_affinity_good(
    output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select predicted and target affinities for GOOD poses only (target > 0).

    This matches GNINA's evaluation focus on well-defined binding affinities
    for good poses. Bad poses (negative targets) are excluded from MAE/RMSE.
    """
    pred = output["affinities_pred"]
    target = output["affinities"]
    mask = target > 0
    if mask.any():
        return pred[mask], target[mask]
    # If no good poses in batch, return empty tensors to avoid crashing metrics
    empty = pred.new_empty(0)
    return empty, empty


def output_transform_ROC(output) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Output transform for the ROC curve.

    Parameters
    ----------
    output:
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Positive class probability and associated labels.

    Notes
    -----
    https://pytorch.org/ignite/generated/ignite.contrib.metrics.ROC_AUC.html#roc-auc
    """
    # Select pose prediction
    pose, labels = output_transform_select_pose(output)

    # Return probability estimates of the positive class
    return pose[:, -1], labels


def output_transform_select_log_flex(
    output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select flexible residues pose :code:`log_softmax` output and labels from output
    dictionary.

    Parameters
    ----------
    output: Dict[str, ignite.metrics.Metric]
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Logarithm of the pose class probabilities (:code:`log_softmax`) and class label

    Notes
    -----
    This function is used as :code:`output_transform` in
    :class:`ignite.metrics.metric.Metric` and allow to select pose results from
    the dictionary that the evaluator returns.

    The output is not activated, i.e. the :code:`log_softmax` output is returned
    unchanged
    """
    # Return pose log class probabilities and true labels
    return output["flexpose_log"], output["flexlabels"]


def output_transform_select_flex(
    output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select flexible residues pose :code:`softmax` output and labels from output
    dictionary.

    Parameters
    ----------
    output: Dict[str, ignite.metrics.Metric]
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Class probabilities and class labels

    Notes
    -----
    This function is used as :code:`output_transform` in
    :class:`ignite.metrics.metric.Metric` and allow to select pose results from
    the dictionary that the evaluator returns.

    The output is activated, i.e. the :code:`log_softmax` output is transformed into
    :code:`softmax`.
    """
    # Return pose class probabilities and true labels
    # log_softmax is transformed into softmax to get the class probabilities
    return torch.exp(output["flexpose_log"]), output["flexlabels"]


def output_transform_ROC_flex(output) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Output transform for the ROC curve (for flexible residues pose)

    Parameters
    ----------
    output:
        Engine output

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Positive class probability and associated labels.

    Notes
    -----
    https://pytorch.org/ignite/generated/ignite.contrib.metrics.ROC_AUC.html#roc-auc
    """
    # Select pose prediction
    flexpose, flexlabels = output_transform_select_flex(output)

    # Return probability estimates of the positive class
    return flexpose[:, -1], flexlabels
