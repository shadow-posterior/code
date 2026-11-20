"""Finalize GenEval benchmark metrics after reducing task counts across ranks."""

GENEVAL_TASKS = ('single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr')


def with_geneval_overall(metrics: dict[str, float]) -> dict[str, float]:
    """Keep dense rewards and add macro averages only for complete task sets.

    The released scorer calls standard GenEval success ``accuracy``. Its
    additional ``strict_accuracy`` requires exact object counts in every task
    and is reported separately; neither is the dense ``geneval`` reward.
    """
    result = dict(metrics)
    for suffix, name in [('accuracy', 'geneval_overall'),
                         ('strict_accuracy', 'geneval_strict_overall')]:
        keys = [f'{task}_{suffix}' for task in GENEVAL_TASKS]
        if all(key in metrics for key in keys):
            result[name] = sum(metrics[key] for key in keys) / len(keys)
    return result
