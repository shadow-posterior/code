"""Expose raw SPM training rewards as one metrics section."""


def with_training_reward_section(sections):
    result = {section: dict(values) for section, values in sections.items()}
    generator_metrics = sections.get('generator', {})
    rewards = {key.removeprefix('reward/'): value for key, value in generator_metrics.items()
               if key.startswith('reward/')}
    if 'reward_mean' in generator_metrics:
        rewards['mean'] = generator_metrics['reward_mean']
    if rewards:
        result['train_reward'] = {**rewards, **result.get('train_reward', {})}
    return result
