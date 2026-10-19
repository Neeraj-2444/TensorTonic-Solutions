def linear_lr(step: int, total_steps: int, initial_lr: float, final_lr: float = 0.0, warmup_steps: int = 0) -> float:
    """
    Returns the learning rate as a float.
    """
    # Write code here
    if warmup_steps>0 and step < warmup_steps:
        return initial_lr*(step/warmup_steps)
    if step>=warmup_steps and step<total_steps:
        return initial_lr + ((step-warmup_steps)/(total_steps-warmup_steps))*(final_lr-initial_lr)
    if step>=total_steps:
        return final_lr
    pass