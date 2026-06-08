from dataclasses import dataclass


@dataclass
class MoETrainingConfig:
    lambda_div: float = 0.01
    tau: float = 1.5
    collapse_threshold: float = 0.40
