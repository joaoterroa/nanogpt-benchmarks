from baseline import Baseline


class Config:
    def __init__(self, n_embd, n_head, block_size):
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size


def params_counter(model_class) -> int:
    dummy_config = Config(n_embd=768, n_head=16, block_size=1024)
    model = model_class(dummy_config)
    num_params = sum(p.numel() for p in model.parameters())

    return num_params


def baseline_difference(model_class):
    from baseline import Baseline

    baseline_params = params_counter(Baseline)
    head_params = params_counter(model_class)
    difference = ((head_params - baseline_params) / baseline_params) * 100
    print(
        f"{model_class.__name__} params difference over Baseline = {difference:.2f}% | num params = {head_params}"
    )

    return baseline_params, head_params, difference


if __name__ == "__main__":
    from baseline import Baseline
    from ssa import SSA
    from slsa import SLSA
    from vlsa import VLSA
    from vsa import VSA
    from lsa import LSA

    baseline_difference(Baseline)
    baseline_difference(SSA)
    baseline_difference(SLSA)
    baseline_difference(LSA)
    baseline_difference(VSA)
    baseline_difference(VLSA)
