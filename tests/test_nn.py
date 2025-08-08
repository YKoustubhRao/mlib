from mlib.nn import (
    Linear,
    ReLU,
    Tensor,
    set_seed,
    SimpleStep,
    ModuleSequence,
    MSELoss,
)
import numpy as np

set_seed(42)


def test_basic():
    x = Tensor(np.random.randn(10, 1))
    z = Tensor(np.random.randn(5, 1))
    epochs = 10

    lin1 = Linear(20, 10)
    relu = ReLU()
    lin2 = Linear(5, 20)

    loss_fn = MSELoss()
    optim = SimpleStep(lr=0.01)
    model = ModuleSequence([lin1, relu, lin2], optim=optim)

    loss_val = 64

    for _ in range(epochs):
        model.zero_grad()
        y = model(x)
        loss = loss_fn(y, z)
        loss.backward()
        model.step()
        assert loss.value < loss_val, (
            f"Loss did not decrease as expected: {loss.value} >= {loss_val}"
        )
        loss_val = loss.value

    assert loss.value < 0.1, f"Final loss is too high than expected: {loss.value}"
