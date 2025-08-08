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

x = Tensor(np.random.randn(10, 1))
z = Tensor(np.random.randn(5, 1))
epochs = 10

lin1 = Linear(20, 10)
relu = ReLU()
lin2 = Linear(5, 20)


loss_fn = MSELoss()
optim = SimpleStep(lr=0.01)
model = ModuleSequence([lin1, relu, lin2], optim=optim)

for epoch in range(epochs):
    model.zero_grad()
    y = model(x)
    loss = loss_fn(y, z)
    loss.backward()
    model.step()
    print(f"Epoch {epoch + 1}/{epochs} loss: {loss.value}")
