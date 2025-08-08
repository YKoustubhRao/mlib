from mlib.nn import (
    Linear,
    ReLU,
    Tensor,
    set_seed,
    SimpleStep,
    ModuleSequence,
    flatten,
    Reshape,
    DropOut,
    SoftMax,
    CrossEntropyLoss,
)
import numpy as np

set_seed(42)

x = Tensor(np.random.randn(5, 1, 2))
z = Tensor(np.array([[0], [0], [0], [1], [0]]))
epochs = 10

res = Reshape((2, 5))
lin1 = Linear(20, 10)
drop = DropOut(0.01)
lin2 = Linear(5, 20)
relu = ReLU()
soft = SoftMax()


loss_fn = CrossEntropyLoss()
optim = SimpleStep(lr=0.01)
model = ModuleSequence([res, flatten, lin1, relu, drop, lin2, soft], optim=optim)

for epoch in range(epochs):
    model.zero_grad()
    y = model(x)
    loss = loss_fn(y, z)
    loss.backward()
    model.step()
    print(f"Epoch {epoch + 1}/{epochs} loss: {loss.value}")
