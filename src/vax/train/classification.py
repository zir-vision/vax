import optax
from flax import nnx
from vax.nn.backbones.resnet import ResNetClassification, ResNet18, ResNet101
from vax.dataset import Dataset, ClassificationDecoder, EagerDataloader
import sys


# See https://github.com/google/flax/blob/main/docs_nnx/mnist_tutorial.ipynb

ds = Dataset.load(sys.argv[1])

decoder = ClassificationDecoder(ds)

dl = EagerDataloader(ds, decoder)

rngs = nnx.Rngs(0)
model = ResNetClassification(ResNet101(rngs=rngs), num_classes=10, rngs=rngs)

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)


def loss_fn(model: ResNetClassification, batch):
    logits = model(batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["label"]).mean()
    return loss, logits


@nnx.jit
def train_step(model: ResNetClassification, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # inplace updates
    optimizer.update(grads)  # inplace updates


@nnx.jit
def eval_step(model: ResNetClassification, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # inplace updates

metrics_history = {
  "train_loss": [],
  "train_accuracy": [],
  "valid_loss": [],
  "valid_accuracy": [],
}

train_steps = 1200
eval_every = 100
batch_size = 64


for step, batch in enumerate(dl.batch_set("train", batch_size)):
  # Run the optimization for one step and make a stateful update to the following:
  # - The train state's model parameters
  # - The optimizer state
  # - The training loss and accuracy batch metrics
  train_step(model, optimizer, metrics, batch)

  if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
    # Log the training metrics.
    for metric, value in metrics.compute().items():  # Compute the metrics.
      metrics_history[f"train_{metric}"].append(value)  # Record the metrics.
    metrics.reset()  # Reset the metrics for the valid set.

    # Compute the metrics on the valid set after each training epoch.
    for valid_batch in dl.batch_set("valid", batch_size):
      eval_step(model, metrics, valid_batch)

    # Log the valid metrics.
    for metric, value in metrics.compute().items():
      metrics_history[f"valid_{metric}"].append(value)
    metrics.reset()  # Reset the metrics for the next training epoch.

    print(
      f"[train] step: {step}, "
      f"loss: {metrics_history["train_loss"][-1]}, "
      f"accuracy: {metrics_history["train_accuracy"][-1] * 100}"
    )
    print(
      f"[valid] step: {step}, "
      f"loss: {metrics_history["valid_loss"][-1]}, "
      f"accuracy: {metrics_history["valid_accuracy"][-1] * 100}"
    )