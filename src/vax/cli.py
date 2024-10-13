from pathlib import Path
from vax.console import console
import rich_click as click
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.WARN, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--debug/--no-debug",
    "-d/-n",
    default=False,
    show_default=True,
    help="Show the debug log messages",
)
@click.version_option("0.0.1", prog_name="vax")
@click.rich_config(console=console)
def cli(debug):
    """
    My amazing tool does all the things.

    This is a minimal example based on documentation
    from the 'click' package.

    You can try using --help at the top level and also for
    specific subcommands.
    """
    # print(f"Debug mode is {'on' if debug else 'off'}")



@cli.group()
def dataset():
    """Manage datasets"""

@dataset.command()
@click.argument("input_type", type=click.Choice(["yolov8-det", "generic-cls", "fashion-mnist"]))
@click.argument("input", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(exists=False, writable=True, path_type=Path))
def convert(input_type, input, output):
    """Convert dataset from one format to another"""
    log.info(f"Converting {input_type} dataset in {input} to vax dataset in {output}")
    match input_type:
        case "yolov8-det":
            from vax.dataset.yolov8 import convert_yolov8_det_dataset
            convert_yolov8_det_dataset(input, output)
        case "fashion-mnist":
            from vax.dataset.fashion_mnist import convert_fashion_mnist_dataset
            convert_fashion_mnist_dataset(input / "fashion-mnist_train.csv", input / "fashion-mnist_test.csv", output)
        case "generic-cls":
            from vax.dataset.generic import convert_generic_classification_dataset
            convert_generic_classification_dataset(input, output)
        