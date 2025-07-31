import typer
from pipelines.data_pipeline import collect_data_pipeline
from pipelines.preprocessing_pipeline import preprocessing_pipeline
from pipelines.dataset_pipeline import generate_dataset_pipeling
from pipelines.train_pipeline import training_pipeline
from pipelines.evaluate_pipeline import evaluation_pipeline
from pipelines.serving_pipeline import serving_pipeline
from orchestration.workflow import full_pipeline
app = typer.Typer()

@app.command()
def collect_data():
    typer.echo("Running Data Collection ....")
    collect_data_pipeline()
    typer.echo("phase 1 done")


@app.command()
def preprocess_data():
    typer.echo("Running Preprocessin ....")
    preprocessing_pipeline
    typer.echo("phase 2 done")


@app.command()
def build_dataset():
    typer.echo("Building Dataset ....")
    generate_dataset_pipeling
    typer.echo("phase 3 done")


# I'll use google colab to Train and Evaluate it for GPU use
@app.command()
def train():
    typer.echo("Training Started ....")
    training_pipeline()
    typer.echo("Training is done ....")
    typer.echo("phase 4 done ....")


@app.command()
def evaluate():
    typer.echo("Running Evaluation ....")
    evaluation_pipeline()
    typer.echo("phase 5 is done ")


@app.command()
def query(
    prompt: str = typer.Argument(..., help="Input prompt"),
    max_tokens: int = typer.Option(
        200, "--max_tokens", "-m", help="Maximum tokens to generate"),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Sampling temperature")
):
    serving_pipeline(prompt, max_tokens, temperature)



@app.command()
def run_flow(
    train: bool = typer.Option(True, help="Run training?"),
    eval: bool = typer.Option(True, help="Run evaluation?")
):
    full_pipeline(run_training=train, run_eval=eval)


@app.command()
def run_all():
    collect_data()
    preprocess_data()
    build_dataset()
    # train()
    # evaluate()
    typer.echo("local pipeline is done")


if __name__ == "__main__":
    app()
