from prefect import flow, task
from pipelines.data_pipeline import collect_data_pipeline
from pipelines.preprocessing_pipeline import preprocessing_pipeline
from pipelines.dataset_pipeline import generate_dataset_pipeling
from pipelines.train_pipeline import training_pipeline
from pipelines.evaluate_pipeline import evaluation_pipeline


@task
def collect_data():
    print("Running Data Collection ....")
    collect_data_pipeline()
    print("phase 1 done")


@task
def preprocess_data():
    print("Running Preprocessing ....")
    preprocessing_pipeline()
    print("phase 2 done")


@task
def build_dataset():
    print("Building Dataset ....")
    generate_dataset_pipeling()
    print("phase 3 done")


@task
def train():
    print("Training Started ....")
    training_pipeline()
    print("Training is done ....")
    print("phase 4 done ....")


@task
def evaluate():
    print("Running Evaluation ....")
    evaluation_pipeline()
    print("phase 5 is done")


@flow(name="llm_pipeline")
def full_pipeline(run_training: bool = True, run_eval: bool = True):
    collect_data()
    preprocess_data()
    build_dataset()
    if run_training:
        train()
    if run_eval:
        evaluate()


if __name__ == "__main__":
    full_pipeline()