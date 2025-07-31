import wandb


def init_tracking(projrct="finetune_shortlm", run_name="qlora-run"):
    wandb.init(project=projrct, name=run_name)
