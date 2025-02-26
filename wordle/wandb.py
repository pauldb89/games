import wandb

from wordle.distributed import is_root_process


def wandb_init(*args: object, **kwargs: object) -> None:
	if is_root_process():
		wandb.init(*args, **kwargs)


def wandb_finish() -> None:
	if wandb.run is not None:
		wandb.finish()


def wandb_log(*args, **kwargs) -> None:
	if wandb.run is not None:
		wandb.log(*args, **kwargs)


def wandb_config_update(*args, **kwargs) -> None:
	if wandb.run is not None:
		wandb.config.update(*args, **kwargs)
