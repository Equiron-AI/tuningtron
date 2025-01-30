from transformers import TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Optional


def _create_tuningtron_optimizer(model,
                                 optimizer_cls,
                                 optimizer_kwargs,
                                 embedding_lr):
    print("Tuningtron: Create custom optimizer based on:", str(optimizer_cls), "with args:", optimizer_kwargs)

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = {
        "embeddings": {},
        "non_embeddings": {}
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ("lm_head" in name) or ("embed_tokens" in name):
            print(f"Tuningtron: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {name}.")
            param_groups["embeddings"][name] = param
        else:
            param_groups["non_embeddings"][name] = param

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": embedding_lr,
        },
        {
            "params": list(param_groups["non_embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        }
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


@dataclass
class TuningtronTrainingArguments(TrainingArguments):
    embedding_learning_rate: Optional[float] = field(default=None)


class TuningtronTrainer(Trainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_tuningtron_optimizer(self.model,
                                                          optimizer_cls,
                                                          optimizer_kwargs,
                                                          embedding_learning_rate)
        return self.optimizer
