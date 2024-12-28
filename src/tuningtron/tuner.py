import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
import torch
import numpy as np
import logging
import deepspeed
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from tuningtron.models import ModelsFactory


logger = logging.getLogger(__name__)


class Tuner:
    def __init__(self, base_model_id, enable_deepspeed=True, enable_offload_optimizer=True):
        self.print_cuda_info()

        self.base_model_id = base_model_id
        self.model_config = ModelsFactory().get_model_config(base_model_id)
        self.tokenizer = self.model_config.tokenizer

        self.device_map = "auto"
        self.deepspeed = None
        self.optim = "adamw_8bit"
        self.bf16 = False
        self.fp16 = False
        self.attn_implementation = None
        if torch.cuda.is_available():
            if enable_deepspeed:
                deepspeed.init_distributed()
                self.device_map = None
                self.deepspeed = self.get_deepspeed_config(enable_offload_optimizer)
                self.optim = "adamw_torch"
                logger.info("deepspeed: enabled")

            if torch.cuda.get_device_capability()[0] >= 8:
                self.bf16 = True
                if self.model_config.config.model_type.startswith("gemma"):
                    self.attn_implementation = "eager"
                else:
                    self.attn_implementation = "flash_attention_2"
            else:
                self.fp16 = True
        else:
            self.bf16 = True
        logger.info(f"Detected hyperparameters: fp16: {self.fp16}, bf16: {self.bf16}, flash_attentions: {self.attn_implementation}")

    def get_instruction(self, record):
        return self.model_config.apply_chat_template(record)

    def map_func(self, record):
        return self.tokenizer(self.get_instruction(record), truncation=True, max_length=self.max_len, padding="max_length")

    def filter_func(self, record):
        return len(self.tokenizer(self.get_instruction(record))["input_ids"]) <= self.max_len

    def sft(self,
            dataset,
            adapter_name,
            do_eval=False,
            max_len_percentile=100,
            max_len=None,
            truncation=False,
            rank=32,
            lora_alpha=None,
            lora_dropout=0.1,
            num_train_epochs=1,
            batch_size=4,
            gradient_steps=2,
            learning_rate=1e-5,
            comp_only=False):
        dataset = datasets.load_dataset(dataset, split="train")

        if max_len:
            self.max_len = max_len
        else:
            inputs = [self.tokenizer(self.get_instruction(record))["input_ids"] for record in dataset]
            target_lenghts = [len(x) for x in inputs]
            self.max_len = int(np.percentile(target_lenghts, max_len_percentile))

        logger.info(f"Dataset max_len detected: {self.max_len}")
        logger.info("Dataset example row after appy chat template:")
        logger.info("---------------------------------------------")
        logger.info(self.get_instruction(dataset[0]))
        logger.info("---------------------------------------------")
        if not truncation:
            print("DS before max_len filtering:", dataset)
            dataset = dataset.filter(lambda record: self.filter_func(record))
            print("DS after max_len filtering:", dataset)

        dataset = dataset.map(self.map_func)
        print("DS after mapping:", dataset)
        logger.info(dataset["input_ids"][0])
        logger.info("---------------------------------------------")

        if "text" in dataset.column_names:
            dataset = dataset.remove_columns(["text"])
            lr_scheduler_type = "linear"

        if "instruct" in dataset.column_names:
            dataset = dataset.remove_columns(["instruct", "input", "output"])
            lr_scheduler_type = "linear"

        if comp_only:
            logger.info("Using data collator: CompletionOnlyLM")
            data_collator = DataCollatorForCompletionOnlyLM(self.model_config.response_template, tokenizer=self.tokenizer)
        else:
            logger.info("Using data collator: LanguageModeling")
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        train_dataset, eval_dataset = self.prepare_datasets(dataset, do_eval)

        args = self.prepare_args(num_train_epochs, learning_rate, batch_size, gradient_steps, lr_scheduler_type)
        print(args)
        config = TrainingArguments(**args)

        peft_model = get_peft_model(self.load_base_model(), self.get_lora_config(rank, lora_alpha))
        logger.info(peft_model.get_model_status())

        trainer = Trainer(model=peft_model,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=data_collator,
                          args=config)
        trainer.train()
        trainer.save_model(adapter_name)

    def dpo(self,
            dataset,
            adapter_name,
            do_eval=False,
            rank=32,
            lora_alpha=None,
            lora_dropout=0.1,
            num_train_epochs=1,
            batch_size=4,
            gradient_steps=2,
            learning_rate=1e-5):
        dataset = datasets.load_dataset(dataset, split="train")
        train_dataset, eval_dataset = self.prepare_datasets(dataset, do_eval)

        args = self.prepare_args(num_train_epochs, learning_rate, batch_size, gradient_steps, "linear")
        print(args)
        config = DPOConfig(**args)

        trainer = DPOTrainer(model=self.load_base_model(),
                             peft_config=self.get_lora_config(rank, lora_alpha),
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             processing_class=self.tokenizer,
                             args=config)
        trainer.train()
        trainer.save_model(adapter_name)

    def prepare_datasets(self, dataset, do_eval):
        eval_dataset = None
        self.eval_strategy = "no"
        self.eval_steps = None

        if do_eval:
            dataset = dataset.train_test_split(test_size=0.1)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
            self.eval_strategy = "steps"
            self.eval_steps = 0.1
            logger.info("Eval dataset:")
            logger.info(eval_dataset)
        else:
            train_dataset = dataset
        logger.info("Train dataset:")
        logger.info(train_dataset)

        return train_dataset, eval_dataset

    def prepare_args(self, num_train_epochs, learning_rate, batch_size, gradient_steps, lr_scheduler_type):
        return {
            "output_dir": ".",
            "num_train_epochs": num_train_epochs,
            "logging_steps": 1,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "gradient_checkpointing": True,
            "save_strategy": "no",
            "bf16": self.bf16,
            "fp16": self.fp16,
            "optim": self.optim,
            "weight_decay": 0.001,
            "learning_rate": learning_rate,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": 0.1,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_steps,
            "eval_accumulation_steps": 1,
            "deepspeed": self.deepspeed
        }

    def get_lora_config(self, rank, lora_alpha):
        lora_alpha = lora_alpha if lora_alpha else rank
        config = LoraConfig(r=rank, lora_alpha=lora_alpha, target_modules=self.model_config.target_modules, lora_dropout=0.1, task_type="CAUSAL_LM")
        print("Lora config:", config)
        return config

    def print_cuda_info(self):
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        logger.info(f"visible_devices: {visible_devices}")
        logger.info("---------------------------------------------------")
        logger.info("CUDA Devices:")
        for i in range(0, torch.cuda.device_count()):
            logger.info("GPU: " + str(i))
            logger.info("Total GPU Memory: " + str(torch.cuda.get_device_properties(i).total_memory))
            logger.info("Reserved GPU Memory: " + str(torch.cuda.memory_reserved(i)))
            logger.info("Allocated GPU Memory: " + str(torch.cuda.memory_allocated(i)))
            logger.info("---------------------------------------------------")

    def merge(self, merged_name, first_adapter):
        base_model = self.load_base_model(False)

        peft_model = PeftModel.from_pretrained(base_model, first_adapter, torch_dtype=torch.bfloat16)
        logger.info(f"Merging adapter: {first_adapter} -> {merged_name}")
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_name)
        # get original tokenizer for save
        tmp_tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        tmp_tokenizer.save_pretrained(merged_name)
        try:
            tmp_tokenizer.save_vocabulary(merged_name)
        except:
            pass

    def load_base_model(self, gradient_checkpointing=True):
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_id,
                                                               torch_dtype=torch.bfloat16,
                                                               attn_implementation=self.attn_implementation,
                                                               device_map=self.device_map)
        print(self.base_model)
        self.base_model.generation_config.cache_implementation = None
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
        else:
            self.base_model.gradient_checkpointing_disable()
        return self.base_model

    def get_deepspeed_config(self, enable_offload_optimizer=True):
        cfg = {
            "zero_force_ds_cpu_optimizer": False,
            "bf16": {"enabled": "auto"},
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu"},
                "overlap_comm": True,
                "reduce_bucket_size": "auto",
                "sub_group_size": 1e9,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "steps_per_print": "auto",
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto"
        }

        if enable_offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

        return cfg
