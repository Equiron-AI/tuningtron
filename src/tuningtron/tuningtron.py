import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
import torch
import logging
import re

from liger_kernel.transformers import AutoLigerKernelForCausalLM
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, PeftModel


logger = logging.getLogger(__name__)


class Tuningtron:
    def __init__(self,
                 base_model_id,
                 enable_deepspeed=True,
                 enable_offload_optimizer=True,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]):
        self.base_model_id = base_model_id
        self.model_config = AutoConfig.from_pretrained(base_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.target_modules = target_modules
        self.device_map = "auto"
        self.deepspeed = None
        self.attn_implementation = "flash_attention_2"
        self.dtype = torch.bfloat16

        self.show_cuda_info()
        self.show_tokenizer_info()

        if enable_deepspeed:
            self.device_map = None
            self.deepspeed = self.get_deepspeed_config(enable_offload_optimizer)
            logger.info("deepspeed: enabled")

    def create_optimizer(self, model, learning_rate):
        LM_HEAD_RE = re.compile(r"(?:^|\.)(lm_head)(?:\.|$)")

        head_params, other_params = [], []
        for name, p in model.named_parameters():
            if p.requires_grad and "lora_" in name:
                (head_params if LM_HEAD_RE.search(name) else other_params).append(p)

        return AdamW(
            [
                {"params": other_params, "lr": learning_rate, "weight_decay": 0.01},
                {"params": head_params,  "lr": min(2e-5, learning_rate/3), "weight_decay": 0.01}  # lm_head — меньший LR
            ], betas=(0.9, 0.95), eps=1e-8)

    def sft(self,
            dataset,
            adapter_name,
            do_eval=False,
            lora_rank=16,
            lora_alpha=None,
            num_train_epochs=1,
            backpropagation_batch_size=1,
            gradient_accum_steps=1,
            learning_rate=1e-5,
            warmup_ratio=0.1):
        dataset = datasets.load_dataset(dataset, split="train")
        train_dataset, eval_dataset = self.prepare_datasets(dataset, do_eval)

        logger.info("Dataset example row after appy chat template:")
        logger.info(self.tokenizer.apply_chat_template(train_dataset["messages"][0], tokenize=False, add_generation_prompt=False))
        logger.info("---------------------------------------------")
        logger.info("Dataset example row after tokenize:")
        logger.info(self.tokenizer.apply_chat_template(train_dataset["messages"][0], tokenize=True, add_generation_prompt=False))

        args = SFTConfig(**self.prepare_args(num_train_epochs, warmup_ratio, backpropagation_batch_size, gradient_accum_steps))
        args.assistant_only_loss = True
        args.use_liger_kernel = True
        logger.info(str(args))

        base_model = self.load_base_model()
        cfg = self.get_lora_config(lora_rank, lora_alpha)
        peft_model = get_peft_model(base_model, cfg)
        logger.info(str(peft_model.get_model_status()))

        optimizer = self.create_optimizer(peft_model, learning_rate)

        trainer = SFTTrainer(model=peft_model,
                             tokenizer=self.tokenizer,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             optimizers=(optimizer, None),
                             args=args)
        trainer.train()
        trainer.save_model(adapter_name)

    def dpo(self,
            dataset,
            adapter_name,
            do_eval=False,
            lora_rank=16,
            lora_alpha=None,
            num_train_epochs=1,
            backpropagation_batch_size=1,
            gradient_accum_steps=1,
            learning_rate=1e-5,
            warmup_ratio=0.1):
        dataset = datasets.load_dataset(dataset, split="train")
        train_dataset, eval_dataset = self.prepare_datasets(dataset, do_eval)

        logger.info("Dataset example row after appy chat template:")
        logger.info("Chosen ------>")
        logger.info(self.tokenizer.apply_chat_template(train_dataset["chosen"][0], tokenize=False, add_generation_prompt=False))
        logger.info("Rejected ------>")
        logger.info(self.tokenizer.apply_chat_template(train_dataset["rejected"][0], tokenize=False, add_generation_prompt=False))
        logger.info("---------------------------------------------")
        logger.info("Dataset example row after tokenize:")
        logger.info("Chosen ------>")
        logger.info(self.tokenizer.apply_chat_template(train_dataset["chosen"][0], tokenize=True, add_generation_prompt=False))
        logger.info("Rejected ------>")
        logger.info(self.tokenizer.apply_chat_template(train_dataset["rejected"][0], tokenize=True, add_generation_prompt=False))

        args = DPOConfig(**self.prepare_args(num_train_epochs, warmup_ratio, backpropagation_batch_size, gradient_accum_steps))
        args.optimize_device_cache = True
        args.use_liger_loss = True
        args.use_num_logits_to_keep = True
        args.max_prompt_length = None
        args.max_completion_length = None
        logger.info(args)

        base_model = self.load_base_model()
        cfg = self.get_lora_config(lora_rank, lora_alpha)
        peft_model = get_peft_model(base_model, cfg)
        logger.info(peft_model.get_model_status())

        optimizer = self.create_optimizer(peft_model, learning_rate)

        trainer = DPOTrainer(model=peft_model,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             tokenizer=self.tokenizer,
                             optimizers=(optimizer, None),
                             args=args)
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

    def prepare_args(self, num_train_epochs, warmup_ratio, batch_size, gradient_accum_steps):
        return {
            "output_dir": ".",
            "num_train_epochs": num_train_epochs,
            "logging_steps": 1,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "save_strategy": "no",
            "bf16": True,
            "warmup_ratio": warmup_ratio,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accum_steps,
            "eval_accumulation_steps": 1,
            "padding_free": True,
            "max_length": None,
            "deepspeed": self.deepspeed
        }

    def get_lora_config(self, rank, lora_alpha):
        lora_alpha = lora_alpha if lora_alpha else rank

        pat_string = r".*lm_head"
        pat = re.compile(pat_string)
        matched = [n for n, _ in self.base_model.named_modules() if pat.search(n)]
        logger.info(f"Matched modules for lm_head: {matched}")

        config = LoraConfig(r=rank,
                            lora_alpha=lora_alpha,
                            rank_pattern={pat_string: max(1, rank // 4)},  # для экономии памяти, т.к. для lm_head не требуются большие матрицы
                            alpha_pattern={pat_string: max(1, (lora_alpha or rank) // 4)},  # для экономии памяти, т.к. для lm_head не требуются большие матрицы
                            target_modules=self.target_modules,
                            task_type="CAUSAL_LM")
        logger.info("Lora config:" + str(config))
        return config

    def show_cuda_info(self):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
        logger.info(f"visible_devices: {visible_devices}")
        logger.info("---------------------------------------------------")
        logger.info("CUDA Devices:")
        for i in range(0, torch.cuda.device_count()):
            logger.info("GPU: " + str(i))
            logger.info("Total GPU Memory: " + str(torch.cuda.get_device_properties(i).total_memory))
            logger.info("Reserved GPU Memory: " + str(torch.cuda.memory_reserved(i)))
            logger.info("Allocated GPU Memory: " + str(torch.cuda.memory_allocated(i)))
            logger.info("---------------------------------------------------")

    def show_tokenizer_info(self):
        logger.info("Pad token: " + self.tokenizer.pad_token)
        logger.info("Bos token: " + self.tokenizer.bos_token)
        logger.info("Eos token: " + self.tokenizer.eos_token)
        logger.info("Padding size: " + self.tokenizer.padding_side)

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
        self.base_model = AutoLigerKernelForCausalLM.from_pretrained(self.base_model_id,
                                                                     torch_dtype=self.dtype,
                                                                     attn_implementation=self.attn_implementation,
                                                                     device_map=self.device_map)
        logger.info(self.base_model)

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
                "sub_group_size": 1e6,
                "use_all_reduce_for_fetch_params": True,
                "max_live_parameters": 1e6,
                "max_reuse_distance": 1e6,
                "prefetch_bucket_size": "auto",
                "param_persistence_threshold": "auto",
                "gather_16bit_weights_on_model_save": True
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto"
        }

        if enable_offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

        return cfg
