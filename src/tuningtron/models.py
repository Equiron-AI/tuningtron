import logging
from transformers import AutoConfig, AutoTokenizer


logger = logging.getLogger(__name__)


class BaseModel:
    def __init__(self, base_model_id, config):
        self.base_model_id = base_model_id
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=False)
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
        if self.tokenizer.pad_token:
            logger.info("Pad token: " + self.tokenizer.pad_token)
        else:
            logging.warn("Pad token not found")

        if self.tokenizer.bos_token:
            logger.info("Bos token: " + self.tokenizer.bos_token)
        else:
            logging.warn("Bos token not found")

        if self.tokenizer.eos_token:
            logger.info("Eos token: " + self.tokenizer.eos_token)
        else:
            logging.warn("Eos token not found")

    def apply_chat_template(self, record):
        pass


class GemmaModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.response_template = "<start_of_turn>model"

    def apply_chat_template(self, record):
        if "text" in record and record["text"]:
            return record["text"].strip()
        chat = [
            {"role": "user",      "content": record["instruct"].strip() + "\n\n\n" + record["input"].strip()},
            {"role": "assistant", "content": record["output"].strip()}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False)


class CohereModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)
        self.response_template = "<|CHATBOT_TOKEN|>"

    def apply_chat_template(self, record):
        if "text" in record and record["text"]:
            return record["text"].strip()
        chat = [
            {"role": "system",    "content": record["instruct"].strip()},
            {"role": "user",      "content": record["input"].strip()},
            {"role": "assistant", "content": record["output"].strip()}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False)


class QwenModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)
        self.response_template = "<|im_start|>assistant"

    def apply_chat_template(self, record):
        if "text" in record and record["text"]:
            return record["text"].strip()
        chat = [
            {"role": "system",    "content": record["instruct"].strip()},
            {"role": "user",      "content": record["input"].strip()},
            {"role": "assistant", "content": record["output"].strip()}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False)


class ModelsFactory:
    def __init__(self):
        pass

    def get_model_config(self, base_model_id):
        config = AutoConfig.from_pretrained(base_model_id)

        if config.model_type.startswith("gemma"):
            return GemmaModel(base_model_id, config)
        elif config.model_type.startswith("cohere"):
            return CohereModel(base_model_id, config)
        elif config.model_type.startswith("qwen"):
            return QwenModel(base_model_id, config)
        else:
            raise Exception("Unsupported model type: " + base_model_id)
