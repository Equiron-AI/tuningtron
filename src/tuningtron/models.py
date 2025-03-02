import logging
from transformers import AutoConfig, AutoTokenizer


logger = logging.getLogger(__name__)


class BaseModel:
    def __init__(self, base_model_id, config):
        self.base_model_id = base_model_id
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
        if self.tokenizer.pad_token:
            logger.info("Pad token: " + self.tokenizer.pad_token)
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.warn("Pad token not found, setting this to eos token")

        if self.tokenizer.bos_token:
            logger.info("Bos token: " + self.tokenizer.bos_token)
        else:
            logging.warn("Bos token not found")

        if self.tokenizer.eos_token:
            logger.info("Eos token: " + self.tokenizer.eos_token)
        else:
            logging.warn("Eos token not found")

        logger.info("Padding size: " + self.tokenizer.padding_side)

    def prepare_chat_template(self, record):
        messages = []

        if record["instruct"].strip():
            messages.append({"role": "system", "content": record["instruct"].strip()})

        messages.append({"role": "user", "content": record["input"].strip()})
        messages.append({"role": "assistant", "content": record["output"].strip()})

        return messages

    def apply_chat_template(self, record):
        chat = self.prepare_chat_template(record)
        text = self.tokenizer.apply_chat_template(chat, tokenize=False)
        if self.tokenizer.bos_token:
            text = text.replace(self.tokenizer.bos_token, "")
        return text


class GemmaModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)

    def prepare_chat_template(self, record):
        if record["instruct"].strip():
            instruct = record["instruct"].strip() + "\n\n\n"
        else:
            instruct = ""

        return [
            {"role": "user",      "content": instruct + record["input"].strip()},
            {"role": "assistant", "content": record["output"].strip()}
        ]


class CohereModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)


class QwenModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)


class MistralModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)


class YandexModel(BaseModel):
    def __init__(self, base_model_id, config):
        BaseModel.__init__(self, base_model_id, config)

    def apply_chat_template(self, record):
        if record["instruct"].strip():
            system = record["instruct"].strip()
        else:
            system = ""

        user = record["input"].strip()
        assistant = record["output"].strip()

        if system:
            return f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""
        else:
            return f"""<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""


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
        elif config.model_type.startswith("mistral"):
            return MistralModel(base_model_id, config)
        elif config.model_type.startswith("llama"):
            return YandexModel(base_model_id, config)
        else:
            raise Exception("Unsupported model type: " + base_model_id)
