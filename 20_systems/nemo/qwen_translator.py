"""
Qwen support for NeMo's vLLM LLMTranslator (nemo.collections.asr.inference.nmt.llm_translator).

The NeMo version this was written against (3.0.0) accepts only EuroLLM-1.7B/9B-Instruct. The
NeMo@IWSLT 2026 paper's final En->De system uses Qwen3.5-27B. QwenLLMTranslator is NeMo's
LLMTranslator with three changes:
  * Qwen model names are accepted (NeMo's model check is bypassed);
  * the prompt is the paper's: its system prompt, and the user prompt
        Translate the following {src} source text to {tgt}:\\n{src}: {text}\\n{tgt}:
    rendered with the model's own chat template, thinking disabled (enable_thinking=False);
  * the translation prefix (and the context) is pre-filled in the assistant turn, so the model
    continues it, as NeMo does for EuroLLM.
Everything else (vLLM loading, batching, get_prefixes = longest common prefix + waitk) is inherited.

Unknown from the paper: whether the target prefix is also shown inside the user turn (its printed prompt
ends with "{tgt_lang}: {tgt_prefix}"); here it is only pre-filled in the assistant turn.
"""
import re
import sys

from nemo.collections.asr.inference.nmt import llm_translator as _nemo_llm
from nemo.collections.asr.inference.nmt.llm_translator import LLMTranslator

# The system prompt of the paper (Qwen models only).
QWEN_SYSTEM_PROMPT = """You are a professional machine translation assistant.
Translate the input text into the target language.
- Output only the translation.
- Do not complete or extend the text.
- The input may be incomplete; preserve incompleteness.
- Do not infer missing content.
- Stop immediately after translating.
- Preserve named entities, numbers, punctuation, and formatting."""

QWEN_USER_PROMPT = "Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src_text}\n{tgt_lang}:"


def is_qwen(model_name):
    return model_name.split("/")[-1].lower().startswith("qwen")


class QwenPromptTemplate:
    """Same interface as NeMo's PromptTemplate classes: format(...) and extract(...)."""

    def __init__(self, tokenizer, enable_thinking=False):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def format(self, src_lang, tgt_lang, src_prefix, tgt_prefix, src_context="", tgt_context=""):
        # same argument order and whitespace handling as NeMo's EuroLLMTranslatorPromptTemplate
        src_text = re.sub(r"\s+", " ", f"{src_context} {src_prefix}").strip()
        tgt_text = re.sub(r"\s+", " ", f"{tgt_context} {tgt_prefix}").strip()
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": QWEN_USER_PROMPT.format(src_lang=src_lang, tgt_lang=tgt_lang,
                                                                  src_text=src_text)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        return prompt + tgt_text  # the prefix is forced: the model continues after it

    @staticmethod
    def extract(response):
        return response.split("\n")[0]


class QwenLLMTranslator(LLMTranslator):
    def __init__(self, model_name, *args, enable_thinking=False, **kwargs):
        self._enable_thinking = enable_thinking
        supported = _nemo_llm.SUPPORTED_TRANSLATION_MODELS
        added = is_qwen(model_name) and model_name not in supported
        if added:  # NeMo's __init__ checks this list; allow the Qwen model for the duration of the call
            supported.append(model_name)
        try:
            super().__init__(model_name, *args, **kwargs)
        finally:
            if added:
                supported.remove(model_name)
        self._show_example()

    def get_prompt_template(self, model_name):  # NeMo calls this after loading the model
        if is_qwen(model_name):
            return QwenPromptTemplate(self.nmt_model.get_tokenizer(), self._enable_thinking)
        return super().get_prompt_template(model_name)

    def _show_example(self):
        if isinstance(self.prompt_template, QwenPromptTemplate):
            example = self.prompt_template.format(
                self.source_language, self.target_language, "Hello, how are", "Hallo, wie", "Good morning.",
                "Guten Morgen.")
            print("--- example prompt (as given to vLLM) ---\n" + example + "\n--- end of example ---",
                  file=sys.stderr, flush=True)
