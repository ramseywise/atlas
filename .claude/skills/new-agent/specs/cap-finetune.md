# cap-finetune — Templates for finetune-builder subagent

## File: {OUTPUT_DIR}/finetune/dataset.py

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports — only needed when this module is used
try:
    from datasets import Dataset, DatasetDict  # type: ignore[import]
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


EVAL_DATASET_PATH = Path(__file__).parent.parent / "evals" / "datasets" / "eval.jsonl"

# Instruction template — follows Alpaca-style instruction tuning format
_INSTRUCTION_TEMPLATE = (
    "You are a helpful support assistant. "
    "Answer the user's question accurately and concisely based on the context provided."
)


class FinetuneDataset:
    """Loads the agent's eval JSONL and converts it to HuggingFace Dataset format.

    Converts each eval item into an instruction/input/output training example
    suitable for supervised fine-tuning (SFT) with HF Trainer or TRL SFTTrainer.

    Usage:
        ds = FinetuneDataset()
        ds.load(path=str(EVAL_DATASET_PATH))
        train_ds, eval_ds = ds.split(train_ratio=0.8)
    """

    def __init__(self) -> None:
        self._raw_items: list[dict[str, Any]] = []
        self._hf_dataset: Any | None = None  # datasets.Dataset

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str | Path | None = None) -> "FinetuneDataset":
        """Load JSONL items from disk.

        Args:
            path: Path to the JSONL eval dataset. Defaults to the agent's default eval set.
        """
        source = Path(path) if path else EVAL_DATASET_PATH

        if not source.exists():
            raise FileNotFoundError(f"Dataset not found: {source}")

        items: list[dict[str, Any]] = []
        with source.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed line: %s", exc)

        # Only use items with golden answers (answerable intent)
        answerable = [
            item for item in items
            if item.get("expected_intent") == "answerable"
            and item.get("golden_answer")
        ]
        logger.info(
            "Loaded %d items (%d answerable) from %s",
            len(items),
            len(answerable),
            source,
        )
        self._raw_items = answerable
        return self

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    def format_example(self, item: dict[str, Any]) -> dict[str, str]:
        """Convert a single eval item to instruction/input/output format.

        Args:
            item: Raw eval item with 'query' and 'golden_answer' keys.

        Returns:
            Dict with 'instruction', 'input', 'output' keys.
        """
        return {
            "instruction": _INSTRUCTION_TEMPLATE,
            "input": item["query"],
            "output": item.get("golden_answer", ""),
            # Include original fields for debugging
            "domain": item.get("domain", ""),
            "expected_intent": item.get("expected_intent", ""),
        }

    def to_hf_dataset(self) -> Any:
        """Convert loaded items to a HuggingFace Dataset object.

        Raises:
            ImportError: If the `datasets` package is not installed.
            RuntimeError: If no items are loaded yet.
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "datasets is required. Run: pip install datasets"
            )
        if not self._raw_items:
            raise RuntimeError("No items loaded. Call .load() first.")

        formatted = [self.format_example(item) for item in self._raw_items]
        self._hf_dataset = Dataset.from_list(formatted)
        return self._hf_dataset

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[Any, Any]:
        """Split the dataset into train and eval subsets.

        Args:
            train_ratio: Fraction of data to use for training (0.0–1.0).
            seed: Shuffle seed for reproducibility.

        Returns:
            Tuple of (train_dataset, eval_dataset) as HuggingFace Dataset objects.
        """
        ds = self._hf_dataset or self.to_hf_dataset()
        split = ds.train_test_split(test_size=1.0 - train_ratio, seed=seed)
        logger.info(
            "Dataset split: %d train, %d eval",
            len(split["train"]),
            len(split["test"]),
        )
        return split["train"], split["test"]
```

## File: {OUTPUT_DIR}/finetune/trainer.py

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore[import]
    from transformers import Trainer as HFTrainer  # type: ignore[import]
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

_DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "v_proj"],
}

# Current prompt version — saved in model card metadata
PROMPT_VERSION = "0.1.0"


class FineTuner:
    """Supervised fine-tuning wrapper with optional PEFT LoRA.

    Usage:
        tuner = FineTuner(
            base_model="google/gemma-2b-it",
            output_dir="models/{AGENT_NAME}-sft",
        )
        train_ds, eval_ds = FinetuneDataset().load().split()
        tuner.train(train_ds, eval_ds=eval_ds, epochs=3)
        metrics = tuner.evaluate(eval_ds)
        tuner.push_to_hub("your-org/{AGENT_NAME}-sft")
    """

    def __init__(
        self,
        base_model: str,
        output_dir: str,
        lora_config: dict[str, Any] | None = None,
        use_lora: bool = True,
    ) -> None:
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers is required. Run: pip install transformers torch accelerate"
            )
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.lora_config_dict = lora_config or _DEFAULT_LORA_CONFIG
        self.use_lora = use_lora and PEFT_AVAILABLE
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        import torch  # noqa: PLC0415

        logger.info("Loading base model: %s", self.base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Add pad token if missing
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            device_map="auto",
        )

        if self.use_lora:
            logger.info("Applying LoRA config: %s", self.lora_config_dict)
            lora_cfg = LoraConfig(**self.lora_config_dict)
            self._model = get_peft_model(self._model, lora_cfg)
            self._model.print_trainable_parameters()

    def _format_for_training(self, example: dict[str, Any]) -> dict[str, Any]:
        """Format an instruction/input/output example as a single prompt string."""
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
        tokenized = self._tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: Any,
        eval_ds: Any | None = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
    ) -> None:
        """Fine-tune the model on the given HuggingFace Dataset.

        Args:
            dataset: Training dataset (HF Dataset with instruction/input/output).
            eval_ds: Optional evaluation dataset.
            epochs: Number of training epochs.
            batch_size: Per-device training batch size.
            learning_rate: AdamW learning rate.
        """
        if self._model is None:
            self._load_model()

        # Tokenize
        train_tokenized = dataset.map(self._format_for_training, batched=False)
        eval_tokenized = eval_ds.map(self._format_for_training, batched=False) if eval_ds else None

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch" if eval_tokenized else "no",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=bool(eval_tokenized),
            report_to="none",  # set to "langsmith" or "wandb" if desired
            fp16=False,
            bf16=True,
        )

        trainer = HFTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=self._tokenizer,
        )

        logger.info("Starting training: %d examples, %d epochs", len(dataset), epochs)
        trainer.train()
        logger.info("Training complete. Saving model to %s", self.output_dir)
        trainer.save_model(str(self.output_dir))
        self._tokenizer.save_pretrained(str(self.output_dir))
        self._save_model_card()

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, dataset: Any) -> dict[str, float]:
        """Run evaluation on the given dataset.

        Returns:
            Dict with at minimum 'eval_loss'. Additional metrics if a compute_metrics
            function is wired in.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .train() or load a saved model first.")

        eval_tokenized = dataset.map(self._format_for_training, batched=False)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=4,
            report_to="none",
        )
        trainer = HFTrainer(
            model=self._model,
            args=training_args,
            eval_dataset=eval_tokenized,
            tokenizer=self._tokenizer,
        )
        metrics = trainer.evaluate()
        logger.info("Eval metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Hub push
    # ------------------------------------------------------------------

    def push_to_hub(self, repo_id: str) -> None:
        """Push the fine-tuned model and tokenizer to HuggingFace Hub.

        Args:
            repo_id: HF Hub repo (e.g. "your-org/{AGENT_NAME}-sft").
        """
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        logger.info("Pushing model to HuggingFace Hub: %s", repo_id)
        self._model.push_to_hub(repo_id)
        self._tokenizer.push_to_hub(repo_id)
        logger.info("Model pushed to %s", repo_id)

    # ------------------------------------------------------------------
    # Model card
    # ------------------------------------------------------------------

    def _save_model_card(self) -> None:
        """Write a model card with PROMPT_VERSION metadata."""
        card_path = self.output_dir / "README.md"
        metadata = {
            "base_model": self.base_model,
            "agent": "{AGENT_NAME}",
            "prompt_version": PROMPT_VERSION,
            "use_lora": self.use_lora,
        }
        content = (
            f"# {'{AGENT_NAME}'} Fine-tuned Model\n\n"
            f"Base model: `{self.base_model}`\n\n"
            f"## Metadata\n\n```json\n{json.dumps(metadata, indent=2)}\n```\n"
        )
        card_path.write_text(content)
        logger.debug("Model card written to %s", card_path)
```
