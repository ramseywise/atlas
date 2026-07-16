# cap-rlhf — Templates for rlhf-builder subagent

## File: {OUTPUT_DIR}/rlhf/preference_collector.py

```python
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PREFERENCES_DIR = Path("data/rlhf")
_PREFERENCES_FILE = _PREFERENCES_DIR / "{AGENT_NAME}_preferences.jsonl"


class PreferenceCollector:
    """Collect and store human preference pairs for RLHF training.

    Each preference pair records a query alongside two agent responses:
    - chosen: the preferred (better) response
    - rejected: the less preferred response

    Pairs are appended to a JSONL file for later use by RewardTrainer.

    Usage:
        collector = PreferenceCollector()
        collector.add_pair(
            query="How do I get started with this?",
            chosen_response="Here is a clear explanation of how to get started...",
            rejected_response="I don't know.",
            annotator_id="human-reviewer-1",
        )
    """

    def __init__(self, output_path: str | Path | None = None) -> None:
        self.output_path = Path(output_path) if output_path else _PREFERENCES_FILE
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_pair(
        self,
        query: str,
        chosen_response: str,
        rejected_response: str,
        annotator_id: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a preference pair to the JSONL store.

        Args:
            query: The user query both responses answered.
            chosen_response: The preferred response text.
            rejected_response: The less preferred response text.
            annotator_id: Identifier of the human annotator.
            metadata: Optional dict with extra fields (e.g., session_id, rating).

        Returns:
            The generated pair_id (UUID string).
        """
        pair_id = str(uuid.uuid4())
        record = {
            "pair_id": pair_id,
            "query": query,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "annotator_id": annotator_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "agent": "{AGENT_NAME}",
            **(metadata or {}),
        }

        with self.output_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        logger.debug("Recorded preference pair %s from annotator %s", pair_id, annotator_id)
        return pair_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_pairs(self) -> list[dict[str, Any]]:
        """Load all recorded preference pairs from disk."""
        if not self.output_path.exists():
            return []

        pairs: list[dict[str, Any]] = []
        with self.output_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed preference record: %s", exc)

        logger.info("Loaded %d preference pairs from %s", len(pairs), self.output_path)
        return pairs

    def stats(self) -> dict[str, Any]:
        """Return basic statistics about the collected preferences."""
        pairs = self.load_pairs()
        annotators = {p.get("annotator_id") for p in pairs}
        return {
            "total_pairs": len(pairs),
            "annotator_count": len(annotators),
            "annotators": sorted(annotators),
            "output_path": str(self.output_path),
        }
```

## File: {OUTPUT_DIR}/rlhf/reward_trainer.py

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports
try:
    from trl import RewardConfig, RewardTrainer as TRLRewardTrainer  # type: ignore[import]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import]
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

try:
    from datasets import Dataset  # type: ignore[import]
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


class RewardTrainer:
    """Train a reward model from human preference pairs and use it to score responses.

    The reward model is a classification head on top of a pretrained LM that
    outputs a scalar reward score. It's trained on (chosen, rejected) pairs using
    the Bradley-Terry preference model loss (TRL RewardTrainer).

    Usage:
        trainer = RewardTrainer(base_model="google/gemma-2b-it", output_dir="models/reward")
        trainer.train_from_jsonl("data/rlhf/{AGENT_NAME}_preferences.jsonl")
        score = trainer.score("Here is how this feature works...")
        winner = trainer.compare(response_a, response_b)
    """

    def __init__(
        self,
        base_model: str = "google/gemma-2b-it",
        output_dir: str = "models/{AGENT_NAME}-reward",
        max_length: int = 512,
    ) -> None:
        if not TRL_AVAILABLE:
            raise ImportError(
                "trl is required for reward training. Run: pip install trl torch accelerate"
            )
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_preference_dataset(self, jsonl_path: str) -> Any:
        """Load a JSONL preference file into a HuggingFace Dataset.

        Expected JSONL format: {"query": "...", "chosen": "...", "rejected": "..."}
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets is required. Run: pip install datasets")

        from {AGENT_NAME}.rlhf.preference_collector import PreferenceCollector  # noqa: PLC0415

        collector = PreferenceCollector(output_path=jsonl_path)
        pairs = collector.load_pairs()

        if not pairs:
            raise ValueError(f"No preference pairs found in {jsonl_path}")

        # TRL RewardTrainer expects 'chosen' and 'rejected' as full conversation strings
        formatted = [
            {
                "chosen": f"Query: {p['query']}\nResponse: {p['chosen']}",
                "rejected": f"Query: {p['query']}\nResponse: {p['rejected']}",
            }
            for p in pairs
        ]

        return Dataset.from_list(formatted)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_jsonl(
        self,
        jsonl_path: str,
        epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
    ) -> None:
        """Train the reward model from a JSONL preferences file.

        Args:
            jsonl_path: Path to the preference JSONL file.
            epochs: Number of training epochs.
            batch_size: Per-device batch size.
            learning_rate: AdamW learning rate.
        """
        import torch  # noqa: PLC0415

        dataset = self.load_preference_dataset(jsonl_path)

        logger.info("Loading reward model base: %s", self.base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=1,  # Scalar reward output
            torch_dtype=dtype,
            device_map="auto",
        )

        config = RewardConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=5,
            save_strategy="epoch",
            max_length=self.max_length,
            report_to="none",
        )

        trainer = TRLRewardTrainer(
            model=self._model,
            args=config,
            tokenizer=self._tokenizer,
            train_dataset=dataset,
        )

        logger.info("Training reward model on %d pairs", len(dataset))
        trainer.train()
        trainer.save_model(str(self.output_dir))
        logger.info("Reward model saved to %s", self.output_dir)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _load_for_inference(self) -> None:
        """Load the saved reward model for inference."""
        import torch  # noqa: PLC0415

        model_path = self.output_dir
        if not model_path.exists():
            raise FileNotFoundError(
                f"Reward model not found at {model_path}. Train it first."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self._model.eval()

    def score(self, response: str) -> float:
        """Score a single response string. Returns a scalar reward (higher = better).

        Args:
            response: The agent response text to score.

        Returns:
            Scalar reward value.
        """
        import torch  # noqa: PLC0415

        if self._model is None:
            self._load_for_inference()

        inputs = self._tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        return float(outputs.logits.squeeze().item())

    def compare(self, response_a: str, response_b: str) -> str:
        """Compare two responses and return 'a' or 'b' for the preferred one.

        Args:
            response_a: First candidate response.
            response_b: Second candidate response.

        Returns:
            'a' if response_a is preferred, 'b' if response_b is preferred.
        """
        score_a = self.score(response_a)
        score_b = self.score(response_b)
        logger.debug("Reward scores — a=%.4f  b=%.4f", score_a, score_b)
        return "a" if score_a >= score_b else "b"
```

## File: {OUTPUT_DIR}/rlhf/ppo_trainer.py

```python
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Lazy imports
try:
    from trl import PPOConfig, PPOTrainer  # type: ignore[import]
    from transformers import AutoModelForCausalLMWithValueHead, AutoTokenizer  # type: ignore[import]
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False


class PPOFinetuner:
    """Online reinforcement learning fine-tuner using PPO + reward model.

    Wraps TRL PPOTrainer. Given a list of queries and an agent function, generates
    responses, scores them with the reward model, and updates the policy.

    Usage:
        reward_trainer = RewardTrainer(...)
        ppo = PPOFinetuner(
            base_model="google/gemma-2b-it",
            reward_fn=reward_trainer.score,
            output_dir="models/{AGENT_NAME}-ppo",
        )
        metrics = await ppo.train_episode(queries=["How do I get started?"], agent_fn=runner.run_text)
    """

    def __init__(
        self,
        base_model: str,
        reward_fn: Callable[[str], float],
        output_dir: str = "models/{AGENT_NAME}-ppo",
        max_new_tokens: int = 256,
        batch_size: int = 4,
        learning_rate: float = 1.4e-5,
        mini_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        if not TRL_AVAILABLE:
            raise ImportError(
                "trl is required for PPO fine-tuning. Run: pip install trl torch accelerate"
            )
        self.base_model = base_model
        self.reward_fn = reward_fn
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._trainer: Any = None
        self._config = PPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _init_trainer(self) -> None:
        import torch  # noqa: PLC0415

        logger.info("Loading PPO policy model: %s", self.base_model)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # PPOTrainer requires a value head on the model
        self._model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._trainer = PPOTrainer(
            config=self._config,
            model=self._model,
            tokenizer=self._tokenizer,
        )

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------

    async def train_episode(
        self,
        queries: list[str],
        agent_fn: Callable[[str], Any],
    ) -> dict[str, float]:
        """Run one PPO episode: generate responses, score, update policy.

        Args:
            queries: List of query strings for this episode.
            agent_fn: Async callable that takes a query string and returns a response string.
                      Can be a thin wrapper around AgentRunner.run() that returns message text.
            Returns:
                Dict with episode metrics: mean_reward, ppo_loss, value_loss, etc.
        """
        import asyncio  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if self._trainer is None:
            self._init_trainer()

        # Generate responses from the agent (policy)
        responses = await asyncio.gather(*(agent_fn(q) for q in queries))

        # Score each response with the reward model
        rewards = [torch.tensor(self.reward_fn(r)) for r in responses]

        logger.info(
            "Episode rewards — mean=%.4f  min=%.4f  max=%.4f",
            sum(r.item() for r in rewards) / len(rewards),
            min(r.item() for r in rewards),
            max(r.item() for r in rewards),
        )

        # Tokenize queries and responses for PPO step
        query_tensors = [
            self._tokenizer.encode(q, return_tensors="pt").squeeze()
            for q in queries
        ]
        response_tensors = [
            self._tokenizer.encode(r, return_tensors="pt").squeeze()
            for r in responses
        ]

        # PPO update step
        stats = self._trainer.step(query_tensors, response_tensors, rewards)

        episode_metrics = {
            "mean_reward": float(sum(r.item() for r in rewards) / len(rewards)),
            "ppo_loss": float(stats.get("ppo/loss/total", 0.0)),
            "value_loss": float(stats.get("ppo/loss/value", 0.0)),
            "policy_loss": float(stats.get("ppo/loss/policy", 0.0)),
            "mean_kl": float(stats.get("ppo/mean_scores", 0.0)),
        }
        logger.info("Episode metrics: %s", episode_metrics)
        return episode_metrics

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save the current PPO policy model to disk."""
        if self._trainer is None:
            raise RuntimeError("No training has run yet.")
        self._trainer.save_pretrained(self.output_dir)
        self._tokenizer.save_pretrained(self.output_dir)
        logger.info("PPO model saved to %s", self.output_dir)
```
