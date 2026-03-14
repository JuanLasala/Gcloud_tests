from __future__ import annotations

from typing import Dict, Optional, Tuple

from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification


def load_hf_model(
	model_name: str,
	num_labels: int,
	id2label: Dict[int, str],
	label2id: Dict[str, int],
	in_channels: Optional[int] = None,
) -> Tuple[nn.Module, AutoImageProcessor]:
	"""Build an RGB HF image-classification model + processor with label mappings.

	The ``in_channels`` parameter is kept only for backward compatibility with
	existing call sites, but it is intentionally ignored in this RGB-only setup.
	"""
	_ = in_channels

	model = AutoModelForImageClassification.from_pretrained(
		model_name,
		num_labels=num_labels,
		id2label=id2label,
		label2id=label2id,
		ignore_mismatched_sizes=True,
	)

	processor = AutoImageProcessor.from_pretrained(model_name)
	return model, processor
