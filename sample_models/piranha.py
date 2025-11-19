#!/usr/bin/env python3
"""
Piranha PII detection with F1/Precision/Recall evaluation
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
from typing import List, Dict, Optional

# Metadata consumed by the backend model registry
MODEL_NAME = "piranha"
MODEL_METADATA = {
    "provider": "Hugging Face",
    "model_id": "iiiorg/piiranha-v1-detect-personal-information",
}

# ---- Configuration ----
model_name = "iiiorg/piiranha-v1-detect-personal-information"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for lazy-loaded model
_tokenizer = None
_model = None
_model_loaded = False

def _ensure_model_loaded():
    """Lazy-load the model on first use."""
    global _tokenizer, _model, _model_loaded
    if _model_loaded:
        return
    
    print("📝 Loading Piranha model (lazy initialization)...")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForTokenClassification.from_pretrained(model_name)
        _model.to(device)
        _model.eval()
        _model_loaded = True
        print(f"✅ Piranha model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load Piranha model: {e}")
        raise

# Mapping from the model's fine-grained label space (see config.json)
# to the coarse labels used in generated_test_cases_with_gt.json.
PIRANHA_TO_GT_LABEL = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "USERNAME": "PERSON",
    "CITY": "LOCATION",
    "STREET": "LOCATION",
    "BUILDINGNUM": "LOCATION",
    "ZIPCODE": "LOCATION",
    "DATEOFBIRTH": "DATE_TIME",
    "EMAIL": "EMAIL_ADDRESS",
    "TELEPHONENUM": "PHONE_NUMBER",
}

def normalize_entity(entity_type: str) -> Optional[str]:
    """
    Normalize model-specific labels to the ground-truth schema.
    Returns None when we don't have a compatible label; those predictions
    are dropped so they don't count as false positives.
    """
    return PIRANHA_TO_GT_LABEL.get(entity_type)

def extract_entities_fixed(text: str, model, tokenizer, device, max_len=512) -> List[Dict]:
    """
    Extract entities with proper handling of:
    - Subword tokenization
    - Long texts (chunking)
    - Overlapping chunks (deduplication)
    
    Returns entities in format: [{"label": str, "start": int, "end": int, "text": str}, ...]
    """
    if not text.strip():
        return []
    
    entities = []
    text_len = len(text)
    stride = max_len - 50  # Overlap to catch entities at chunk boundaries
    
    for chunk_start in range(0, text_len, stride):
        chunk_end = min(chunk_start + max_len, text_len)
        chunk_text = text[chunk_start:chunk_end]
        
        # Tokenize with offset mapping
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
        id2label = model.config.id2label
        
        # Build entities from BIO tags
        current_entity = None
        
        for token_idx, pred_id in enumerate(predictions):
            label = id2label[pred_id]
            token_start, token_end = offset_mapping[token_idx]
            
            # Skip special tokens and empty tokens
            if token_start == token_end:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Adjust offsets to original text position
            abs_start = chunk_start + token_start
            abs_end = chunk_start + token_end
            
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            # Extract base label (remove B-/I- prefix)
            if label.startswith("B-") or label.startswith("I-"):
                base_label = label[2:]
            else:
                base_label = label
            
            # Handle B- tags (start of entity)
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": base_label,
                    "start": abs_start,
                    "end": abs_end,
                }
            # Handle I- tags (continuation)
            elif label.startswith("I-"):
                if current_entity and current_entity["label"] == base_label:
                    # Extend current entity
                    current_entity["end"] = abs_end
                else:
                    # Orphan I- tag, treat as new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "label": base_label,
                        "start": abs_start,
                        "end": abs_end,
                    }
            # Handle tags without prefix
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": base_label,
                    "start": abs_start,
                    "end": abs_end,
                }
        
        if current_entity:
            entities.append(current_entity)
        
        # Break if we've processed the entire text
        if chunk_end >= text_len:
            break
    
    # Add text field and normalize labels (skip unsupported ones)
    normalized_entities = []
    for e in entities:
        mapped_label = normalize_entity(e["label"])
        if mapped_label is None:
            continue
        # Ensure entity has all required fields for backend compatibility
        entity_dict = {
            "label": mapped_label,
            "text": text[e["start"]:e["end"]] if "text" not in e else e["text"],
            "start": e["start"],
            "end": e["end"],
            "confidence": e.get("confidence", 0.8)  # Default confidence if missing
        }
        normalized_entities.append(entity_dict)
    
    # Remove duplicates from overlapping chunks
    normalized_entities.sort(key=lambda x: (x['start'], x['end']))
    unique_entities = []
    for entity in normalized_entities:
        is_duplicate = False
        for existing in unique_entities:
            if (entity['text'] == existing['text'] and 
                entity['label'] == existing['label'] and
                abs(entity['start'] - existing['start']) <= 5):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_entities.append(entity)
    
    return unique_entities

def piranha_detect_pii(text: str):
    """Process text with Piranha and return detected entities."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        _ensure_model_loaded()
        return extract_entities_fixed(text, _model, _tokenizer, device)
    except Exception as e:
        print(f"Error processing text with Piranha: {e}")
        return []

def mask_text(text: str, entities: list) -> str:
    """Replace detected entities with [LABEL]."""
    if not entities:
        return text
    
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    masked = text
    for e in sorted_entities:
        masked = masked[:e["start"]] + f"[{e['label']}]" + masked[e["end"]:]
    
    return masked

def calculate_metrics(tp, fp, fn):
    """Calculate Precision, Recall, and F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_predictions(ground_truth_list: list, predicted_list: list):
    """
    Compares ground truth labels to predicted labels.
    Uses simple overlap + label match.
    """
    tp = 0
    fp = 0
    fn = 0

    matched_gt_indices = set()

    # Iterate over predictions to find TPs and FPs
    for pred in predicted_list:
        found_match = False
        for i, gt in enumerate(ground_truth_list):
            if i in matched_gt_indices:
                continue

            # Check for overlap: max(start) < min(end)
            has_overlap = max(gt['start'], pred['start']) < min(gt['end'], pred['end'])

            # Check for label match
            is_label_match = gt['label'] == pred['label']

            if has_overlap and is_label_match:
                tp += 1
                matched_gt_indices.add(i)
                found_match = True
                break # Stop searching for this prediction

        if not found_match:
            fp += 1

    # Any ground truths not matched are False Negatives
    fn = len(ground_truth_list) - len(matched_gt_indices)

    metrics = calculate_metrics(tp, fp, fn)
    metrics.update({"tp": tp, "fp": fp, "fn": fn})

    return metrics

def process_dataset(input_file: str, output_file: str):
    """Process generated_test_cases_with_gt.json dataset and detect PII in all inputs."""
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {
        "summary": {},
        "evaluation_by_category": {},
        "results_by_category": {},
    }

    all_test_case_results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    total_inputs = 0

    print("Processing test cases by category...")

    for category, content in data.items():
        print(f"  -> Processing category: {category}")
        category_results = []
        category_tp, category_fp, category_fn = 0, 0, 0

        for test_case in content["test_cases"]:
            if not test_case or "text" not in test_case:
                continue

            total_inputs += 1
            text = test_case["text"]
            ground_truth = test_case["labels"]

            if not text.strip():
                detected_pii = []
                masked_text = ""
                evaluation = {"tp": 0, "fp": 0, "fn": len(ground_truth)}
            else:
                # Detect entities with Piranha
                detected_pii = piranha_detect_pii(text)
                # Mask text
                masked_text = mask_text(text, detected_pii)
                # Evaluate predictions
                evaluation = evaluate_predictions(ground_truth, detected_pii)

            # Store result
            result = {
                "category": category,
                "hypothesis": content["hypothesis"],
                "original_text": text,
                "ground_truth": ground_truth,
                "detected_pii": detected_pii,
                "evaluation": evaluation,
                "masked_text": masked_text,
            }

            category_results.append(result)
            all_test_case_results.append(result)

            # Aggregate stats
            category_tp += evaluation["tp"]
            category_fp += evaluation["fp"]
            category_fn += evaluation["fn"]

        # Store category-level results and metrics
        results["results_by_category"][category] = category_results
        category_metrics = calculate_metrics(category_tp, category_fp, category_fn)
        results["evaluation_by_category"][category] = {
            "hypothesis": content["hypothesis"],
            "metrics": category_metrics,
            "counts": {"tp": category_tp, "fp": category_fp, "fn": category_fn},
            "total_cases": len(category_results)
        }

        # Add to global stats
        total_tp += category_tp
        total_fp += category_fp
        total_fn += category_fn

    # Generate Final Summary
    print("Generating final summary...")

    global_metrics = calculate_metrics(total_tp, total_fp, total_fn)

    results["summary"] = {
        "model_used": model_name,
        "total_inputs_processed": total_inputs,
        "global_metrics": global_metrics,
        "global_counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_file}")
    print(f"📊 Global Summary:")
    print(f"   - Total inputs processed: {total_inputs}")
    print(f"   - Overall Precision: {global_metrics['precision']:.4f}")
    print(f"   - Overall Recall:    {global_metrics['recall']:.4f}")
    print(f"   - Overall F1-Score:  {global_metrics['f1']:.4f}")
    print("\n📊 Evaluation by Category (F1-Score):")

    # Sort categories by F1 score (worst first) to see problem areas
    sorted_eval = sorted(
        results["evaluation_by_category"].items(),
        key=lambda item: item[1]["metrics"]["f1"]
    )

    for category, eval_data in sorted_eval:
        print(f"   - {category:<35} F1: {eval_data['metrics']['f1']:.4f}")

    return results


if __name__ == "__main__":
    input_json_path = "generated_test_cases_with_gt.json"
    output_json_path = "output_piranha_evaluation.json"

    process_dataset(input_json_path, output_json_path)
    print("\n✅ Piranha detection and evaluation complete!")


def detect_pii(text: str, language: str = "en"):
    """
    Entry point used by the Flask backend. The `language` parameter is
    accepted for API compatibility but not used by this model.
    """
    _ = language  # unused
    return piranha_detect_pii(text)

