#!/usr/bin/env python3
"""
HuggingFace NER PII detection with F1/Precision/Recall evaluation
"""

import json
from transformers import pipeline
from collections import defaultdict

MODEL_NAME = "bdmbz"
MODEL_METADATA = {
    "provider": "Hugging Face",
    "pipeline": "ner",
    "aggregation": "simple",
}

# Global variable for lazy-loaded pipeline
_ner = None
_ner_loaded = False

def _ensure_ner_loaded():
    """Lazy-load the NER pipeline on first use."""
    global _ner, _ner_loaded
    if _ner_loaded:
        return
    
    print("📝 Initializing HuggingFace NER pipeline (lazy initialization)...")
    try:
        _ner = pipeline("ner", aggregation_strategy="simple")
        _ner_loaded = True
        print("✅ HuggingFace NER pipeline loaded")
    except Exception as e:
        print(f"❌ Failed to load HuggingFace NER pipeline: {e}")
        raise

def normalize_entity(entity_type: str) -> str:
    """Normalize entity type to match ground truth format."""
    entity_mapping = {
        "PERSON": "PERSON",
        "PER": "PERSON",
        "LOCATION": "LOCATION",
        "LOC": "LOCATION",
        "DATE_TIME": "DATE_TIME",
        "DATE": "DATE_TIME",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "ORG": "ORGANIZATION",
    }
    return entity_mapping.get(entity_type, entity_type)

def huggingface_detect_pii(text: str):
    """Process text with HuggingFace NER and return detected entities."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        _ensure_ner_loaded()
        entities = _ner(text)
        detected = []
        for ent in entities:
            if 'start' in ent and 'end' in ent and 'entity_group' in ent:
                detected.append({
                    "text": ent['word'],
                    "start": ent['start'],
                    "end": ent['end'],
                    "label": normalize_entity(ent['entity_group']),
                    "confidence": float(ent.get('score', 0.0))
                })
        return detected
    except Exception as e:
        print(f"Error processing text: {e}")
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
                # Detect entities with HuggingFace NER
                detected_pii = huggingface_detect_pii(text)
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
        "model_used": "huggingface-ner",
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
    output_json_path = "output_huggingface_evaluation.json"

    process_dataset(input_json_path, output_json_path)
    print("\n✅ HuggingFace NER detection and evaluation complete!")


def detect_pii(text: str, language: str = "en"):
    """
    Entry point consumed by the backend model registry. The Hugging Face
    pipeline does not currently distinguish languages, but we keep the
    parameter for interface compatibility.
    """
    _ = language
    return huggingface_detect_pii(text)

