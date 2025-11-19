#!/usr/bin/env python3
"""
NeMo Curator PII detection with F1/Precision/Recall evaluation
"""

import os
import json
import re
import torch
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from collections import defaultdict

print("🚀 Starting NeMo Curator PII detection and evaluation...")

MODEL_NAME = "nemo"
MODEL_METADATA = {
    "provider": "NVIDIA NeMo Curator",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Global variables for Dask client and modifier
client = None
modifier = None
modify_pipeline = None

def initialize_nemo():
    """Initialize Dask client and NeMo modifier."""
    global client, modifier, modify_pipeline
    
    if client is None:
        client = Client()
        print("📝 Dask client started:", client)

    # Configure NeMo modifier with better settings
    if modifier is None:
        modifier = PiiModifier(
            language="en",
            supported_entities=["PERSON", "ADDRESS", "DATE_TIME", "EMAIL_ADDRESS", "PHONE_NUMBER"],
            anonymize_action="replace",
            batch_size=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        modify_pipeline = Modify(modifier)

def normalize_entity(entity_type: str) -> str:
    """Normalize entity type to match ground truth format."""
    entity_mapping = {
        "PERSON": "PERSON",
        "ADDRESS": "LOCATION",
        "DATE_TIME": "DATE_TIME",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
    }
    return entity_mapping.get(entity_type, entity_type)

def process_single_text_with_nemo(text: str):
    """
    Process a single text with NeMo Curator and return the masked text.
    """
    if not text.strip():
        return ""
    
    # Initialize if not already done
    initialize_nemo()
        
    try:
        df = pd.DataFrame({"text": [text]})
        ddf = dd.from_pandas(df, npartitions=1)
        dataset = DocumentDataset(ddf)

        masked_dataset = modify_pipeline(dataset)
        masked_text = masked_dataset.df.compute()["text"].iloc[0]
        return masked_text
    except Exception as e:
        print(f"Error processing text with NeMo: {e}")
        return text  # Return original text if processing fails

def extract_entities_from_masked_text(original_text: str, masked_text: str):
    """
    Simple entity extraction by comparing original and masked text.
    Extract entities that were replaced with <ENTITY_TYPE> tags.
    """
    entities = []
    
    # Find all entity tags in masked text
    pattern = re.compile(r'<([^>]+)>')
    entity_matches = list(pattern.finditer(masked_text))
    
    if not entity_matches:
        return entities
    
    # For each entity tag, find what was replaced
    for match in entity_matches:
        entity_type = match.group(1)
        entity_start = match.start()
        entity_end = match.end()
        
        # Find the corresponding text in the original
        # by looking at the text before and after the entity
        before_text = masked_text[:entity_start].strip()
        after_text = masked_text[entity_end:].strip()
        
        # Find these texts in the original
        before_pos = original_text.find(before_text) if before_text else 0
        after_pos = original_text.find(after_text) if after_text else len(original_text)
        
        if before_pos != -1 and after_pos != -1:
            # Extract the text that was replaced
            replaced_text = original_text[before_pos + len(before_text):after_pos].strip()
            if replaced_text:
                entities.append({
                    "label": normalize_entity(entity_type),
                    "text": replaced_text,
                    "start": before_pos + len(before_text),
                    "end": after_pos,
                    "confidence": 0.8  # NeMo doesn't provide confidence, use default
                })
    
    return entities

def nemo_detect_pii(text: str):
    """Process text with NeMo Curator and return detected entities."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        masked_text = process_single_text_with_nemo(text)
        entities = extract_entities_from_masked_text(text, masked_text)
        return entities
    except Exception as e:
        print(f"Error processing text with NeMo: {e}")
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
                # Detect entities with NeMo Curator
                detected_pii = nemo_detect_pii(text)
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
        "model_used": "nemo-curator",
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
    try:
        # Initialize NeMo components
        initialize_nemo()
        
        input_json_path = "generated_test_cases_with_gt.json"
        output_json_path = "output_nemo_evaluation.json"
        
        process_dataset(input_json_path, output_json_path)
        print("\n✅ NeMo Curator detection and evaluation complete!")
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up Dask client
        if client is not None:
            client.close()
            print("📝 Dask client closed")


def detect_pii(text: str, language: str = "en"):
    """
    Entry point invoked by the Flask backend. The NeMo Curator pipeline
    currently supports English; the language parameter is kept for parity
    with the API.
    """
    _ = language
    return nemo_detect_pii(text)

