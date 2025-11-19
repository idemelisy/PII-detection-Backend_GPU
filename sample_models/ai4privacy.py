#!/usr/bin/env python3
"""
AI4Privacy PII detection with F1/Precision/Recall evaluation
"""

import json
from ai4privacy import protect
from typing import List, Dict, Tuple
import re
from collections import defaultdict

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: transformers not available, using character-based token estimation")

print("🚀 Starting AI4Privacy PII detection and evaluation...")

MODEL_NAME = "ai4privacy"
MODEL_METADATA = {
    "provider": "AI4Privacy",
    "notes": "Optimized for numeric & temporal entities",
}

# Map AI4Privacy's granular labels to the coarse ground-truth schema.
AI4PRIVACY_TO_GT_LABEL = {
    # Person-like entities
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "TITLE": "PERSON",
    "SEX": "PERSON",
    # Location-like entities
    "CITY": "LOCATION",
    "COUNTRY": "LOCATION",
    "STATE": "LOCATION",
    "PROVINCE": "LOCATION",
    "STREET": "LOCATION",
    "BUILDINGNUM": "LOCATION",
    "ZIPCODE": "LOCATION",
    # Temporal entities
    "DATE": "DATE_TIME",
    "DATEOFBIRTH": "DATE_TIME",
    "TIME": "DATE_TIME",
    # Contact entities
    "EMAIL": "EMAIL_ADDRESS",
    "TELEPHONENUM": "PHONE_NUMBER",
}

# ---- 1. AI4Privacy Chunked Processor ----

class AI4PrivacyChunkedProcessor:
    """
    Chunked processor for AI4Privacy to handle long documents.
    AI4Privacy has a 1536 token limit, so we split long texts into chunks.
    """
    
    def __init__(self):
        """Initialize the chunked processor."""
        self.max_tokens = 1536
        self.chunk_size_tokens = 1200  # Safe margin
        self.overlap_tokens = 150      # Overlap to catch boundary entities
        
        # Load tokenizer for accurate token counting
        self.tokenizer = None
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"
                )
            except Exception:
                pass
    
    def count_tokens(self, text: str) -> int:
        """Count actual tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except:
                pass
        # Fallback estimation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def split_into_sentences(self, text: str) -> List[Tuple[str, int]]:
        """Split text into sentences with their positions."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = []
        
        current_pos = 0
        for match in re.finditer(sentence_pattern, text):
            sentence = text[current_pos:match.start()].strip()
            if sentence:
                sentences.append((sentence, current_pos))
            current_pos = match.end()
        
        # Add final sentence
        if current_pos < len(text):
            final_sentence = text[current_pos:].strip()
            if final_sentence:
                sentences.append((final_sentence, current_pos))
        
        return sentences
    
    def create_chunks(self, text: str) -> List[Tuple[str, int]]:
        """
        Create chunks that respect token limits and sentence boundaries.
        
        Returns:
            List of (chunk_text, start_pos) tuples
        """
        total_tokens = self.count_tokens(text)
        
        if total_tokens <= self.chunk_size_tokens:
            return [(text, 0)]
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_start_pos = 0
        
        for i, (sentence, sentence_pos) in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_chunk_tokens + sentence_tokens > self.chunk_size_tokens and current_chunk_sentences:
                # Save current chunk
                chunk_text = ' '.join([s[0] for s in current_chunk_sentences])
                chunks.append((chunk_text, chunk_start_pos))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                # Add sentences from the end of current chunk for overlap
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    sent_text, sent_pos = current_chunk_sentences[j]
                    sent_tokens = self.count_tokens(sent_text)
                    
                    if overlap_tokens + sent_tokens <= self.overlap_tokens:
                        overlap_sentences.insert(0, (sent_text, sent_pos))
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                # Start new chunk
                current_chunk_sentences = overlap_sentences + [(sentence, sentence_pos)]
                current_chunk_tokens = overlap_tokens + sentence_tokens
                chunk_start_pos = overlap_sentences[0][1] if overlap_sentences else sentence_pos
                
            else:
                # Add sentence to current chunk
                if not current_chunk_sentences:
                    chunk_start_pos = sentence_pos
                current_chunk_sentences.append((sentence, sentence_pos))
                current_chunk_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join([s[0] for s in current_chunk_sentences])
            chunks.append((chunk_text, chunk_start_pos))
        
        return chunks
    
    def process_chunk(self, chunk_text: str, chunk_start: int) -> List[Dict]:
        """Process a single chunk with AI4Privacy."""
        try:
            result = protect(chunk_text, classify_pii=True, verbose=True)
            
            entities = []
            if isinstance(result, dict) and 'replacements' in result:
                for replacement in result['replacements']:
                    entities.append({
                        'label': replacement['label'],
                        'text': replacement['value'],
                        'start': replacement['start'] + chunk_start,
                        'end': replacement['end'] + chunk_start,
                        'confidence': replacement.get('confidence', 0.8)  # Use confidence if available
                    })
            
            return entities
            
        except Exception as e:
            print(f"Error processing chunk at position {chunk_start}: {e}")
            return []
    
    def remove_duplicates(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities from overlapping chunks."""
        entities.sort(key=lambda x: (x['start'], x['end']))
        
        unique_entities = []
        
        for entity in entities:
            # Check if this entity overlaps with any existing one
            is_duplicate = False
            
            for existing in unique_entities:
                # Check for position overlap and same text/label
                if (entity['text'] == existing['text'] and 
                    entity['label'] == existing['label'] and
                    abs(entity['start'] - existing['start']) <= 5):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities
    
    def process_text(self, text: str) -> List[Dict]:
        """Process text using chunking if necessary."""
        total_tokens = self.count_tokens(text)
        
        # Use chunking for texts longer than our safe limit
        if total_tokens > self.chunk_size_tokens:
            chunks = self.create_chunks(text)
            
            all_entities = []
            for chunk_text, chunk_start in chunks:
                chunk_entities = self.process_chunk(chunk_text, chunk_start)
                all_entities.extend(chunk_entities)
            
            # Remove duplicates from overlapping chunks
            return self.remove_duplicates(all_entities)
        else:
            # Process normally for short texts
            return self.process_chunk(text, 0)

# Global processor instance to avoid reloading tokenizer/model
_processor_instance = None

def get_processor():
    """Get a global processor instance to avoid reloading tokenizer."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AI4PrivacyChunkedProcessor()
    return _processor_instance

def normalize_entity(entity_type: str):
    """Normalize AI4Privacy labels to match the ground-truth schema."""
    return AI4PRIVACY_TO_GT_LABEL.get(entity_type)

def ai4privacy_detect_pii(text: str):
    """Process text with AI4Privacy and return detected entities."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        processor = get_processor()
        detected_entities = processor.process_text(text)
        normalized_entities = []
        for entity in detected_entities:
            mapped_label = normalize_entity(entity["label"])
            if mapped_label is None:
                continue  # Skip entity types we don't have ground-truth for
            entity["label"] = mapped_label
            normalized_entities.append(entity)
        
        return normalized_entities
    except Exception as e:
        print(f"Error processing text with AI4Privacy: {e}")
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
                # Detect entities with AI4Privacy
                detected_pii = ai4privacy_detect_pii(text)
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
        "model_used": "ai4privacy",
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
    output_json_path = "output_ai4privacy_evaluation.json"

    process_dataset(input_json_path, output_json_path)
    print("\n✅ AI4Privacy detection and evaluation complete!")


def detect_pii(text: str, language: str = "en"):
    """
    Entry point used by the backend server. AI4Privacy currently handles
    English best; the `language` argument is reserved for future support.
    """
    _ = language
    return ai4privacy_detect_pii(text)

