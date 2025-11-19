#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from presidio_analyzer import AnalyzerEngine

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # CORS tam çözüldü

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Presidio başlat
try:
    analyzer = AnalyzerEngine()
    logger.info("Presidio başarıyla yüklendi")
except Exception as e:
    logger.error(f"Presidio hatası: {e}")
    analyzer = None

@app.route('/')
def index():
    return jsonify({"service": "PII Detection API", "status": "running"})

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "presidio": analyzer is not None,
        "models": ["presidio"]
    })

@app.route('/detect-pii', methods=['POST'])
def detect_pii():
    if not analyzer:
        return jsonify({"error": "Presidio yüklü değil"}), 500

    data = request.get_json()
    text = data.get('text', '')
    language = data.get('language', 'en')

    if not text.strip():
        return jsonify({"detected_entities": [], "total_entities": 0})

    try:
        results = analyzer.analyze(text=text, language=language, entities=[
            "PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "LOCATION", 
            "CREDIT_CARD", "IBAN_CODE", "TR_IDENTITY_NUMBER"
        ])
        
        entities = []
        for r in results:
            entities.append({
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "text": text[r.start:r.end],
                "score": round(r.score, 3)
            })

        return jsonify({
            "model_used": "presidio",
            "total_entities": len(entities),
            "detected_entities": entities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
