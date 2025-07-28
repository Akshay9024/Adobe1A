import numpy as np
import pickle
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

@dataclass
class HeadingFeatures:
    """Feature vector for ML classification"""
    # Font features
    font_size_ratio: float  # Ratio to body text size
    is_bold: int  # Binary
    is_italic: int  # Binary
    has_distinct_font: int  # Binary
    
    # Text features
    word_count: int
    char_count: int
    is_uppercase: int  # Binary
    is_titlecase: int  # Binary
    starts_with_capital: int  # Binary
    
    # Numbering features
    has_numbering: int  # Binary
    numbering_depth: int  # 0-3
    has_decimal_number: int  # Binary
    has_letter_number: int  # Binary
    
    # Position features
    relative_y_position: float  # 0-1 (top to bottom of page)
    vertical_gap_before: float  # Normalized
    is_first_on_page: int  # Binary
    
    # Statistical features
    avg_word_length: float
    punctuation_ratio: float
    digit_ratio: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model"""
        return np.array([
            self.font_size_ratio,
            self.is_bold,
            self.is_italic,
            self.has_distinct_font,
            self.word_count,
            self.char_count,
            self.is_uppercase,
            self.is_titlecase,
            self.starts_with_capital,
            self.has_numbering,
            self.numbering_depth,
            self.has_decimal_number,
            self.has_letter_number,
            self.relative_y_position,
            self.vertical_gap_before,
            self.is_first_on_page,
            self.avg_word_length,
            self.punctuation_ratio,
            self.digit_ratio
        ])
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get feature names for interpretability"""
        return [
            'font_size_ratio', 'is_bold', 'is_italic', 'has_distinct_font',
            'word_count', 'char_count', 'is_uppercase', 'is_titlecase',
            'starts_with_capital', 'has_numbering', 'numbering_depth',
            'has_decimal_number', 'has_letter_number', 'relative_y_position',
            'vertical_gap_before', 'is_first_on_page', 'avg_word_length',
            'punctuation_ratio', 'digit_ratio'
        ]


class FeatureExtractor:
    """Extracts ML features from heading candidates"""
    
    def __init__(self, page_height: float = 792.0):
        self.page_height = page_height
        
    def extract_features(self, candidate: 'HeadingCandidate', 
                        is_first_on_page: bool = False) -> HeadingFeatures:
        """Extract feature vector from candidate"""
        text = candidate.text
        
        # Text statistics
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Character ratios
        punct_count = sum(1 for c in text if c in '.,;:!?-()[]{}')
        digit_count = sum(1 for c in text if c.isdigit())
        punctuation_ratio = punct_count / max(char_count, 1)
        digit_ratio = digit_count / max(char_count, 1)
        
        # Numbering detection
        has_decimal = bool(re.match(r'^\d+\.?\s', text))
        has_letter = bool(re.match(r'^[A-Z]\.?\s', text, re.IGNORECASE))
        
        return HeadingFeatures(
            font_size_ratio=candidate.size_ratio,
            is_bold=int(candidate.is_bold),
            is_italic=int(candidate.is_italic),
            has_distinct_font=int(candidate.has_distinct_font),
            word_count=word_count,
            char_count=char_count,
            is_uppercase=int(candidate.is_uppercase),
            is_titlecase=int(candidate.is_titlecase),
            starts_with_capital=int(text and text[0].isupper()),
            has_numbering=int(candidate.has_numbering),
            numbering_depth=candidate.numbering_depth,
            has_decimal_number=int(has_decimal),
            has_letter_number=int(has_letter),
            relative_y_position=min(candidate.block.y0 / self.page_height, 1.0),
            vertical_gap_before=min(candidate.vertical_gap_before / 20.0, 2.0),  # Normalize
            is_first_on_page=int(is_first_on_page),
            avg_word_length=avg_word_length,
            punctuation_ratio=punctuation_ratio,
            digit_ratio=digit_ratio
        )


class LightweightMLClassifier:
    """Lightweight ML model for heading detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor()
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with a lightweight model
            self.model = RandomForestClassifier(
                n_estimators=50,  # Small forest
                max_depth=10,     # Limit depth
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.scaler = StandardScaler()
            
    def train(self, training_data: List[Tuple['HeadingCandidate', bool, bool]]):
        """Train the model on labeled data
        
        Args:
            training_data: List of (candidate, is_heading, is_first_on_page) tuples
        """
        if not training_data:
            logging.warning("No training data provided")
            return
            
        # Extract features and labels
        X = []
        y = []
        
        for candidate, is_heading, is_first in training_data:
            features = self.feature_extractor.extract_features(candidate, is_first)
            X.append(features.to_array())
            y.append(int(is_heading))
            
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Log performance
        scores = cross_val_score(self.model, X_scaled, y, cv=3)
        logging.info(f"ML model cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Log feature importance
        self._log_feature_importance()
        
    def predict_proba(self, candidate: 'HeadingCandidate', 
                     is_first_on_page: bool = False) -> float:
        """Get probability that candidate is a heading"""
        if self.model is None:
            return 0.5  # No model, neutral probability
            
        features = self.feature_extractor.extract_features(candidate, is_first_on_page)
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get probability of being a heading (class 1)
        proba = self.model.predict_proba(X_scaled)[0, 1]
        return proba
        
    def save_model(self, path: str):
        """Save model and scaler to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': HeadingFeatures.get_feature_names()
        }
        
        # Use joblib for scikit-learn models (more efficient than pickle)
        joblib.dump(model_data, path)
        
        # Check file size
        import os
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logging.info(f"Model saved to {path} (size: {size_mb:.2f} MB)")
        
    def load_model(self, path: str):
        """Load model and scaler from disk"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            logging.info(f"Model loaded from {path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            
    def _log_feature_importance(self):
        """Log feature importance for interpretability"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = HeadingFeatures.get_feature_names()
            
            # Sort by importance
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            logging.info("Top 10 most important features:")
            for i, idx in enumerate(indices):
                logging.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")


class HybridHeadingClassifier:
    """Combines rule-based and ML approaches"""
    
    def __init__(self, 
                 rule_weight: float = 0.7,
                 ml_weight: float = 0.3,
                 ml_model_path: Optional[str] = None):
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight
        self.ml_classifier = LightweightMLClassifier(ml_model_path)
        
        # Thresholds
        self.rule_threshold = 40.0  # Confidence score threshold
        self.ml_threshold = 0.7     # Probability threshold
        self.combined_threshold = 0.6
        
    def classify(self, candidates: List['HeadingCandidate']) -> List['HeadingCandidate']:
        """Classify candidates using hybrid approach"""
        if not candidates:
            return []
            
        # Track first candidates on each page
        first_on_page = set()
        for page_num in set(c.page_num for c in candidates):
            page_candidates = [c for c in candidates if c.page_num == page_num]
            if page_candidates:
                first = min(page_candidates, key=lambda c: c.block.y0)
                first_on_page.add(id(first))
                
        # Classify each candidate
        final_candidates = []
        
        for candidate in candidates:
            is_first = id(candidate) in first_on_page
            
            # Get rule-based score (normalized to 0-1)
            rule_score = min(candidate.confidence_score / 100.0, 1.0)
            
            # Get ML probability
            ml_score = self.ml_classifier.predict_proba(candidate, is_first)
            
            # Combine scores
            combined_score = (self.rule_weight * rule_score + 
                            self.ml_weight * ml_score)
            
            # Decision logic
            is_heading = False
            decision_reason = ""
            
            # Strong rule-based signal
            if rule_score >= self.rule_threshold / 100.0:
                is_heading = True
                decision_reason = "strong_rule"
                
            # Strong ML signal
            elif ml_score >= self.ml_threshold:
                is_heading = True
                decision_reason = "strong_ml"
                
            # Combined signal
            elif combined_score >= self.combined_threshold:
                is_heading = True
                decision_reason = "combined"
                
            # Special case: numbered headings always included
            elif candidate.has_numbering:
                is_heading = True
                decision_reason = "numbering"
                
            if is_heading:
                # Store scores for debugging
                candidate.rule_score = rule_score
                candidate.ml_score = ml_score
                candidate.combined_score = combined_score
                candidate.decision_reason = decision_reason
                
                final_candidates.append(candidate)
                
        # Log statistics
        self._log_classification_stats(candidates, final_candidates)
        
        return final_candidates
        
    def _log_classification_stats(self, all_candidates: List, 
                                 selected: List):
        """Log classification statistics"""
        total = len(all_candidates)
        selected_count = len(selected)
        
        if selected:
            reasons = {}
            for c in selected:
                reason = getattr(c, 'decision_reason', 'unknown')
                reasons[reason] = reasons.get(reason, 0) + 1
                
            logging.info(f"Hybrid classifier: {selected_count}/{total} selected")
            logging.info(f"Decision breakdown: {reasons}")
            
    def train_ml_component(self, training_data: List[Tuple['HeadingCandidate', bool, bool]]):
        """Train the ML component"""
        self.ml_classifier.train(training_data)
        
    def save_ml_model(self, path: str):
        """Save ML model"""
        self.ml_classifier.save_model(path)


# Training data generator (for bootstrap training)
def generate_synthetic_training_data(candidates: List['HeadingCandidate']) -> List[Tuple]:
    """Generate synthetic training data from high-confidence candidates"""
    training_data = []
    
    # Positive examples: high-confidence headings
    for c in candidates:
        if c.confidence_score > 60:  # High confidence
            training_data.append((c, True, False))  # Simplified, ignoring page position
            
    # Negative examples: create from low-confidence candidates
    for c in candidates:
        if c.confidence_score < 20:  # Low confidence
            training_data.append((c, False, False))
            
    logging.info(f"Generated {len(training_data)} training examples")
    return training_data


# Integration function
def create_hybrid_classifier(candidates: List['HeadingCandidate'], 
                           model_path: Optional[str] = None) -> HybridHeadingClassifier:
    """Create and optionally train a hybrid classifier"""
    classifier = HybridHeadingClassifier(ml_model_path=model_path)
    
    # If no pre-trained model, bootstrap with synthetic data
    if model_path is None and candidates:
        training_data = generate_synthetic_training_data(candidates)
        if training_data:
            classifier.train_ml_component(training_data)
            
    return classifier