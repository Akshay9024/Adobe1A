#!/usr/bin/env python3
"""
Test ML Integration Status
"""
import sys
import os
sys.path.append('src')

def test_model_loading():
    """Test if the trained model can be loaded"""
    try:
        from src.hybrid_classifier import LightweightMLClassifier
        
        model_path = "models/heading_classifier.joblib"
        classifier = LightweightMLClassifier(model_path)
        
        if classifier.model is not None and classifier.scaler is not None:
            print("✅ ML model loaded successfully")
            return True
        else:
            print("❌ Model failed to load properly")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_hybrid_classifier():
    """Test hybrid classifier initialization"""
    try:
        from src.hybrid_classifier import HybridHeadingClassifier
        
        model_path = "models/heading_classifier.joblib"
        classifier = HybridHeadingClassifier(ml_model_path=model_path)
        
        print("✅ Hybrid classifier initialized successfully")
        print(f"   Rule weight: {classifier.rule_weight}")
        print(f"   ML weight: {classifier.ml_weight}")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid classifier failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction"""
    try:
        from src.hybrid_classifier import FeatureExtractor, HeadingFeatures
        
        extractor = FeatureExtractor()
        
        # Create a mock candidate
        class MockBlock:
            def __init__(self):
                self.y0 = 100
                
        class MockCandidate:
            def __init__(self):
                self.text = "1. Introduction"
                self.size_ratio = 1.2
                self.is_bold = True
                self.is_italic = False
                self.has_distinct_font = True
                self.is_uppercase = False
                self.is_titlecase = True
                self.has_numbering = True
                self.numbering_depth = 1
                self.vertical_gap_before = 15.0
                self.block = MockBlock()
        
        candidate = MockCandidate()
        features = extractor.extract_features(candidate)
        feature_array = features.to_array()
        
        print("✅ Feature extraction working")
        print(f"   Feature vector length: {len(feature_array)}")
        print(f"   Sample features: font_size_ratio={features.font_size_ratio}, word_count={features.word_count}")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing ML Integration Status...")
    print()
    
    results = []
    results.append(test_model_loading())
    results.append(test_hybrid_classifier())
    results.append(test_feature_extraction())
    
    print()
    if all(results):
        print("🎉 All ML integration tests passed!")
        print("The trained model is properly integrated and ready for use.")
    else:
        print("⚠️  Some ML integration issues detected.")
        
    print()
    print("Model file details:")
    if os.path.exists("models/heading_classifier.joblib"):
        size = os.path.getsize("models/heading_classifier.joblib")
        print(f"   Size: {size/1024:.1f} KB")
        print(f"   Path: models/heading_classifier.joblib")
    else:
        print("   Model file not found!")