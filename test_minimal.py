#!/usr/bin/env python3
"""
Minimal test script to verify PyTorch and sentence-transformers work
"""

import streamlit as st
import sys

def test_minimal_imports():
    """Test minimal imports"""
    st.write("🔍 Testing minimal imports...")
    
    try:
        import torch
        st.success(f"✅ PyTorch {torch.__version__} imported")
        
        import sentence_transformers
        st.success(f"✅ Sentence Transformers {sentence_transformers.__version__} imported")
        
        return True
    except Exception as e:
        st.error(f"❌ Import failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    st.write("🗄️ Testing model loading...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try loading a small model
        model_name = "paraphrase-MiniLM-L3-v2"
        st.info(f"Loading model: {model_name}")
        
        model = SentenceTransformer(model_name, device='cpu')
        st.success("✅ Model loaded successfully")
        
        # Test encoding
        test_text = "Hello world"
        embedding = model.encode(test_text)
        st.success(f"✅ Encoding successful, dimension: {len(embedding)}")
        
        return True
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return False

def main():
    """Main test function"""
    st.title("🧪 Minimal PyTorch Test")
    
    st.write("This script tests if the basic PyTorch and sentence-transformers functionality works.")
    
    # Test 1: Imports
    if test_minimal_imports():
        st.success("✅ All imports successful")
    else:
        st.error("❌ Import test failed")
        return
    
    # Test 2: Model loading
    if test_model_loading():
        st.success("✅ Model loading successful")
        st.balloons()
    else:
        st.error("❌ Model loading test failed")

if __name__ == "__main__":
    main()
