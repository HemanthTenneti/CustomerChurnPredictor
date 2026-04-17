#!/bin/bash
set -e

echo "======================================"
echo "VERIFICATION TEST SUITE"
echo "======================================"

# Check 1: Build vector store
echo ""
echo "[1/5] Building vector store with rag/ingest.py..."
cd /Users/hemanth10etii/Coding/CustomerChurnPredictor
python rag/ingest.py
if [ $? -eq 0 ]; then
    echo "✅ Vector store built successfully"
else
    echo "❌ Vector store build FAILED"
    exit 1
fi

# Check 2: Import app module
echo ""
echo "[2/5] Checking app.py imports..."
python -c "import app; print('✅ app.py imports successfully')" 2>&1
if [ $? -eq 0 ]; then
    echo "✅ app.py imports OK"
else
    echo "❌ app.py import FAILED"
    exit 1
fi

# Check 3: Test agent (requires GROQ_API_KEY)
echo ""
echo "[3/5] Testing agent with sample profile..."
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  GROQ_API_KEY not set — skipping agent test"
else
    python agent/churn_agent.py
    if [ $? -eq 0 ]; then
        echo "✅ Agent test passed"
    else
        echo "❌ Agent test FAILED"
        exit 1
    fi
fi

# Check 4: Verify LaTeX
echo ""
echo "[4/5] Checking LaTeX compilation..."
cd /Users/hemanth10etii/Coding/CustomerChurnPredictor/Report/End_Sem_Report
if command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode main.tex > /tmp/latex.log 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ LaTeX compiles successfully"
    else
        echo "❌ LaTeX compilation FAILED"
        tail -20 /tmp/latex.log
        exit 1
    fi
else
    echo "⚠️  pdflatex not installed — skipping"
fi

# Check 5: Verify .gitignore
echo ""
echo "[5/5] Verifying .gitignore..."
cd /Users/hemanth10etii/Coding/CustomerChurnPredictor
if grep -q "\.env" .gitignore && grep -q "__pycache__" .gitignore && grep -q "rag/chroma_db" .gitignore; then
    echo "✅ .gitignore looks good"
else
    echo "❌ .gitignore missing required entries"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ ALL VERIFICATION TESTS PASSED"
echo "======================================"
