# Customer Churn Predictor - Deployment Guide

## Platform-Specific Deployment Instructions

### 1. Streamlit Community Cloud

**Recommended for this project**

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Set main file path: `ui/streamlit_app.py`
5. Deploy automatically

**Environment Variables:**

- No additional setup needed for Milestone 1
- For Milestone 2: Add `OPENAI_API_KEY` in Streamlit secrets

### 2. Hugging Face Spaces

**Alternative deployment option**

1. Create new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Streamlit" as SDK
3. Upload project files
4. Ensure `app.py` exists in root (already created)
5. Space will auto-deploy

**File Requirements:**

- `app.py` in root directory ✅
- `requirements.txt` with dependencies ✅

### 3. Render (Free Tier)

**Backup deployment option**

1. Connect GitHub repository to Render
2. Create new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run ui/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
5. Deploy

**Important:** Render requires PORT environment variable handling

## Memory and Resource Requirements

### Milestone 1 Requirements:

- **RAM**: ~200MB (ML models + Streamlit)
- **Storage**: ~50MB (code + data)
- **CPU**: Minimal (inference only)

### Milestone 2 Additional Requirements:

- **RAM**: +300MB (for local LLM if used)
- **API Calls**: OpenAI free tier limits
- **Storage**: +100MB (vector stores)

## Platform Compatibility Checklist

✅ **Relative file paths used throughout project**  
✅ **Environment variables configured**  
✅ **No localhost-specific code**  
✅ **Requirements.txt with version constraints**  
✅ **Entry points for all platforms created**

## Testing Before Deployment

1. Test locally with production settings:

   ```bash
   streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
   ```

2. Test environment variable loading:

   ```bash
   python -c "import config; print('Config loaded successfully')"
   ```

3. Validate all file paths resolve correctly in hosted environment
