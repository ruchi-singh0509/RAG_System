# 🚀 Deployment Guide for Streamlit Cloud

## Overview
This guide will help you deploy your FAISS-based RAG system on Streamlit Cloud.

## Prerequisites
- GitHub repository with your code
- OpenAI API key
- Streamlit Cloud account

## Step-by-Step Deployment

### 1. Prepare Your Repository
Ensure your repository has:
- ✅ `app.py` (main Streamlit application)
- ✅ `requirements.txt` (all dependencies)
- ✅ `.streamlit/config.toml` (Streamlit configuration)
- ✅ `.gitignore` (excludes sensitive files)

### 2. Deploy on Streamlit Cloud

#### A. Go to Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account

#### B. Create New App
1. Click "New app"
2. Select your repository
3. Set main file path: `app.py`
4. Click "Deploy!"

### 3. Configure Secrets (CRITICAL!)

#### A. Access App Settings
1. In your deployed app, click the menu (☰)
2. Select "Settings"

#### B. Add Secrets
In the "Secrets" section, add:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
FAISS_PERSIST_DIRECTORY = "./faiss_db"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = "1000"
CHUNK_OVERLAP = "200"
MAX_FILE_SIZE = "50MB"
TESSERACT_CMD = "tesseract"
OCR_LANGUAGE = "eng"
TEMP_DIR = "./temp"
UPLOAD_DIR = "./uploads"
```

#### C. Save and Redeploy
1. Click "Save"
2. Your app will automatically redeploy

### 4. Verify Deployment

#### A. Check App Status
- App should show "Running" status
- No more "OpenAI API key not provided" errors

#### B. Test Basic Functionality
1. Upload a test document
2. Try asking a question
3. Verify RAG pipeline works

## Troubleshooting

### Common Issues

#### 1. "OpenAI API key not provided"
**Solution**: Check Streamlit secrets configuration
- Ensure `OPENAI_API_KEY` is set correctly
- No extra spaces or quotes around the key
- Redeploy after saving secrets

#### 2. "Module not found" errors
**Solution**: Check `requirements.txt`
- Ensure all dependencies are listed
- Use exact versions if needed
- Redeploy after updating requirements

#### 3. App crashes on startup
**Solution**: Check logs
- Look at Streamlit Cloud logs
- Verify all imports work
- Check for syntax errors

### Debug Steps
1. **Check App Logs**: Look at the logs in Streamlit Cloud
2. **Verify Secrets**: Ensure all environment variables are set
3. **Test Locally**: Run `streamlit run app.py` locally first
4. **Check Dependencies**: Verify all packages in requirements.txt

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | ✅ Yes |
| `FAISS_PERSIST_DIRECTORY` | FAISS database directory | `./faiss_db` | ❌ No |
| `MODEL_NAME` | Sentence transformer model | `all-MiniLM-L6-v2` | ❌ No |
| `CHUNK_SIZE` | Text chunk size | `1000` | ❌ No |
| `CHUNK_OVERLAP` | Chunk overlap | `200` | ❌ No |

## Security Notes

### ✅ What's Safe to Commit
- Source code
- Configuration files
- Documentation
- Requirements

### ❌ What's NOT Safe to Commit
- `.env` files
- API keys
- Database files
- Personal data

### 🔐 How Secrets Work
- Streamlit Cloud stores secrets securely
- Secrets are encrypted at rest
- Only accessible to your app
- Never exposed in logs or code

## Performance Considerations

### Streamlit Cloud Limits
- **Free Tier**: 1GB RAM, 1GB storage
- **Pro Tier**: 4GB RAM, 4GB storage

### Optimization Tips
1. **Use smaller models** for free tier
2. **Limit file uploads** to reasonable sizes
3. **Implement caching** for expensive operations
4. **Use async operations** where possible

## Next Steps After Deployment

1. **Monitor Performance**: Watch app metrics
2. **Test with Real Data**: Upload actual documents
3. **Optimize**: Adjust chunk sizes and models
4. **Scale**: Consider Pro tier for production use

## Support

If you encounter issues:
1. Check this deployment guide
2. Review Streamlit Cloud documentation
3. Check GitHub issues for similar problems
4. Contact Streamlit support for platform issues

---

**Happy Deploying! 🎉**
