# 🚀 Deployment Guide

## Quick Deploy Options

### 1. Streamlit Community Cloud (Recommended - FREE)

**Steps:**
1. Push your code to GitHub (see GitHub setup below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Select the main branch and `app.py` as the main file
6. Click "Deploy!"

**Pros:** Free, automatic SSL, easy updates via GitHub
**Cons:** Limited resources for heavy usage

### 2. Heroku (FREE Tier Available)

**Setup:**
```bash
# Install Heroku CLI
# Login
heroku login

# Create app
heroku create your-emotion-detector-app

# Deploy
git push heroku main
```

**Configuration:** 
- `Procfile` ✅ (included)
- `setup.sh` ✅ (included)
- `requirements.txt` ✅ (included)

### 3. Google Cloud Platform

**Setup:**
```bash
# Install Google Cloud SDK
gcloud auth login
gcloud config set project your-project-id

# Deploy
gcloud app deploy
```

**Configuration:**
- `app.yaml` ✅ (included)

### 4. Docker Deployment

**Local Docker:**
```bash
# Build
docker build -t emotion-detector .

# Run
docker run -p 8501:8501 emotion-detector
```

**Docker Hub:**
```bash
# Tag and push
docker tag emotion-detector yourusername/emotion-detector
docker push yourusername/emotion-detector
```

### 5. Railway (Modern Alternative)

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically

### 6. Render (Another Good Option)

1. Go to [render.com](https://render.com)
2. Connect GitHub repository
3. Choose "Web Service"
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Environment Variables

Some platforms may require these environment variables:

```
PORT=8501
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

## Pre-deployment Checklist

- [ ] All files committed to Git
- [ ] Repository pushed to GitHub
- [ ] Model files included (check file sizes)
- [ ] Requirements.txt updated
- [ ] App tested locally
- [ ] Secrets/credentials removed
- [ ] README.md updated with live demo URL

## Troubleshooting

**Common Issues:**

1. **Large Model Files:** 
   - Use Git LFS for files > 100MB
   - Consider model compression
   - Host models externally if needed

2. **Memory Issues:**
   - Optimize model loading
   - Use model caching
   - Consider lighter model architectures

3. **Slow Loading:**
   - Implement model caching
   - Use async loading
   - Add loading indicators

## GitHub Setup Instructions

See `GITHUB_SETUP.md` for detailed GitHub configuration steps.

## Local Development

```bash
# Clone and setup
git clone your-repo-url
cd emotion-detection-app
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Performance Optimization

1. **Caching:** Model loading is cached in Streamlit
2. **Memory:** Monitor memory usage in production
3. **Speed:** Consider model quantization for faster inference

## Security Considerations

- No sensitive data in repository
- Environment variables for secrets
- HTTPS enforced in production
- Input validation implemented

---

Choose the deployment option that best fits your needs and budget!
