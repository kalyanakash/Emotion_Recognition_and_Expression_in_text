# 🐙 GitHub Setup Guide

## Step 1: Create GitHub Repository

1. **Go to GitHub**: [github.com](https://github.com)
2. **Click "New Repository"** (green button)
3. **Repository Settings**:
   - **Name**: `AI-Emotion-Detector` (or keep current name)
   - **Description**: `🎭 Modern AI-powered emotion detection from text using deep learning and Streamlit`
   - **Visibility**: Public (recommended for free deployment)
   - **Initialize**: ❌ Don't initialize (we already have files)

## Step 2: Connect Local Repository to GitHub

```bash
# Add GitHub remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/AI-Emotion-Detector.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Setup Git LFS (for Large Files)

If your model files are > 100MB, you'll need Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git lfs track "*.weights.h5"

# Add and commit LFS tracking
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
git push
```

## Step 4: Deploy to Streamlit Cloud

1. **Go to**: [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Repository**: Select your repository
5. **Branch**: main
6. **Main file path**: app.py
7. **Click "Deploy!"**

## Quick Commands Summary

```bash
# Connect to GitHub (replace USERNAME and REPO-NAME)
git remote add origin https://github.com/USERNAME/REPO-NAME.git
git branch -M main
git push -u origin main
```

🎉 **Your project is now ready for deployment!**
