# 🎭 AI Emotion Detector

A modern, interactive web application for real-time text emotion analysis powered by deep learning and built with Streamlit.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Real-time Emotion Detection**: Analyze text emotions instantly with high accuracy
- **Beautiful Modern UI**: Glass-morphism design with responsive layout
- **Interactive Visualizations**: Dynamic charts showing confidence scores and emotion trends
- **Emotion History**: Track and analyze your emotion detection history
- **Preprocessing Options**: Optional text preprocessing for better accuracy
- **Dashboard Analytics**: Statistics and insights about your usage
- **8 Emotion Categories**: Happy, Sad, Anger, Fear, Surprise, Love, Disgust, Neutral

## 🚀 Demo

Try the live demo: [AI Emotion Detector](your-deployed-url-here)

## 📷 Screenshots

### Main Interface
![Main Interface](screenshots/main-interface.png)

### Analytics Dashboard
![Analytics](screenshots/analytics-dashboard.png)

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Emotion-Recognition_and-Expression_in_Text.git
cd Emotion-Recognition_and-Expression_in_Text
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 🏗️ Project Structure

```
Emotion-Recognition_and-Expression_in_Text/
│
├── app.py                          # Main Streamlit application
├── model_architecture.json         # Trained model architecture
├── model_weights.h5                # Trained model weights
├── train.txt                       # Training dataset
├── test.txt                        # Test dataset
├── val.txt                         # Validation dataset
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment file
├── setup.sh                       # Deployment setup script
├── app.yaml                        # Google Cloud deployment
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

## 🤖 Model Information

- **Architecture**: Deep Neural Network with LSTM layers
- **Framework**: TensorFlow/Keras
- **Training Data**: Emotion-labeled text dataset
- **Accuracy**: ~85% on test set
- **Emotions Detected**: 8 categories (Happy, Sad, Anger, Fear, Surprise, Love, Disgust, Neutral)

## 🌐 Deployment Options

### 1. Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repository

### 2. Heroku
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create new app
heroku create your-app-name

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### 3. Google Cloud Platform
```bash
# Install Google Cloud SDK
# Authenticate
gcloud auth login

# Deploy
gcloud app deploy
```

### 4. Docker
```bash
# Build image
docker build -t emotion-detector .

# Run container
docker run -p 8501:8501 emotion-detector
```

## 🎯 Usage

1. **Enter Text**: Type or paste any text in the input area
2. **Choose Options**: Select preprocessing options if needed
3. **Detect Emotion**: Click the "Detect Emotion" button
4. **View Results**: See the detected emotion with confidence score
5. **Analyze Trends**: Check the analytics dashboard for insights
6. **Review History**: Browse through your detection history

## 📊 API Usage (Optional)

You can also use this as an API by modifying the code:

```python
import requests

response = requests.post('your-api-endpoint', 
                        json={'text': 'I am feeling great today!'})
emotion = response.json()['emotion']
confidence = response.json()['confidence']
```

## 🔧 Configuration

### Streamlit Configuration
Edit `.streamlit/config.toml` to customize:
- Theme colors
- Server settings
- Browser behavior

### Model Configuration
- Model files: `model_architecture.json` and `model_weights.h5`
- Training data: `train.txt`, `test.txt`, `val.txt`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit team for the amazing web app framework
- The open-source community for various libraries used

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🐛 Issues and Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/Emotion-Recognition_and-Expression_in_Text/issues) page
2. Create a new issue if your problem isn't already listed
3. Provide detailed information about the problem

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Voice emotion detection
- [ ] Batch text processing
- [ ] Custom model training interface
- [ ] REST API endpoints
- [ ] Mobile app version
- [ ] Real-time chat emotion analysis

---

⭐ If you found this project helpful, please give it a star on GitHub!


