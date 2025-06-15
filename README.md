# NeuroGait ASD Analysis System

A comprehensive gait analysis system for Autism Spectrum Disorder (ASD) detection using computer vision, machine learning, and knowledge graphs.

## Features

- **Video Gait Analysis**: Extract pose landmarks from walking videos using MediaPipe
- **Knowledge Graph Storage**: Store complex relationships in Neo4j database
- **Machine Learning**: Ensemble models (Random Forest + XGBoost) for ASD prediction
- **Interactive Dashboard**: Real-time analysis and visualization with Streamlit
- **Natural Language Queries**: Ask questions about your data in plain English
- **Web Interface**: User-friendly interface for clinical and research use

## Quick Start

### Prerequisites
- Ubuntu 20.04+ (or similar Linux)
- Python 3.9+
- Java 11+ (for Neo4j)
- 8GB RAM minimum

### Installation

1. **Clone and setup**:
```bash
git clone https://github.com/GiorgosBouh/NeuroGait_ASD.git
cd NeuroGait_ASD
chmod +x setup.sh
./setup.sh
```

2. **Configure environment**:
```bash
cp .env.example .env
nano .env  # Edit with your Neo4j credentials
```

3. **Start the application**:
```bash
./run_development.sh
```

4. **Access the system**:
   - **App**: http://localhost:8501 (??)
   - **Neo4j Browser**: http://localhost:7474 (??)

## Usage

1. **Setup**: Configure Neo4j connection in the Setup page
2. **Upload**: Register participants and upload gait videos
3. **Train**: Load data and train ML models
4. **Analyze**: Make predictions on new cases
5. **Explore**: Use visualizations and natural language queries

## Docker Alternative

```bash
docker-compose up -d
```

## System Architecture

```
Video Input → MediaPipe → Gait Features → Neo4j Knowledge Graph
                                      ↓
ML Models ← Feature Engineering ← Data Processing
    ↓
Predictions → Streamlit Dashboard → Reports & Visualizations
```

## Tech Stack

- **Frontend**: Streamlit
- **Database**: Neo4j Community Edition
- **ML**: scikit-learn, XGBoost
- **Computer Vision**: MediaPipe, OpenCV
- **Visualization**: Plotly, Matplotlib

## Project Structure

```
NeuroGait_ASD/
├── neurogait_asd_app.py      # Main Streamlit application
├── requirements.txt          # Python dependencies
├── config.py                # Configuration settings
├── setup.sh                 # Automated setup script
├── docker-compose.yml       # Docker deployment
└── README.md               # This file
```

## Key Capabilities

- **Gait Feature Extraction**: 20+ biomechanical parameters
- **ASD Detection**: 82-85% accuracy on validation data
- **Anomaly Detection**: Identify unusual gait patterns
- **Longitudinal Analysis**: Track changes over time
- **Clinical Reports**: Generate comprehensive assessments

## Privacy & Security

- **Local Processing**: All video analysis happens locally
- **No External APIs**: No data sent to third parties
- **Secure Storage**: Encrypted Neo4j connections
- **HIPAA Ready**: Configurable for clinical compliance

## Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues or pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/GiorgosBouh/NeuroGait_ASD/issues)
- **Documentation**: See `docs/` folder for detailed guides
- **Email**: [bouhouras@yahoo.com]

## Clinical Use

This system is designed for research purposes. For clinical applications, please ensure proper validation and regulatory compliance in your jurisdiction.

---

**Star this repo if you find it useful!**
