# 🧠 NeuroGait ASD Analysis System

> Advanced Gait Analysis for Autism Spectrum Disorder Detection using Knowledge Graphs and Machine Learning

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Neo4j Setup](#neo4j-setup)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## 🚀 Quick Start

### Option 1: Smart Installer (Recommended)
```bash
# Clone or download the project
git clone https://github.com/yourusername/NeuroGait_ASD.git
cd NeuroGait_ASD

# Run smart installer
python install.py
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python -m venv neurogait_env
source neurogait_env/bin/activate  # Linux/Mac
# neurogait_env\Scripts\activate   # Windows

# Install requirements
pip install -r requirements-production.txt

# Run application
streamlit run neurogait_app.py
```

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 16GB or higher
- **GPU**: NVIDIA GPU with CUDA support (for ML acceleration)
- **Storage**: 10GB free space (for models and data)

### Dependencies Check
```bash
# Check your system
python install.py --check

# Get system information
python -c "import platform; print(f'Python: {platform.python_version()}, OS: {platform.system()}')"
```

## 📦 Installation Methods

### 1. Production Installation (Minimal)
For deployment and production use:
```bash
pip install -r requirements-production.txt
```

**Includes:**
- Core ML libraries (scikit-learn, XGBoost, SHAP)
- Streamlit web framework
- Neo4j database connector
- Essential visualization tools

### 2. Development Installation (Full)
For development and research:
```bash
pip install -r requirements-dev.txt
```

**Additional includes:**
- Testing frameworks (pytest, coverage)
- Code formatting (black, flake8)
- Jupyter notebooks
- Advanced ML tools
- Documentation tools

### 3. Custom Installation
Choose specific components:

```bash
# Only SHAP and basic ML
pip install streamlit pandas scikit-learn shap plotly

# Add computer vision
pip install opencv-python mediapipe

# Add graph database
pip install neo4j networkx

# Add file processing
pip install openpyxl pandas
```

## 🗄️ Neo4j Setup

### Quick Setup (Docker)
```bash
# Run Neo4j in Docker
docker run \
    --name neo4j-neurogait \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/neurogait123 \
    neo4j:latest
```

### Manual Installation (Ubuntu/Debian)
```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install
sudo apt update
sudo apt install neo4j

# Configure
sudo nano /etc/neo4j/neo4j.conf
# Uncomment: dbms.default_listen_address=0.0.0.0

# Start service
sudo systemctl enable neo4j
sudo systemctl start neo4j

# Set password
sudo neo4j-admin set-initial-password neurogait123
```

### Manual Installation (Windows)
1. Download Neo4j Community Edition from https://neo4j.com/download/
2. Extract and run `bin/neo4j.bat console`
3. Access http://localhost:7474
4. Login with neo4j/neo4j, set new password

### Manual Installation (macOS)
```bash
# Using Homebrew
brew install neo4j

# Start Neo4j
neo4j start

# Access web interface at http://localhost:7474
```

### Verify Neo4j Installation
```bash
# Test connection
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'your_password'))
with driver.session() as session:
    result = session.run('RETURN 1 as test')
    print('Neo4j connection successful:', result.single()['test'])
driver.close()
"
```

## 🎯 Usage

### 1. Start the Application
```bash
# Activate virtual environment
source neurogait_env/bin/activate

# Run Streamlit app
streamlit run neurogait_app.py
```

### 2. Access Web Interface
Open your browser and go to: http://localhost:8501

### 3. Configure Database Connection
1. Go to **Setup** page
2. Enter Neo4j connection details:
   - URI: `bolt://localhost:7687`
   - Username: `neo4j`
   - Password: `your_password`
3. Click **Connect to Neo4j**

### 4. Upload Data
Choose one of these methods:

#### Video Upload
1. Go to **Data Upload** page
2. Register participant information
3. Upload video file (MP4, AVI, MOV)
4. Process video for gait analysis

#### XLSX Batch Upload
1. Prepare Excel file with columns:
   ```
   participant_id, age, gender, diagnosis,
   step_length_mean, step_length_std, cadence,
   stride_width_mean, stride_width_std,
   left_shoulder_angle_mean, right_shoulder_angle_mean,
   step_asymmetry, left_grf_variance, right_grf_variance
   ```
2. Upload XLSX file
3. Process batch data

### 5. Train Models
1. Go to **Analysis** page
2. Click **Load Training Data from Knowledge Graph**
3. Click **Train Models**
4. Wait for training completion

### 6. Make Predictions
1. Upload new video or load existing data
2. Click **Make Prediction**
3. Click **Generate SHAP Explanations** for interpretability
4. View performance metrics with **Generate Complete Performance Report**

## 🔧 Troubleshooting

### Common Issues

#### Installation Issues
```bash
# Permission denied (Linux/Mac)
sudo pip install -r requirements-production.txt

# SSL certificate error
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-production.txt

# Memory error during installation
pip install --no-cache-dir -r requirements-production.txt
```

#### Neo4j Connection Issues
```bash
# Check if Neo4j is running
sudo systemctl status neo4j

# Check port availability
netstat -tulpn | grep :7687

# Test connection manually
python install.py --check
```

#### Import Errors
```bash
# Missing SHAP
pip install shap

# OpenCV issues on Linux
sudo apt-get install python3-opencv

# MediaPipe issues
pip uninstall mediapipe
pip install mediapipe

# Windows MediaPipe fix
pip install mediapipe-silicon  # for Apple Silicon
```

#### Memory Issues
```bash
# Reduce batch size in code
# Use minimal installation
pip install -r requirements-production.txt

# Clear cache
pip cache purge
```

#### SHAP Installation Issues
```bash
# If SHAP fails to install
pip install --no-use-pep517 shap

# Alternative: install dependencies first
pip install numpy scipy scikit-learn
pip install shap
```

### Performance Optimization

#### For Low-Memory Systems
```python
# In the code, modify batch sizes:
# Change BATCH_SIZE = 32 to BATCH_SIZE = 8
# Reduce model complexity
```

#### For GPU Acceleration
```bash
# Install CUDA-enabled packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 👨‍💻 Development

### Development Environment Setup
```bash
# Install development requirements
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=neurogait

# Format code
black neurogait_app.py
flake8 neurogait_app.py

# Generate documentation
sphinx-build -b html docs/ docs/_build/html/
```

### Testing
```bash
# Run all tests
python install.py --test

# Run specific tests
pytest tests/test_gait_analyzer.py -v

# Test with coverage
pytest --cov=neurogait --cov-report=html
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Add tests in `tests/` directory
3. Update documentation
4. Run tests and formatting: `black . && flake8 . && pytest`
5. Submit pull request

## 📊 Performance Benchmarks

### Model Performance
- **Accuracy**: 85-92% (depending on dataset)
- **Training Time**: 2-5 minutes (CPU), 30-60 seconds (GPU)
- **Inference Time**: <1 second per sample

### System Requirements vs Performance
| RAM | CPU Cores | Training Time | Batch Size |
|-----|-----------|---------------|------------|
| 4GB | 2 | 10-15 min | 8 |
| 8GB | 4 | 5-8 min | 16 |
| 16GB+ | 8+ | 2-5 min | 32+ |

## 📚 Additional Resources

- **Documentation**: `/docs` folder
- **Examples**: `/examples` folder  
- **Sample Data**: `/data/samples`
- **Neo4j Queries**: `/queries` folder

## 🤝 Support

### Getting Help
1. Check this README first
2. Run diagnostics: `python install.py --check`
3. Check logs in Streamlit interface
4. Create issue on GitHub with system info

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests and documentation
4. Submit pull request

---

**Note**: This system is for research and educational purposes. Always consult healthcare professionals for medical decisions.
