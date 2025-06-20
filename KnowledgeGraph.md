# 🧠 NeuroGait ASD Knowledge Graph

A comprehensive knowledge graph system for analyzing gait patterns in Autism Spectrum Disorder (ASD) research using biomechanical motion capture data.

## 📋 Overview

This system creates a Neo4j knowledge graph from gait analysis data, enabling:
- **Semantic analysis** of biomechanical relationships
- **Pattern discovery** between ASD and neurotypical populations  
- **Anatomical network analysis** of body part connections
- **Statistical comparison** of movement patterns
- **Interactive querying** through natural language and Cypher

## 🏗️ System Architecture

```
📊 Raw Data (Excel) → 🔄 Parser → 🕸️ Knowledge Graph → 📈 Analysis → 📋 Reports
                                        ↓
                                   🔍 Query Interface
```

### Knowledge Graph Schema

- **🏃 Subjects**: Individual participants with ASD/Typical classifications
- **🦴 Body Parts**: Anatomical joints and segments (Head, Shoulders, Hips, etc.)
- **📏 Measurements**: Statistical measures (mean, variance, std) across dimensions (x,y,z)
- **🚶 Gait Parameters**: Temporal and spatial gait characteristics
- **🔗 Relationships**: Anatomical connections and measurement associations

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Make the setup script executable and run it
chmod +x setup_neurogait_kg.sh
./setup_neurogait_kg.sh
```

### 2. Start Neo4j Database
Choose one option:

**Option A: Neo4j Desktop**
- Download from https://neo4j.com/download/
- Create new project → Add local DBMS
- Set password to `palatiou`
- Start the database

**Option B: Docker**
```bash
docker run --name neo4j-neurogait \
  -p7474:7474 -p7687:7687 \
  -v $PWD/neo4j:/data \
  --env NEO4J_AUTH=neo4j/palatiou \
  neo4j:latest
```

### 3. Build Knowledge Graph
```bash
# Activate your environment
source neurogait_env/bin/activate

# Make sure you're in the directory with your Excel file
cd ~/NeuroGait_ASD

# Build the complete knowledge graph
python neurogait_kg_builder.py
```

### 4. Analyze the Data
```bash
# Launch interactive analysis interface
python neurogait_kg_query.py
```

## 📁 File Structure

```
NeuroGait_ASD/
├── 📄 Final dataset.xlsx              # Your gait analysis data
├── 🐍 neurogait_kg_builder.py         # Knowledge graph builder
├── 🔍 neurogait_kg_query.py           # Query and analysis interface
├── ⚙️ setup_neurogait_kg.sh           # Environment setup script
├── 🔧 .env                            # Configuration file
├── 📊 logs/                           # Application logs
├── 📈 outputs/                        # Generated reports and visualizations
└── 🗄️ neo4j/                          # Neo4j data directory (if using Docker)
```

## 🔍 Analysis Capabilities

### 1. Database Overview
- Node and relationship counts
- Classification distributions
- Body region analysis

### 2. Classification Comparison
- Statistical differences between ASD and Typical groups
- Effect size calculations
- Sample size analysis

### 3. Gait Parameter Analysis
- Temporal parameters (cycle time, stance/swing phases)
- Spatial parameters (step length, stride width)
- Kinematic features (velocity, acceleration)

### 4. Bilateral Symmetry Analysis
- Left-right asymmetry patterns
- Classification-specific symmetry differences

### 5. Anatomical Network Analysis
- Body part connectivity patterns
- Kinematic chain relationships

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=palatiou

# Optional: OpenAI for future LLM integration
OPENAI_API_KEY=your_api_key_here
```

### Neo4j Browser Access
- URL: http://localhost:7474
- Username: neo4j
- Password: palatiou

## 📊 Example Queries

### Basic Statistics
```cypher
// Count all nodes by type
MATCH (n) 
RETURN labels(n)[0] as type, count(n) as count 
ORDER BY count DESC

// Classification distribution
MATCH (s:Subject)-[:CLASSIFIED_AS]->(c:Classification)
RETURN c.label, count(s) as subjects
```

### Analysis Queries
```cypher
// Average gait velocity by classification
MATCH (s:Subject)-[rel:HAS_GAIT_VALUE]->(gp:GaitParameter {code: 'Velocity'})
MATCH (s)-[:CLASSIFIED_AS]->(c:Classification)
RETURN c.label as classification, 
       avg(rel.value) as avg_velocity,
       count(rel.value) as sample_size

// Body parts with highest measurement variance
MATCH (s:Subject)-[rel:HAS_VALUE]->(bp:BodyPart)
WHERE rel.measurement_type = 'variance'
RETURN bp.name as body_part,
       rel.dimension as dimension,
       avg(rel.value) as avg_variance
ORDER BY avg_variance DESC
LIMIT 10
```

### Advanced Pattern Discovery
```cypher
// Find subjects with unusual gait patterns
MATCH (s:Subject)-[rel:HAS_GAIT_VALUE]->(gp:GaitParameter)
WITH s, avg(rel.value) as avg_gait_value
WHERE avg_gait_value > 2 * stdev(avg_gait_value)
MATCH (s)-[:CLASSIFIED_AS]->(c:Classification)
RETURN s.id, c.label, avg_gait_value
ORDER BY avg_gait_value DESC
```

## 📈 Generated Outputs

### 1. Analysis Reports
- HTML comprehensive reports
- Statistical comparisons
- Effect size calculations

### 2. Visualizations
- Network diagrams
- Statistical plots
- Distribution comparisons

### 3. Data Exports
- CSV files with analysis results
- JSON format for further processing

## 🛠️ Customization

### Adding New Analysis Functions
```python
# In neurogait_kg_query.py
def your_custom_analysis(self) -> pd.DataFrame:
    """Your custom analysis description"""
    query = """
    // Your Cypher query here
    MATCH (n:YourNodeType)
    RETURN n.property as result
    """
    results = self.execute_query(query)
    return pd.DataFrame(results)
```

### Extending the Graph Schema
```python
# In neurogait_kg_builder.py
def create_new_node_type(self):
    """Add new node types to the graph"""
    with self.driver.session() as session:
        session.run("""
            CREATE (n:NewNodeType {
                property1: $value1,
                property2: $value2
            })
        """, value1="example", value2=123)
```

## 🔬 Research Applications

### Clinical Research
- **Biomarker Discovery**: Identify gait features that distinguish ASD
- **Severity Assessment**: Correlate movement patterns with ASD severity
- **Intervention Tracking**: Monitor changes over time

### Computational Analysis
- **Pattern Recognition**: ML models on graph-embedded features
- **Network Analysis**: Anatomical connectivity differences
- **Predictive Modeling**: Classification and regression tasks

### Educational Applications
- **Movement Assessment**: Objective evaluation tools
- **Intervention Planning**: Targeted therapy recommendations
- **Progress Monitoring**: Longitudinal tracking systems

## 🤝 Contributing

### Data Integration
- Ensure Excel files follow the expected format
- Body part naming conventions should match the schema
- Classification labels: 'A' for ASD, 'T' for Typical

### Code Contributions
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## 📚 References

### Scientific Background
- Gait analysis in autism spectrum disorders
- Knowledge graphs for medical data
- Neo4j for biomedical applications

### Technical Documentation
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/)
- [Python Neo4j Driver](https://neo4j.com/docs/python-manual/)

## 🆘 Troubleshooting

### Common Issues

**Connection Failed**
```bash
# Check if Neo4j is running
netstat -an | grep 7687

# Verify credentials in .env file
cat .env | grep NEO4J
```

**Memory Issues**
```bash
# Increase Neo4j memory (in neo4j.conf)
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
```

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**File Not Found Errors**
```bash
# Check you're in the correct directory
pwd  # Should show ~/NeuroGait_ASD or similar

# Verify file exists
ls -la "Final dataset.xlsx"

# If file is elsewhere, update the path in neurogait_kg_builder.py
```

### Performance Optimization

**Large Datasets**
- Use `sample_size` parameter during testing
- Create indexes on frequently queried properties
- Use PROFILE in Cypher queries to optimize

**Query Performance**
```cypher
// Use indexes
CREATE INDEX ON :Subject(classification_code)
CREATE INDEX ON :BodyPart(name)

// Profile slow queries
PROFILE MATCH (s:Subject)-[r:HAS_VALUE]->(bp:BodyPart)
RETURN count(*)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions, issues, or collaborations:
- Repository: https://github.com/GiorgosBouh/NeuroGait_ASD.git
- Issues: Use GitHub Issues for bug reports and feature requests

---

**🧠 Happy Knowledge Graph Analysis! 🧠**