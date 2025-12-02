# MLOps and DataOps Architecture

A production-ready MLOps/DataOps architecture for multi-tenant, HIPAA/GDPR-regulated environments demonstrating cryptographic isolation, feature engineering, and Models-as-a-Service deployment.

## 🎯 Overview

This project implements a complete MLOps pipeline with:

1. **DataOps Foundation**: Multi-tenant data isolation with application-level encryption
2. **Automated Feature Engineering (AFE)**: Complex feature pipeline with versioning
3. **Feature Store**: Persistent feature storage preventing training-serving skew
4. **Models-as-a-Service (MaaS)**: End-to-end prediction pipeline from raw data to scores

## 🏗️ Architecture

```
Raw Data → Encryption → AFE Pipeline → Feature Store → Model Training → MaaS Endpoint
   ↓           ↓            ↓              ↓              ↓                ↓
Tenant A   Tenant A    Features      Versioned      Logistic        Propensity
Tenant B   Tenant B    v1.0          Storage        Regression      Scores
```

## ✨ Features

### DataOps Foundation
- **Multi-Tenant Isolation**: Separate encryption keys per tenant
- **Application-Level Encryption (ALE)**: SHA256-based encryption for sensitive data
- **Tenant Configuration**: JSON-based configuration management
- **Persistent Storage**: Encrypted data saved to disk

### Automated Feature Engineering
- **Complex Features**: 
  - `Recency_Days`: Days since last visit
  - `Frequency_Ratio`: Login count relative to tenant average
  - `High_Value_Segment`: Binary flag for top 20% purchase amounts
  - `Purchase_Login_Ratio`: Purchase amount per login
  - `Activity_Score`: Composite activity metric
- **Feature Versioning**: Versioned feature storage (v1.0, v2.0, etc.)
- **Training-Serving Consistency**: Prevents data drift through versioning

### Feature Store
- **Versioned Storage**: Features saved with version tags
- **Tenant-Specific**: Separate feature files per tenant
- **Consistency Checks**: Internal logging verifies feature versions
- **Read/Write Functions**: Standardized API for feature access

### Models-as-a-Service (MaaS)
- **Model Training**: Logistic Regression with feature scaling
- **Model Persistence**: Serialized models saved with joblib
- **Prediction Endpoint**: Complete pipeline from raw data to scores
- **End-to-End Flow**: Raw input → AFE → Model → Propensity Scores

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- joblib >= 1.2.0

## 🚀 Installation

1. Navigate to the project directory:
```bash
cd mlops-dataops-architecture
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the complete MLOps pipeline:

```bash
python mlops_dataops_engine.py
```

The script will:
1. Create directory structure (config, data, feature_store, models)
2. Generate tenant configuration with encryption keys
3. Create 100,000 rows of synthetic raw data
4. Apply encryption to sensitive data per tenant
5. Run Automated Feature Engineering pipeline
6. Save features to Feature Store (versioned)
7. Train Logistic Regression model for Tenant_A
8. Demonstrate MaaS prediction endpoint
9. Verify all files are saved to disk

## 📊 Output

### Generated Files
- `config/tenant_config.json` - Tenant configuration with encryption keys
- `data/raw_data_encrypted.csv` - Encrypted raw data (100,000 rows)
- `feature_store/tenant_A_v1.0.csv` - Tenant A features
- `feature_store/tenant_B_v1.0.csv` - Tenant B features
- `models/tenant_A_propensity_model.joblib` - Trained model
- `models/tenant_A_scaler.joblib` - Feature scaler

### Console Output
- Directory creation status
- Data generation statistics
- Feature engineering summary
- Model training metrics
- MaaS prediction demonstration
- File verification report

## 📁 Project Structure

```
mlops-dataops-architecture/
├── mlops_dataops_engine.py      # Main MLOps pipeline script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
├── config/                       # Configuration files
│   └── tenant_config.json        # Tenant config (included in repo)
├── data/                         # Encrypted data (gitignored)
│   └── raw_data_encrypted.csv
├── feature_store/                # Feature versions (gitignored)
│   ├── tenant_A_v1.0.csv
│   └── tenant_B_v1.0.csv
└── models/                       # Trained models (gitignored)
    ├── tenant_A_propensity_model.joblib
    └── tenant_A_scaler.joblib
```

## 🔧 Technical Details

### Data Specifications
- **Raw Data**: 100,000 rows
- **Tenants**: Tenant_A (60%), Tenant_B (40%)
- **Columns**: Tenant_ID, Sensitive_Data, Login_Count, Purchase_Amount, Last_Visit_Date
- **Time Range**: Data spread over 1 year

### Encryption
- **Method**: SHA256-based hashing (simplified for demo)
- **Per-Tenant Keys**: Unique encryption keys per tenant
- **Note**: In production, use proper AES encryption from cryptography library

### Feature Engineering
- **5 Engineered Features**: Recency, Frequency, High Value, Purchase Ratio, Activity Score
- **Feature Versioning**: v1.0 (extensible to v2.0, v3.0, etc.)
- **Tenant-Specific**: Features calculated relative to tenant averages

### Model Configuration
- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler for feature normalization
- **Target**: Synthetic binary classification (High Value/Activity based)
- **Serialization**: joblib for model persistence

### MaaS Endpoint
The `predict_propensity_maas()` function demonstrates:
1. Raw data input
2. Automatic AFE pipeline execution
3. Model loading from disk
4. Feature scaling
5. Propensity score generation

## 🎯 Use Cases

- **Multi-Tenant SaaS**: Isolate data and models per customer
- **HIPAA/GDPR Compliance**: Encrypt sensitive data per tenant
- **Feature Store Implementation**: Versioned feature management
- **MLOps Pipeline**: End-to-end model deployment workflow
- **Production ML Systems**: Demonstrates production-ready patterns

## 🔄 Workflow

1. **Setup**: Create directories and tenant configuration
2. **Data Generation**: Generate synthetic raw data
3. **Encryption**: Apply tenant-specific encryption
4. **Feature Engineering**: Run AFE pipeline
5. **Feature Store**: Save versioned features
6. **Model Training**: Train and persist model
7. **MaaS Deployment**: Demonstrate prediction endpoint
8. **Verification**: Confirm all files saved correctly

## 🔐 Security Notes

- **Encryption Keys**: Stored in config file (in production, use secure key management)
- **Sensitive Data**: Encrypted before storage
- **Tenant Isolation**: Separate encryption keys ensure data isolation
- **Model Security**: Models serialized with joblib (consider signing in production)

## 📈 Best Practices Demonstrated

- **Feature Versioning**: Prevents training-serving skew
- **Model Persistence**: Enables model versioning and rollback
- **Separation of Concerns**: DataOps, Feature Engineering, and MaaS are separate
- **Reproducibility**: Random seed ensures consistent results
- **Logging**: Internal logging for feature store operations

## 🔄 Reproducibility

All random operations use `random_state=42` to ensure:
- Consistent data generation across runs
- Reproducible model training
- Same results when re-running the pipeline

## 📝 Notes

- Generated files (data, features, models) are excluded from version control
- Configuration file is included for demonstration
- Encryption is simplified for demo purposes (use proper encryption in production)
- The system generates all data and models on first run

## 🤝 Contributing

This is a demonstration project. For production use, consider:
- Secure key management (AWS KMS, Azure Key Vault, etc.)
- Proper encryption libraries (cryptography)
- Database integration instead of CSV files
- API framework (FastAPI, Flask) for MaaS endpoint
- Model registry and versioning system
- Monitoring and observability
- CI/CD pipelines for model deployment

## 📄 License

This project is for educational and demonstration purposes.
