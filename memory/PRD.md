# Dry Bean Classification System - PRD

## Original Problem Statement
Build a dry bean classification project using the user's train_dataset.csv file. Frontend using HTML and backend using Python for ML-based classification with data visualization and prediction capabilities.

## Architecture
- **Frontend**: React.js with pure CSS styling (dark theme)
- **Backend**: Python FastAPI with scikit-learn ML
- **Database**: MongoDB (not actively used - data from CSV)
- **ML Model**: Random Forest Classifier

## User Personas
1. **Researchers/Students**: Analyzing dry bean datasets for academic purposes
2. **Agricultural Analysts**: Classifying beans for quality assessment
3. **ML Learners**: Understanding classification workflows

## Core Requirements (Static)
- [x] Load and analyze dry bean dataset
- [x] Train ML model for classification
- [x] Predict bean class from 16 features
- [x] Display class distribution visualization
- [x] Show feature statistics
- [x] Provide sample data for testing

## What's Been Implemented (Jan 6, 2026)

### Backend (Python/FastAPI)
- `/api/dataset-info` - Dataset information (2500 samples, 16 features, 7 classes)
- `/api/statistics` - Feature statistics (mean, std, min, max)
- `/api/sample-data` - Sample data for each bean class
- `/api/model-status` - Model training status
- `/api/train` - Train Random Forest model
- `/api/predict` - Predict bean class from input features

### Frontend (React + CSS)
- Dashboard with stats cards (samples, features, classes, accuracy)
- Class distribution bar chart
- Model training section with accuracy display
- Bean classification form with 16 input fields
- Sample data buttons to pre-fill form
- Prediction results with confidence scores
- Feature statistics table

### ML Model
- **Algorithm**: Random Forest Classifier (100 trees, max_depth=15)
- **Accuracy**: 92.2%
- **Classes**: DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY

## Dataset Information
- **Source**: User-provided train_dataset.csv
- **Samples**: 2,500
- **Features**: 16 (Area, Perimeter, MajorAxisLength, etc.)
- **Classes**: 7 bean types

## Prioritized Backlog

### P0 (Critical) - DONE
- [x] Dataset loading and analysis
- [x] ML model training
- [x] Prediction endpoint
- [x] Frontend dashboard

### P1 (High Priority)
- [ ] CSV upload functionality for custom datasets
- [ ] Model comparison (SVM, KNN, etc.)
- [ ] Export predictions to CSV

### P2 (Nice to Have)
- [ ] Batch prediction mode
- [ ] Feature importance visualization
- [ ] Model hyperparameter tuning UI
- [ ] Historical prediction tracking

## Next Tasks
1. Add file upload for custom CSV datasets
2. Implement multiple ML model comparison
3. Add confusion matrix visualization
4. Export results functionality
