# NEST ML Service

This is the ML service for NEST, an AI-powered online examination proctoring system. This repository contains the backend code for the ML service.


![NEST Architecture](./images/architecture/architecture-nest.webp)

## Project Architecture

The NEST system consists of several interconnected components:

### 1. Browser Extension (Chrome/Edge)
- Built with React and Manifest V3
- Tracks user's mouse movements and keyboard activity
- Communicates with the exam interface via WebSocket/API
- Sends event logs to the backend for analysis

### 2. Frontend ([Link](https://github.com/ProctorAI/exam-frontend))
- Marketing website and exam portal demo
- Built with Next.js and TailwindCSS
- Features:
  - Landing page showcasing the product
  - Interactive demo of the exam interface
  - Documentation for integration


### 3. Backend ([Link](https://github.com/ProctorAI/admin-dashboard))
- Built with Next.js for API routes
- Handles:
  - Event log processing from the extension
  - Communication with Supabase database
  - Data serving to admin dashboard

### 4. AI/ML Service ([This Repository](https://github.com/ProctorAI/risk-classifier))
- Built with FastAPI and Python
- Features:
  - Fetches user logs from Supabase
  - Runs ML models for suspicious activity detection
  - Assigns risk scores to potential cheating events
  - Stores predictions back in Supabase
  - humanly intuited

### 5. Admin Dashboard ([This Repository](https://github.com/ProctorAI/admin-dashboard))
- Built with Next.js, and TailwindCSS
- Features:
  - Real-time monitoring of exam sessions
  - Flagged events and suspicious activity display
  - Risk score visualization

## ML Feature Extraction

The system extracts behavioral features from three main categories:

### 1. Mouse Features (30% of Total Score)
Key indicators of suspicious mouse behavior:
- `avg_norm_x`, `avg_norm_y`: Average normalized cursor position
- `std_norm_x`, `std_norm_y`: Cursor movement variability
- `top_edge_time`, `bottom_edge_time`: Time spent at screen edges
- `idle_percentage`: Percentage of time without movement

### 2. Keyboard Features (35% of Total Score)
Key indicators of suspicious keyboard behavior:
- `key_press_rate`: Frequency of keystrokes
- `shortcut_key_ratio`: Proportion of shortcut key usage
- `backspace_ratio`: Frequency of corrections
- `rapid_key_ratio`: Bursts of rapid typing
- `clipboard_operation_rate`: Frequency of copy/paste actions

### 3. Window State Features (35% of Total Score)
Key indicators of suspicious window behavior:
- `blur_count`: Number of window focus losses
- `tab_switch_count`: Frequency of tab switching
- `total_blur_duration`: Time spent in other windows
- `rapid_switch_count`: Quick switches between windows/tabs
- `suspicious_resize_count`: Suspicious window resizing events

## Scoring System

The system uses a weighted scoring approach to detect suspicious behavior:

1. Individual Category Scores (0-100):
   ```python
   def calculate_mouse_score(features):
       # Higher weights for edge time and sudden movements
       score = (
           30 * normalize(features['idle_percentage']) +
           35 * normalize(features['top_edge_time'] + features['bottom_edge_time']) +
           35 * normalize(features['std_norm_x'] + features['std_norm_y'])
       )
       return min(100, score)

   def calculate_keyboard_score(features):
       # Higher weights for clipboard and rapid typing
       score = (
           25 * normalize(features['shortcut_key_ratio']) +
           30 * normalize(features['clipboard_operation_rate']) +
           25 * normalize(features['rapid_key_ratio']) +
           20 * normalize(features['backspace_ratio'])
       )
       return min(100, score)

   def calculate_window_score(features):
       # Higher weights for blur duration and rapid switching
       score = (
           30 * normalize(features['total_blur_duration']) +
           30 * normalize(features['rapid_switch_count']) +
           20 * normalize(features['tab_switch_count']) +
           20 * normalize(features['suspicious_resize_count'])
       )
       return min(100, score)
   ```

2. Combined Risk Score:
   ```python
   def calculate_total_score(mouse_score, keyboard_score, window_score):
       return (
           0.30 * mouse_score +
           0.35 * keyboard_score +
           0.35 * window_score
       )
   ```

3. Risk Levels:
   - Low Risk: 0-30
   - Medium Risk: 31-70
   - High Risk: 71-100


## ML based scoring: 

- The ML model is trained on a dataset of suspicious and non-suspicious behavior.
- The model is trained using a combination of mouse, keyboard, and window features.
- We use a Random Forest Model and a Multivariate Regression Model to assign scores to the events.

## Getting Started

1. Clone the repository

# Risk Classifier API

FastAPI application for real-time risk classification and feature extraction from proctoring data.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

3. Run the development server:
```bash
uvicorn src.main:app --reload --port 8000
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Name: risk-classifier-api
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`

4. Add environment variables:
   - SUPABASE_URL
   - SUPABASE_KEY


## Available Endpoints

### Features
- POST `/features/extract`: Extract features from proctoring events
- GET `/features/test`: Test endpoint

### Scoring
- POST `/scoring/calculate`: Calculate risk scores
- GET `/scoring/summary/{test_id}`: Get exam risk summary
