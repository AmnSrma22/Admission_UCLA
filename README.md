# 🎓 UCLA Graduate Admission Score Estimator
Viwe the app from here "https://admissionucla-awuvovuhfebyarq7glecvu.streamlit.app/"

Predict a candidate’s likelihood of being accepted to UCLA using test scores, GPA, and research background. Built with a neural network and deployed via Streamlit.

## 📌 Highlights
- Uses MLPRegressor to predict a probability (0.0 to 1.0)
- Inputs: GRE, TOEFL, SOP, LOR, CGPA, Research experience
- Returns chance of admission and visualizes prediction performance
- Real-time prediction UI built with Streamlit
- Saves trained model for fast access

## 🔍 Contents
- `streamlit.py`: Streamlit web interface
- `main.py`: Training script
- `models/`: Saved trained neural model
- `src/`: Code for data loading, feature engineering, training, and evaluation

## ▶️ Launch the App
```bash
streamlit run streamlit.py
```

## ⚙️ Dependencies
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn