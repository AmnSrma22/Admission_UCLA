import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_residuals(y_true, y_pred):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel('Actual Chance')
    plt.ylabel('Predicted Chance')
    plt.title('Residual Plot')
    st.pyplot(plt.gcf())