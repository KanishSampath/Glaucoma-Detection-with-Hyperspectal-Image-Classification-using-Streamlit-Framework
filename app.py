import streamlit as st
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import io
from datetime import datetime

# Device configuration
device = torch.device("cpu")

# Load the model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Glaucoma vs. Non-Glaucoma
model.load_state_dict(torch.load('glaucoma_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize DataFrame in session state
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = pd.DataFrame(columns=["Image", "Prediction", "Probability"])

# Prediction function
def predict_image(image):
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities

# Enhanced PDF Report Generation
def generate_pdf(dataframe):
    pdf = FPDF()
    pdf.add_page()
    pdf.image("logo.jpg", x=10, y=8, w=30)
    pdf.ln(30)
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="SRI RAMAKRISHNA HOSPITAL", ln=True, align="C")
    pdf.set_font("Arial", style="B", size=20)
    pdf.cell(200, 10, txt="Glaucoma Detection Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(60, 10, "Image", 1)
    pdf.cell(60, 10, "Prediction", 1)
    pdf.cell(60, 10, "Probability", 1)
    pdf.ln()
    pdf.set_font("Arial", size=12)
    for index, row in dataframe.iterrows():
        pdf.cell(60, 10, str(row["Image"]), 1)
        pdf.cell(60, 10, str(row["Prediction"]), 1)
        pdf.cell(60, 10, f"{row['Probability']:.4f}", 1)
        pdf.ln()
    pdf.ln(10)
    pdf.set_font("Arial", style="I", size=10)
    pdf.cell(200, 10, txt="Thank you for using Glaucoma Detection App", ln=True, align="C")
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    return pdf_output

# Streamlit Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Visualization", "Report Generation"])

# Home Page
if page == "Home":
    st.title("Glaucoma Detection Using Hyper Spectral Image Classification")
    
    # About Section
    st.subheader("About the Project")
    st.write("""
    This application detects **glaucoma** using **fundus images** and deep learning techniques. It helps in **early diagnosis** and assists healthcare professionals in making informed decisions.
    
    **Motivation:**
    - Glaucoma is a **leading cause of blindness** worldwide.
    - Early detection can **prevent vision loss**.
    - AI-powered diagnosis can **aid doctors** in faster and more accurate assessments.
    
    **Technologies Used:**
    - **Deep Learning** (ResNet-18 model)
    - **PyTorch** (for model training)
    - **Streamlit** (for web-based UI)
    - **Pandas & Plotly** (for data visualization)
    - **FPDF** (for PDF report generation)
    
    """)
    
    if st.button("Instructions"):
        st.info("""
        1. **Upload an Image:** Click on the upload button and select a fundus image.
        2. **View Prediction:** The system will classify the image as **Glaucoma** or **Non-Glaucoma**.
        3. **Check Probability:** The confidence level of the prediction will be displayed.
        4. **Visualize Data:** Visit the **Data Visualization** tab for graphical insights.
        5. **Generate Report:** Go to **Report Generation** and download a detailed report.
        """)
    
    st.write("Upload an image for Glaucoma classification")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        predicted_class, probabilities = predict_image(image)
        result = "Glaucoma" if predicted_class == 1 else "Non-Glaucoma"
        probability = probabilities[0][predicted_class].item()
        st.write(f"Predicted Class: {result}")
        st.write(f"Probability: {probability:.4f}")
        new_row = {"Image": uploaded_file.name, "Prediction": result, "Probability": probability}
        st.session_state.prediction_data = pd.concat([st.session_state.prediction_data, pd.DataFrame([new_row])], ignore_index=True)

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization of Glaucoma Detection Results")
    if not st.session_state.prediction_data.empty:
        st.subheader("Classification Distribution")
        grouped_data = st.session_state.prediction_data.groupby("Prediction").size().reset_index(name="Count")
        fig = px.bar(grouped_data, x="Prediction", y="Count", title="Glaucoma vs. Non-Glaucoma")
        st.plotly_chart(fig)
        st.subheader("Probability Distribution")
        fig2 = px.histogram(st.session_state.prediction_data, x="Probability", title="Probability Distribution of Predictions")
        st.plotly_chart(fig2)
    else:
        st.warning("No data available for visualization. Please make some predictions first.")

# Report Generation Page
elif page == "Report Generation":
    st.title("Generate and Download Reports")
    st.dataframe(st.session_state.prediction_data)
    if st.button("Generate PDF Report"):
        pdf_file = generate_pdf(st.session_state.prediction_data)
        st.success("PDF Report Generated!")
        st.download_button("Download PDF", data=pdf_file, file_name="glaucoma_report.pdf", mime="application/pdf")
