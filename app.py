import torch
import torch.nn.functional as F
from monai.networks.nets import UNet # type: ignore
import nibabel as nib # type: ignore
import numpy as np
import streamlit as st
import openai
import os
from dotenv import load_dotenv
from scipy.stats import ttest_ind

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# **Set up device (CPU/GPU)**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Load Model**
def load_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # tumor, ventricles, gray matter
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
    ).to(device)  # Move model to the correct device

    model.load_state_dict(torch.load('segmentation_model.pth', map_location=device))
    model.eval()
    return model

# **Predict Segmentation**
def predict_segmentation(model, mri_path):
    try:
        mri = nib.load(mri_path)
        mri_image = mri.get_fdata()
        voxel_dims = mri.header.get_zooms()[:3]
        affine = mri.affine

        # Convert to tensor and move to device
        image_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Resize image to model dimensions
        image_tensor = F.interpolate(image_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).cpu().numpy()  # Move to CPU for further processing

        return prediction.squeeze(), voxel_dims, affine
    except Exception as e:
        return f"Error in MRI processing: {e}", None, None

# **Calculate p-value (Fixed NaN Issue)**
def calculate_p_value(segmentation_result):
    # Convert segmentation data into binary masks
    tumor_values = (segmentation_result.flatten() == 1).astype(int)
    ventricle_values = (segmentation_result.flatten() == 2).astype(int)

    # Ensure enough data exists for a valid comparison
    if np.sum(tumor_values) > 1 and np.sum(ventricle_values) > 1:
        try:
            stat, p_value = ttest_ind(tumor_values, ventricle_values, equal_var=False)
            return f"p-value = {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not significant'})"
        except Exception as e:
            return f"Error calculating p-value: {e}"
    else:
        return "Statistical comparison not possible (insufficient data)."

# **Summarize Findings**
def summarize_findings(segmentation_result, voxel_dims, affine):
    if segmentation_result is None:
        return "Error: Invalid segmentation data."

    voxel_volume = np.prod(voxel_dims)
    tumor_voxels = np.sum(segmentation_result == 1)
    ventricles_voxels = np.sum(segmentation_result == 2)
    gray_matter_voxels = np.sum(segmentation_result == 0)

    tumor_volume = tumor_voxels * voxel_volume / 1000  # cm³
    ventricles_volume = ventricles_voxels * voxel_volume / 1000  # cm³
    gray_matter_volume = gray_matter_voxels * voxel_volume / 1000  # cm³

    # Calculate tumor centroid (Handles empty segmentation properly)
    tumor_indices = np.argwhere(segmentation_result == 1)
    if tumor_indices.size > 0:
        centroid_voxel = np.mean(tumor_indices, axis=0)
        centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)
        anatomical_location = f"Tumor centroid coordinates (mm): {', '.join(map(str, centroid_mm.round(2).tolist()))}"
    else:
        anatomical_location = "Tumor not clearly detected."

    # Compute p-value (Using fixed function)
    p_value_text = calculate_p_value(segmentation_result)

    summary = f"""
    Tumor Volume: {tumor_volume:.2f} cm³
    Ventricles Volume: {ventricles_volume:.2f} cm³
    Gray Matter Volume: {gray_matter_volume:.2f} cm³
    {anatomical_location}
    Statistical Comparison (Tumor vs Ventricles): {p_value_text}
    Data Augmentation Techniques Used: Affine transformations, Intensity shifts.
    """
    return summary

# **Streamlit UI**
st.title("🧠 Enhanced MRI/CT Clinical Analyzer & Report Generator")

uploaded_file = st.file_uploader("Upload MRI/CT Scan (.nii)", type=['nii'])

if uploaded_file:
    with open("uploaded_scan.nii", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("🔍 Performing Enhanced Multi-class Segmentation...")
    model = load_model()
    segmentation_result, voxel_dims, affine = predict_segmentation(model, "uploaded_scan.nii")

    if isinstance(segmentation_result, str):  # Error handling
        st.error(segmentation_result)
    else:
        findings = summarize_findings(segmentation_result, voxel_dims, affine)
        st.write("**Detailed Multi-class Segmentation & Statistical Findings:**", findings)

        st.write("✍️ Generating Comprehensive GPT Clinical Report...")
        prompt = f"""
        MRI segmentation and statistical analysis results:
        {findings}

        Generate a structured clinical report explicitly including precise anatomical details, comprehensive morphological characteristics (borders, shape, necrosis, edema), statistical significance interpretation, detailed differential diagnoses (glioblastoma, meningioma, metastasis), explicit clinical management recommendations, and clearly mentioned data augmentation methods.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000
        )

        report = response["choices"][0]["message"]["content"]
        st.subheader("📜 Comprehensive Clinical Report")
        st.write(report)
