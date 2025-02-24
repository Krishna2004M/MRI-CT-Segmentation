import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
import nibabel as nib
import numpy as np
import streamlit as st
import openai, os
from dotenv import load_dotenv
from scipy.stats import ttest_ind

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Model loading function
def load_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # tumor, ventricles, gray matter
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
    ).cuda()

    model.load_state_dict(torch.load('segmentation_model.pth', map_location='cuda'))
    model.eval()
    return model

# Predict segmentation
def predict_segmentation(model, mri_path):
    mri = nib.load(mri_path)
    mri_image = mri.get_fdata()
    voxel_dims = mri.header.get_zooms()[:3]
    affine = mri.affine

    image_tensor = torch.tensor(mri_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

    # Resize image to model dimensions
    image_tensor = F.interpolate(image_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).cpu().numpy()

    return prediction.squeeze(), voxel_dims, affine

# Summarize segmentation findings with statistical analysis
def summarize_findings(segmentation_result, voxel_dims, affine):
    voxel_volume = np.prod(voxel_dims)
    tumor_voxels = np.sum(segmentation_result == 1)
    ventricles_voxels = np.sum(segmentation_result == 2)
    gray_matter_voxels = np.sum(segmentation_result == 0)

    tumor_volume = tumor_voxels * voxel_volume / 1000  # in cm³
    ventricles_volume = ventricles_voxels * voxel_volume / 1000  # in cm³
    gray_matter_volume = gray_matter_voxels * voxel_volume / 1000  # in cm³

    # Statistical comparison (example: tumor vs. ventricles)
    stat, p_value = ttest_ind(segmentation_result.flatten() == 1, segmentation_result.flatten() == 2, equal_var=False)

    # Tumor centroid calculation
    tumor_indices = np.argwhere(segmentation_result == 1)
    if tumor_indices.size > 0:
        centroid_voxel = np.mean(tumor_indices, axis=0)
        centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)
        anatomical_location = f"Tumor centroid coordinates (mm): {', '.join(map(str, centroid_mm.round(2).tolist()))}"
    else:
        anatomical_location = "Tumor not clearly detected."

    summary = f"""
    Tumor Volume: {tumor_volume:.2f} cm³
    Ventricles Volume: {ventricles_volume:.2f} cm³
    Gray Matter Volume: {gray_matter_volume:.2f} cm³
    {anatomical_location}
    Statistical Comparison (Tumor vs Ventricles): p-value = {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not significant'})
    Data Augmentation Techniques Used: Affine transformations, Intensity shifts.
    """
    return summary

# Streamlit UI
st.title("🧠 Enhanced MRI/CT Clinical Analyzer & Report Generator")

uploaded_file = st.file_uploader("Upload MRI/CT Scan (.nii)", type=['nii'])

if uploaded_file:
    with open("uploaded_scan.nii", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("🔍 Performing Enhanced Multi-class Segmentation...")
    model = load_model()
    segmentation_result, voxel_dims, affine = predict_segmentation(model, "uploaded_scan.nii")

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

    report = response.choices[0].message.content
    st.subheader("📜 Comprehensive Clinical Report")
    st.write(report)
