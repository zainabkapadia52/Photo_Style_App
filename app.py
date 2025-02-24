import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

# --------------------------
# Helper Functions
# --------------------------
def equalized_histogram(source_img):
    height, width = source_img.shape
    hist = np.zeros(256)
    for i in range(height):
        for j in range(width):
            intensity = source_img[i, j]
            hist[intensity] += 1
    total_pixels = np.sum(hist)
    hist_norm = hist / total_pixels
    cdf = np.cumsum(hist_norm)
    mapped_bins = np.round(cdf * 255).astype(np.uint8)
    eq_img = mapped_bins[source_img]
    eq_hist = np.zeros(256)
    for i in range(height):
        for j in range(width):
            intensity = eq_img[i, j]
            eq_hist[intensity] += 1
    eq_cdf = np.cumsum(eq_hist / total_pixels)
    return eq_img, eq_cdf, eq_hist

def find_value_target(val, target_cdf):
    min_diff = float('inf')
    chosen_index = 0
    for i in range(256):
        diff = abs(target_cdf[i] - val)
        if diff < min_diff:
            min_diff = diff
            chosen_index = i
    return chosen_index

def eq_match_histogram(eq_source_img, src_cdf, eq_target_img, tgt_cdf, blend=1.0):
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = find_value_target(src_cdf[i], tgt_cdf)
    height, width = eq_source_img.shape
    matched_img = np.zeros_like(eq_source_img)
    for i in range(height):
        for j in range(width):
            new_val = mapping[eq_source_img[i, j]]
            matched_img[i, j] = np.uint8(blend * new_val + (1 - blend) * eq_source_img[i, j])
    return matched_img

def process_color_image(source_img, target_img, blend=1.0):
    # Convert both images from BGR to YCrCb
    source_ycrcb = cv2.cvtColor(source_img, cv2.COLOR_BGR2YCrCb)
    target_ycrcb = cv2.cvtColor(target_img, cv2.COLOR_BGR2YCrCb)
    
    # Use the Y (luminance) channel for histogram matching
    source_Y = source_ycrcb[:, :, 0]
    target_Y = target_ycrcb[:, :, 0]
    
    eq_source, src_cdf, _ = equalized_histogram(source_Y)
    eq_target, tgt_cdf, _ = equalized_histogram(target_Y)
    matched_Y = eq_match_histogram(eq_source, src_cdf, eq_target, tgt_cdf, blend=blend)
    
    # Replace Y channel in the source image with the matched Y channel
    source_ycrcb[:, :, 0] = matched_Y
    result_img = cv2.cvtColor(source_ycrcb, cv2.COLOR_YCrCb2BGR)
    return result_img, source_Y, target_Y, matched_Y

def plot_histogram(image, title='Histogram'):
    hist = np.bincount(image.ravel(), minlength=256)
    fig = px.bar(x=list(range(256)), y=hist, labels={'x': 'Pixel Intensity', 'y': 'Frequency'})
    fig.update_layout(title=title)
    return fig

# --------------------------
# Streamlit App Layout
# --------------------------
st.set_page_config(page_title="Photo Style Consistency App", layout="wide")

# Custom CSS for buttons and inputs
st.markdown("""
<style>
.stButton > button {
    background-color: #4CAF50;
    color: white;
}
.stTextInput > div > div > input {
    background-color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.image("placeholder.jpg", width=100)  # Replace with your logo image
    st.title("Controls")
    blend = st.slider("Blending Factor (0.0 = original, 1.0 = full effect)", 0.0, 1.0, 1.0, step=0.1)
    st.markdown("---")
    st.write("Upload a target style image and one or more source images. The target image determines the tonal distribution which will be transferred to the source images via histogram matching.")

# Main Content
st.title("📸 Photo Style Consistency App")
st.write("This app applies histogram matching to transfer the style (contrast, brightness, tonal distribution) from a target image to one or more source images.")

# File Uploaders
target_file = st.file_uploader("Upload Target Style Image", type=["jpg", "jpeg", "png"])
source_files = st.file_uploader("Upload Source Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if target_file is not None and source_files:
    # Process Target Image
    target_bytes = np.asarray(bytearray(target_file.read()), dtype=np.uint8)
    target_img = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)
    
    # Convert target image to YCrCb and extract Y channel
    target_ycrcb = cv2.cvtColor(target_img, cv2.COLOR_BGR2YCrCb)
    target_Y = target_ycrcb[:, :, 0]
    eq_target, tgt_cdf, _ = equalized_histogram(target_Y)
    
    st.subheader("Target Style Image")
    st.image(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), caption="Target Image", use_container_width=True)
    st.plotly_chart(plot_histogram(target_Y, "Target Histogram"), use_container_width=True, key="target_hist_main")
    
    # Process Each Source Image
    for idx, source_file in enumerate(source_files):
        source_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
        source_img = cv2.imdecode(source_bytes, cv2.IMREAD_COLOR)
        
        # Process the source image using the target style
        result_img, source_Y, _, matched_Y = process_color_image(source_img, target_img, blend=blend)
        
        st.markdown("---")
        st.subheader(f"Processed Image: {source_file.name}")
        
        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB), caption="Original Source Image", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
        
        # Display histograms for source, target, and matched images
        st.write("### Histograms")
        col_hist1, col_hist2, col_hist3 = st.columns(3)
        with col_hist1:
            st.plotly_chart(plot_histogram(source_Y, "Source Histogram"), use_container_width=True, key=f"source_hist_{idx}")
        with col_hist2:
            st.plotly_chart(plot_histogram(target_Y, "Target Histogram"), use_container_width=True, key=f"target_hist_{idx}")
        with col_hist3:
            st.plotly_chart(plot_histogram(matched_Y, "Matched Histogram"), use_container_width=True, key=f"matched_hist_{idx}")
        
        # Download button for processed image
        buf = BytesIO()
        plt.imsave(buf, cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), format='png')
        st.download_button(
            label="Download Processed Image",
            data=buf.getvalue(),
            file_name=f"processed_{source_file.name}",
            mime="image/png",
            key=f"download_{idx}"
        )
