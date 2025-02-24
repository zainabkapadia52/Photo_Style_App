# Photo Style Consistency App

[![**Open in Streamlit**](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zainabkapadia52-photo-style.streamlit.app)

## Overview
This web application applies histogram matching to transfer the tonal characteristics of a target (reference) image to one or more source images. By matching the brightness, contrast, and overall pixel intensity distribution, the app helps you achieve a consistent style across a collection of images—perfect for photographers, content creators, and designers.

## Features
- **Target & Source Upload:** Upload a target style image and multiple source images.
- **Histogram Matching with Blending:** Adjust the strength of the effect using an interactive blending slider.
- **Color Image Processing:** Processes images in the YCrCb color space to preserve color information.
- **Interactive Visualization:** View interactive histograms for the source, target, and processed images using Plotly.
- **Download Options:** Download the processed images directly from the interface.

## Technologies Used
- Python
- Streamlit
- OpenCV
- NumPy
- Plotly
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zainabkapadia52/Photo_Style_Consistency_App.git
2. Navigate to the project directory:
   ```bash
   cd Photo_Style_Consistency_App
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
2. Open your web browser and go to `http://localhost:8501`
3. Upload a target style image and one or more source images.
4. Adjust the blending factor and view the resulting processed images along with their histograms.
5. Download the processed images using the provided download buttons.

## Contributing
Contributions to improve the app are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.
