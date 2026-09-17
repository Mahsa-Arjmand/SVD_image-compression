# SVD Image Compression

A web-based application for image analysis and compression using Singular Value Decomposition (SVD). Built with Streamlit, this tool provides an interactive interface to explore image compression techniques and compare different methods.

## Features

- **Image Upload & Display**: Support for multiple image formats (JPG, PNG, BMP, TIFF)
- **SVD Compression**: Compress images using Singular Value Decomposition with adjustable component count (k)
- **Multiple Compression Methods**:
  - Number of components (k)
  - Compression percentage
  - Target file size
- **Quality Metrics**: Calculate PSNR, SSIM, and MSE to assess image quality
- **Method Comparison**: Compare SVD compression with JPEG compression
- **Interactive Visualizations**:
  - Histogram comparisons
  - Quality vs. compression ratio charts
  - Multi-k value comparison gallery
- **Automatic Report Generation**: Generate detailed analysis reports
- **Download Options**: Download compressed images and analysis reports

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Mahsa-Arjmand/SVD_image-compression.git
cd SVD_image-compression
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### How to Use

1. **Upload an Image**: Use the sidebar to upload an image file
2. **Adjust Settings**: Configure compression parameters:
   - Maximum image dimension
   - Compression method (k value, percentage, or target size)
   - Multiple k values for comparison
3. **Explore Tabs**:
   - **SVD Compression**: Perform single compression with custom parameters
   - **Method Comparison**: Compare SVD with different k values and JPEG
   - **Quality Analysis**: View quality metrics and charts
   - **Report**: Generate and download analysis reports
4. **Download Results**: Download compressed images and generated reports

## Technical Details

### SVD Compression Algorithm

The application uses Singular Value Decomposition to compress images by:
1. Decomposing image matrices using SVD: `A = U * Σ * V^T`
2. Selecting the top k singular values and corresponding vectors
3. Reconstructing the image using only these components
4. Applying to each color channel (RGB) independently

### Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of reconstructed images
- **SSIM (Structural Similarity Index)**: Assesses perceptual image quality
- **MSE (Mean Squared Error)**: Calculates the average squared difference between images

## Dependencies

- streamlit
- numpy
- Pillow
- matplotlib
- scikit-learn
- scikit-image
- pandas

## Project Structure

```
SVD_image-compression/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Author

Mahsa Arjmand

## Acknowledgments

- Built with Streamlit
- Uses NumPy for numerical computations
- Image quality metrics from scikit-image