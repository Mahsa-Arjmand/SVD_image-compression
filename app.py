import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import pandas as pd
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Image Analysis & Compression with SVD",
    layout="wide",
    initial_sidebar_state="expanded"
)


def compress_image_svd(image, k):
    """Compress image using SVD"""
    img_array = np.array(image, dtype=np.float64)
    
    if len(img_array.shape) == 3:
        channels = []
        for channel in range(3):
            
            U, s, Vt = np.linalg.svd(img_array[:,:,channel], full_matrices=False)
            
            U_k = U[:, :k]
            s_k = np.diag(s[:k])
            Vt_k = Vt[:k, :]
        
            compressed = U_k @ s_k @ Vt_k
            channels.append(compressed)
        compressed_image = np.stack(channels, axis=2)
    else:
        U, s, Vt = np.linalg.svd(img_array, full_matrices=False)
        U_k = U[:, :k]
        s_k = np.diag(s[:k])
        Vt_k = Vt[:k, :]
        compressed_image = U_k @ s_k @ Vt_k
    
    compressed_image = np.clip(compressed_image, 0, 255)
    return compressed_image.astype(np.uint8)

def compress_image_jpeg(image, quality):
    """JPEG compression"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    return np.array(compressed)

def calculate_metrics(original, compressed):
    """Calculate image quality metrics"""
    
    if original.shape != compressed.shape:
        compressed = np.array(Image.fromarray(compressed).resize(
            (original.shape[1], original.shape[0]), Image.LANCZOS))
    
    
    if len(original.shape) == 3:
        original_gray = np.mean(original, axis=2)
        compressed_gray = np.mean(compressed, axis=2)
    else:
        original_gray = original
        compressed_gray = compressed
    
    data_range = 255.0
    
    try:
        psnr_value = psnr(original, compressed, data_range=data_range)
    except:
        psnr_value = 0
    
    try:
        ssim_value = ssim(original_gray, compressed_gray, data_range=data_range)
    except:
        ssim_value = 0
    
    try:
        mse_value = mse(original, compressed)
    except:
        mse_value = 0
    
    return {
        'PSNR': round(psnr_value, 2),
        'SSIM': round(ssim_value, 4),
        'MSE': round(mse_value, 2)
    }

def get_file_size_bytes(image_array, format="PNG"):
    """Calculate file size"""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.tell()

def resize_image(image, max_size=800):
    """Resize image"""
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image

def create_download_link(img_array, filename, text):
    """Create download link"""
    buffer = io.BytesIO()
    Image.fromarray(img_array).save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

def generate_report(original_info, results_df, k_values):
    """Generate automatic report"""
    report = f"""
     Image Compression Analysis Report
    {'='*50}
    
     Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
     Original Image Information:
    - Dimensions: {original_info['dimensions']}
    - Size: {original_info['size_kb']:.2f} KB
    - Channels: {original_info['channels']}
    
     SVD Compression Results:
    {'-'*30}
    """
    
    for k in k_values:
        row = results_df[results_df['k'] == k].iloc[0]
        report += f"""
    k = {k}:
    - PSNR: {row['PSNR']} dB
    - SSIM: {row['SSIM']}
    - MSE: {row['MSE']}
    - Size: {row['Compressed_Size_KB']:.2f} KB
    - Compression Ratio: {row['Compression_Ratio']:.2f}x
    """
    
    report += f"""
    {'='*50}
     Conclusion:
    - Best Quality (PSNR): k={results_df.loc[results_df['PSNR'].idxmax(), 'k']}
    - Best Compression: k={results_df.loc[results_df['Compression_Ratio'].idxmax(), 'k']}
    - Quality-Size Balance: k={results_df.loc[results_df['SSIM'].idxmax(), 'k']}
    """
    
    return report


st.title(" Image Analysis & Compression with SVD")
st.markdown("---")


with st.sidebar:
    st.header(" Settings")
    
    
    uploaded_file = st.file_uploader(
        " Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPG, PNG, BMP, TIFF"
    )
    
    if uploaded_file:
        
        st.markdown("---")
        st.subheader(" Processing Parameters")
        
        max_size = st.slider(
            "Maximum Image Dimension",
            min_value=256,
            max_value=2048,
            value=800,
            step=128,
            help="Large images will be resized to this dimension"
        )
        
       
        comparison_mode = st.checkbox(
            " Compare Multiple k Values",
            value=True,
            help="Show results for multiple k values"
        )
        
        if comparison_mode:
            k_values = st.multiselect(
                "k Values for Comparison:",
                options=[1, 5, 10, 20, 30, 50, 75, 100, 150, 200],
                default=[5, 20, 50, 100]
            )
        
        st.markdown("---")
        st.subheader(" Compression Settings")
        
        compression_method = st.radio(
            "Compression Method:",
            ["Number of Components (k)", "Compression Percentage", "Target Size (KB)"]
        )


if uploaded_file is not None:
    try:
        
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        image_resized = resize_image(image, max_size)
        original_array = np.array(image_resized)
        
        
        original_size_bytes = get_file_size_bytes(original_array)
        original_size_kb = original_size_bytes / 1024
        
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image_resized, caption="Original Image", use_container_width=True)

        
        with col2:
            st.markdown("###  Original Image Information")
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Dimensions", f"{original_size[0]}×{original_size[1]}")
                st.metric("Size", f"{original_size_kb:.1f} KB")
            with info_col2:
                st.metric("Channels", "RGB" if len(original_array.shape) == 3 else "Grayscale")
                st.metric("Max k", min(original_array.shape[0], original_array.shape[1]))
            
            
            fig_hist, ax_hist = plt.subplots(figsize=(4, 2))
            if len(original_array.shape) == 3:
                colors = ('r', 'g', 'b')
                for i, color in enumerate(colors):
                    ax_hist.hist(original_array[:,:,i].ravel(), bins=50, 
                               color=color, alpha=0.5, label=f'Channel {color.upper()}')
            else:
                ax_hist.hist(original_array.ravel(), bins=50, color='gray', alpha=0.7)
            ax_hist.set_title('Original Image Histogram')
            ax_hist.legend()
            st.pyplot(fig_hist)
        
        st.markdown("---")
        
        
        tab1, tab2, tab3, tab4 = st.tabs([
            " SVD Compression",
            " Method Comparison",
            " Quality Analysis",
            " Report"
        ])
        
        with tab1:
            st.header("SVD Compression")
            
            
            if compression_method == "Number of Components (k)":
                max_k = min(original_array.shape[0], original_array.shape[1])
                k = st.slider("Number of Components (k):", 1, max_k, min(50, max_k))
            elif compression_method == "Compression Percentage":
                compression_percentage = st.slider("Compression Percentage:", 1, 99, 50)
                max_k = min(original_array.shape[0], original_array.shape[1])
                k = max(1, int(max_k * (1 - compression_percentage / 100)))
                st.info(f"Calculated k: {k}")
            else:  
                target_size_kb = st.number_input("Target Size (KB):", min_value=1, value=100)
               
                k = max(1, int(np.sqrt(target_size_kb * 1024 / 8)))
                max_k = min(original_array.shape[0], original_array.shape[1])
                k = min(k, max_k)
                st.info(f"Estimated k: {k}")
            
            if st.button(" Perform Compression", type="primary", use_container_width=True):
                with st.spinner("Compressing..."):
                    
                    compressed_array = compress_image_svd(image_resized, k)
                    compressed_size = get_file_size_bytes(compressed_array)
                    
                    
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.image(compressed_array, caption=f"Compressed Image (k={k})", use_container_width=True)

                    
                    with col_result2:
                        st.markdown("###  Compression Results")
                        
                        
                        metrics = calculate_metrics(original_array, compressed_array)
                        
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("PSNR (dB)", f"{metrics['PSNR']:.2f}", 
                                     )
                            st.metric("SSIM", f"{metrics['SSIM']:.4f}",
                                     )
                        with metric_col2:
                            st.metric("MSE", f"{metrics['MSE']:.2f}")
                            compression_ratio = original_size_bytes / compressed_size
                            st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
                        
                        st.metric("Compressed Size", f"{compressed_size/1024:.1f} KB",
                                 delta=f"-{((1 - compressed_size/original_size_bytes) * 100):.1f}%")
                    
                    
                    st.markdown("###  Histogram Comparison")
                    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                    
                    if len(original_array.shape) == 3:
                        for i, (title, array) in enumerate([("Original", original_array), ("Compressed", compressed_array)]):
                            colors = ('r', 'g', 'b')
                            for j, color in enumerate(colors):
                                axes[i].hist(array[:,:,j].ravel(), bins=50, 
                                           color=color, alpha=0.5, label=f'{color.upper()}')
                            axes[i].set_title(f'{title} Image Histogram')
                            axes[i].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    
                    st.markdown("---")
                    col_dl1, col_dl2 = st.columns([1, 2])
                    with col_dl1:
                        buffer = io.BytesIO()
                        Image.fromarray(compressed_array).save(buffer, format="PNG")
                        st.download_button(
                            label=" Download Compressed Image",
                            data=buffer.getvalue(),
                            file_name=f"compressed_k{k}.png",
                            mime="image/png",
                            use_container_width=True
                        )
        
        with tab2:
            st.header(" Compression Method Comparison")
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.subheader("SVD with Different k Values")
                if comparison_mode and k_values:
                    if st.button(" Run SVD Comparison", use_container_width=True):
                        with st.spinner("Comparing..."):
                            results = []
                            
                            for k_val in k_values:
                                compressed = compress_image_svd(image_resized, k_val)
                                metrics = calculate_metrics(original_array, compressed)
                                comp_size = get_file_size_bytes(compressed)
                                
                                results.append({
                                    'k': k_val,
                                    'PSNR': metrics['PSNR'],
                                    'SSIM': metrics['SSIM'],
                                    'MSE': metrics['MSE'],
                                    'Compressed_Size_KB': comp_size / 1024,
                                    'Compression_Ratio': original_size_bytes / comp_size
                                })
                            
                           
                            st.session_state.svd_results = pd.DataFrame(results)
                            
                            
                            st.dataframe(
                                st.session_state.svd_results.style.background_gradient(
                                    subset=['PSNR', 'SSIM'], cmap='RdYlGn'),
                                use_container_width=True
                            )
                            
                            
                            st.markdown("###  Results Gallery")
                            cols = st.columns(min(4, len(k_values)))
                            for idx, k_val in enumerate(k_values):
                                compressed = compress_image_svd(image_resized, k_val)
                                with cols[idx % 4]:
                                    st.image(compressed, caption=f"k={k_val}", use_container_width=True)
            
            with col_comp2:
                st.subheader("SVD vs JPEG Comparison")
                jpeg_quality = st.slider("JPEG Quality:", 10, 100, 50)
                
                if st.button(" Compare SVD vs JPEG", use_container_width=True):
                    with st.spinner("Comparing..."):
                        
                        k_svd = st.session_state.get('last_k', 50)
                        svd_compressed = compress_image_svd(image_resized, k_svd)
                        
                        
                        jpeg_compressed = compress_image_jpeg(image_resized, jpeg_quality)
                        
                        
                        svd_metrics = calculate_metrics(original_array, svd_compressed)
                        jpeg_metrics = calculate_metrics(original_array, jpeg_compressed)
                        
                        svd_size = get_file_size_bytes(svd_compressed)
                        jpeg_size = get_file_size_bytes(jpeg_compressed)
                        
                        
                        comp_cols = st.columns(3)
                        with comp_cols[0]:
                            st.image(original_array, caption="Original", use_container_width=True)
                        with comp_cols[1]:
                            st.image(svd_compressed, caption=f"SVD (k={k_svd})", use_container_width=True)
                        with comp_cols[2]:
                            st.image(jpeg_compressed, caption=f"JPEG (Q={jpeg_quality})", use_container_width=True)
                        
                       
                        comparison_df = pd.DataFrame({
                            'Metric': ['PSNR', 'SSIM', 'MSE', 'Size (KB)'],
                            'SVD': [svd_metrics['PSNR'], svd_metrics['SSIM'], 
                                   svd_metrics['MSE'], f"{svd_size/1024:.1f}"],
                            'JPEG': [jpeg_metrics['PSNR'], jpeg_metrics['SSIM'], 
                                    jpeg_metrics['MSE'], f"{jpeg_size/1024:.1f}"]
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        methods = ['SVD', 'JPEG']
                        psnr_values = [svd_metrics['PSNR'], jpeg_metrics['PSNR']]
                        ssim_values = [svd_metrics['SSIM'], jpeg_metrics['SSIM']]
                        
                        x = np.arange(len(methods))
                        width = 0.35
                        
                        ax.bar(x - width/2, psnr_values, width, label='PSNR (dB)', color='blue', alpha=0.7)
                        ax_twin = ax.twinx()
                        ax_twin.bar(x + width/2, ssim_values, width, label='SSIM', color='green', alpha=0.7)
                        
                        ax.set_xticks(x)
                        ax.set_xticklabels(methods)
                        ax.set_ylabel('PSNR (dB)')
                        ax_twin.set_ylabel('SSIM')
                        ax.legend(loc='upper left')
                        ax_twin.legend(loc='upper right')
                        ax.set_title('SVD vs JPEG Comparison')
                        
                        st.pyplot(fig)
        
        with tab3:
            st.header(" Quality Analysis by k Value")
            
            if 'svd_results' in st.session_state:
                results_df = st.session_state.svd_results
                
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                
                axes[0, 0].plot(results_df['k'], results_df['PSNR'], 'b-o', linewidth=2)
                axes[0, 0].set_xlabel('k (Number of Components)')
                axes[0, 0].set_ylabel('PSNR (dB)')
                axes[0, 0].set_title('PSNR vs k')
                axes[0, 0].grid(True, alpha=0.3)
                
                
                axes[0, 1].plot(results_df['k'], results_df['SSIM'], 'g-o', linewidth=2)
                axes[0, 1].set_xlabel('k (Number of Components)')
                axes[0, 1].set_ylabel('SSIM')
                axes[0, 1].set_title('SSIM vs k')
                axes[0, 1].grid(True, alpha=0.3)
                
                
                axes[1, 0].plot(results_df['k'], results_df['MSE'], 'r-o', linewidth=2)
                axes[1, 0].set_xlabel('k (Number of Components)')
                axes[1, 0].set_ylabel('MSE')
                axes[1, 0].set_title('MSE vs k')
                axes[1, 0].grid(True, alpha=0.3)
                
               
                axes[1, 1].plot(results_df['k'], results_df['Compressed_Size_KB'], 'm-o', linewidth=2)
                axes[1, 1].set_xlabel('k (Number of Components)')
                axes[1, 1].set_ylabel('Size (KB)')
                axes[1, 1].set_title('File Size vs k')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            
                st.markdown("###  Quality-Size Balance Chart")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                
                scatter = ax2.scatter(results_df['Compressed_Size_KB'], 
                                     results_df['PSNR'],
                                     c=results_df['k'], 
                                     cmap='viridis', 
                                     s=100,
                                     alpha=0.6)
                
                
                for i, row in results_df.iterrows():
                    ax2.annotate(f"k={row['k']:.0f}", 
                               (row['Compressed_Size_KB'], row['PSNR']),
                               xytext=(5, 5), textcoords='offset points')
                
                ax2.set_xlabel('File Size (KB)')
                ax2.set_ylabel('PSNR (dB)')
                ax2.set_title('Quality vs File Size Trade-off')
                plt.colorbar(scatter, label='k')
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig2)
                
                
                st.markdown("### Optimal k Recommendation")
                
                col_opt1, col_opt2, col_opt3 = st.columns(3)
                
                with col_opt1:
                    best_quality_k = results_df.loc[results_df['PSNR'].idxmax(), 'k']
                    st.metric("Best Quality", f"k={best_quality_k:.0f}",
                             delta=f"PSNR: {results_df['PSNR'].max():.1f} dB")
                
                with col_opt2:
                    best_compression_k = results_df.loc[results_df['Compression_Ratio'].idxmax(), 'k']
                    st.metric("Best Compression", f"k={best_compression_k:.0f}",
                             delta=f"Ratio: {results_df['Compression_Ratio'].max():.1f}x")
                
                with col_opt3:
                    
                    normalized_psnr = (results_df['PSNR'] - results_df['PSNR'].min()) / \
                                    (results_df['PSNR'].max() - results_df['PSNR'].min())
                    normalized_size = (results_df['Compressed_Size_KB'] - results_df['Compressed_Size_KB'].min()) / \
                                     (results_df['Compressed_Size_KB'].max() - results_df['Compressed_Size_KB'].min())
                    balance_score = normalized_psnr - normalized_size
                    best_balance_k = results_df.loc[balance_score.idxmax(), 'k']
                    st.metric("Best Balance", f"k={best_balance_k:.0f}",
                             delta="Quality & Size")
            else:
                st.info(" First calculate SVD results in the 'Method Comparison' tab.")
        
        with tab4:
            st.header(" Automatic Report")
            
            if 'svd_results' in st.session_state:
               
                original_info = {
                    'dimensions': f"{original_size[0]}×{original_size[1]}",
                    'size_kb': original_size_kb,
                    'channels': 'RGB' if len(original_array.shape) == 3 else 'Grayscale'
                }
                
                report = generate_report(original_info, st.session_state.svd_results, k_values)
                
                
                st.text_area("Analysis Report", report, height=400)
                
                
                st.download_button(
                    label=" Download Report (TXT)",
                    data=report,
                    file_name=f"image_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                
                st.markdown("###  Statistical Summary")
                st.dataframe(st.session_state.svd_results.describe(), use_container_width=True)
            else:
                st.info(" First calculate SVD results in the 'Method Comparison' tab.")
    
    except Exception as e:
        st.error(f" Error processing image: {str(e)}")
        st.info("Please try another image or change settings.")

else:
    
    st.markdown("""
    
    ###  Features:
    -  Upload and display images
    -  SVD-based compression
    -  Comparison of different compression methods
    -  Image quality analysis (PSNR, SSIM, MSE)
    -  Interactive charts
    -  Automatic report generation
    -  Download compressed images
    
    ###  User Guide:
    1. Upload an image from the sidebar
    2. Adjust compression parameters
    3. View results in different tabs
    4. Download images and reports
    """)
    
    
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    with col_demo1:
        st.info(" SVD Compression")
    with col_demo2:
        st.info(" Quality Metrics")
    with col_demo3:
        st.info(" Analysis Charts")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Image Analysis with SVD"
    "</p>",
    unsafe_allow_html=True
)