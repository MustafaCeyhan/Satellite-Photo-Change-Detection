import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import tempfile
import time
from openai import OpenAI
import base64
import os
import re


@st.cache_data
def load_image_from_upload(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def resize_images(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2


def preprocess_satellite_image(img):
    """Uydu görüntüleri için ön işleme"""
    # Histogram eşitlemesi - aydınlatma farklarını azaltır
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur - gürültüyü azaltır
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred


def align_images_enhanced(img1, img2):
    """Geliştirilmiş görüntü hizalama"""
    img1_gray = preprocess_satellite_image(img1)
    img2_gray = preprocess_satellite_image(img2)
    
    # ORB feature matching ile daha güvenilir hizalama
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    
    if des1 is not None and des2 is not None:
        # FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Homography matrix
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    aligned_img2 = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
                    return aligned_img2
        except:
            pass
    
    # Fallback to ECC
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(img1_gray, img2_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        aligned_img2 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_img2
    except:
        return img2


def align_images(img1, img2):
    """Eski hizalama fonksiyonu - geriye uyumluluk için"""
    return align_images_enhanced(img1, img2)


def evaluate_mask(pred_mask, gt_mask):
    # Daha güvenli normalizasyon
    pred_flat = (pred_mask > 127).astype(int)  # 127 threshold'u ile binary
    gt_flat = (gt_mask > 127).astype(int)
    
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
    
    return precision, recall, f1


def detect_changes_ssim(img1, img2, base_image, threshold=0.3):
    """Uydu görüntüleri için optimize edilmiş SSIM"""
    # Ön işleme
    gray1 = preprocess_satellite_image(img1)
    gray2 = preprocess_satellite_image(img2)
    
    # SSIM hesaplama - pencere boyutunu artır
    score, diff = ssim(gray1, gray2, full=True, win_size=11)
    
    # Diff'i normalize et
    diff_normalized = ((1 - diff) * 255).astype("uint8")
    
    # Adaptive threshold
    threshold_value = int(threshold * 255)
    _, mask = cv2.threshold(diff_normalized, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morfolojik operasyonlar - uydu görüntüleri için optimize
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Minimum alan filtresi - küçük gürültüleri kaldır
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    min_area = 100  # Minimum değişiklik alanı (piksel)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            filtered_contours.append(contour)
    
    # Temiz mask oluştur
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, filtered_contours, -1, 255, -1)
    
    # Çıktı görüntüsü
    output = base_image.copy()
    cv2.drawContours(output, filtered_contours, -1, (255, 0, 0), 2)
    
    return diff_normalized, clean_mask, output, score


def detect_changes_absdiff(img1, img2, base_image, threshold=60):
    """Uydu görüntüleri için optimize edilmiş AbsDiff"""
    gray1 = preprocess_satellite_image(img1)
    gray2 = preprocess_satellite_image(img2)
    
    # Absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Adaptive threshold veya Otsu threshold
    if threshold == 0:  # Otsu threshold kullan
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Kontur filtreleme
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # Temiz mask
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, filtered_contours, -1, 255, -1)
    
    output = base_image.copy()
    cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)
    
    return diff, clean_mask, output


def detect_changes_canny(img1, img2, base_image, low_thresh=50, high_thresh=150):
    """Uydu görüntüleri için optimize edilmiş Canny"""
    gray1 = preprocess_satellite_image(img1)
    gray2 = preprocess_satellite_image(img2)
    
    # Canny edge detection
    edges1 = cv2.Canny(gray1, low_thresh, high_thresh, apertureSize=3, L2gradient=True)
    edges2 = cv2.Canny(gray2, low_thresh, high_thresh, apertureSize=3, L2gradient=True)
    
    # Edge difference
    diff = cv2.absdiff(edges1, edges2)
    
    # Dilate edges to make them more visible
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff = cv2.dilate(diff, kernel, iterations=1)
    
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Kontur filtreleme
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 50]
    
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, filtered_contours, -1, 255, -1)
    
    output = base_image.copy()
    cv2.drawContours(output, filtered_contours, -1, (0, 0, 255), 2)
    
    return diff, clean_mask, output


def combine_methods(methods_results, base_image):
    """Üç yöntemi birleştiren hibrit yaklaşım"""
    if len(methods_results) < 2:
        return None, None
    
    masks = []
    method_names = []
    
    for method_name, result in methods_results.items():
        if result is not None and 'mask' in result:
            mask = result['mask']
            if mask is not None:
                masks.append(mask // 255)  # 0-1 arası normalize
                method_names.append(method_name)
    
    if len(masks) < 2:
        return None, None
    
    # Mask'leri birleştir (voting)
    combined_mask = np.zeros_like(masks[0])
    vote_count = np.sum(masks, axis=0)
    
    # En az 2 yöntem hemfikir olduğu yerleri al
    threshold = min(2, len(masks))
    combined_mask[vote_count >= threshold] = 255
    
    # Final output
    final_output = base_image.copy()
    contours, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final_output, contours, -1, (255, 255, 0), 2)  # Sarı renk
    
    return combined_mask.astype(np.uint8), final_output


def detect_changes_llm(img1, img2, api_key):
    import tempfile, base64
    from PIL import Image
    from openai import OpenAI
    import json
    import re
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_before, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_after:
        Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)).save(f_before.name)
        Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).save(f_after.name)
        before_b64 = base64.b64encode(open(f_before.name, "rb").read()).decode("utf-8")
        after_b64 = base64.b64encode(open(f_after.name, "rb").read()).decode("utf-8")
    
    client = OpenAI(api_key=api_key)
    
    # İlk olarak genel analiz yap
    general_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare the following two satellite images. Describe any visual differences such as new buildings, changes in vegetation, roads, or land use. Sonucu Türkçe yaz."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{before_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{after_b64}"}},
                ],
            }
        ],
        max_tokens=1000
    )
    
    general_analysis = general_response.choices[0].message.content
    
    # Eğer değişiklik tespit edildiyse, koordinat tespiti yap
    if any(keyword in general_analysis.lower() for keyword in ['değişiklik', 'fark', 'yeni', 'eklendi', 'kaldırıldı', 'artmış', 'azalmış']):
        coordinate_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """Bu iki uydu görüntüsü arasındaki değişiklikleri analiz et ve değişen bölgelerin yaklaşık koordinatlarını belirt. 
                        Görüntüyü sol üst köşeden (0,0) başlayarak yüzde cinsinden koordinatlarla tarif et. 
                        Örnek format:
                        - Değişiklik türü: [açıklama]
                        - Konum: x=%10-20, y=%30-40
                        - Değişiklik türü: [açıklama]  
                        - Konum: x=%60-70, y=%20-30
                        
                        Sonucu Türkçe yaz."""},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{before_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{after_b64}"}},
                    ],
                }
            ],
            max_tokens=1000
        )
        coordinate_analysis = coordinate_response.choices[0].message.content
    else:
        coordinate_analysis = None
    
    return general_analysis, coordinate_analysis


def parse_llm_coordinates(coordinate_text, img_height, img_width):
    """LLM'den gelen koordinat bilgilerini parse et ve bounding box'lar oluştur"""
    if not coordinate_text:
        return []
    
    boxes = []
    # x=%10-20, y=%30-40 formatını ara
    pattern = r'x=%?(\d+)-(\d+).*?y=%?(\d+)-(\d+)'
    matches = re.findall(pattern, coordinate_text)
    
    for match in matches:
        x1_pct, x2_pct, y1_pct, y2_pct = map(int, match)
        
        # Yüzde değerlerini piksel koordinatlarına çevir
        x1 = int((x1_pct / 100) * img_width)
        x2 = int((x2_pct / 100) * img_width)
        y1 = int((y1_pct / 100) * img_height)
        y2 = int((y2_pct / 100) * img_height)
        
        boxes.append(((x1, y1), (x2, y2)))
    
    return boxes


def create_llm_visualization(img1, img2, coordinate_text):
    """LLM koordinatlarına göre görselleştirme oluştur"""
    if not coordinate_text:
        return None
    
    # Koordinatları parse et
    boxes = parse_llm_coordinates(coordinate_text, img1.shape[0], img1.shape[1])
    
    if not boxes:
        return None
    
    # Görselleştirme oluştur
    overlay = img2.copy()
    
    for i, (pt1, pt2) in enumerate(boxes):
        # Renk palette (her box için farklı renk)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        color = colors[i % len(colors)]
        
        # Dikdörtgen çiz
        cv2.rectangle(overlay, pt1, pt2, color, 3)
        
        # Numaralama ekle
        cv2.putText(overlay, f"{i+1}", (pt1[0], pt1[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return overlay


def create_llm_heatmap(img1, img2, base_image):
    """Basit bir heatmap oluştur (LLM olmadan da kullanılabilir)"""
    # Farklı yöntemlerle heatmap oluştur
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Gaussian blur ile yumuşat
    diff_smooth = cv2.GaussianBlur(diff, (15, 15), 0)
    
    # Colormap uygula
    heatmap = cv2.applyColorMap(diff_smooth, cv2.COLORMAP_JET)
    
    # Base image ile karıştır
    result = cv2.addWeighted(base_image, 0.6, heatmap, 0.4, 0)
    
    return result


# --- DL Model Fonksiyonu ---
import torch
from models.model_zoo import get_model
from utils.palette import color_map

def normalize_for_dl(img):
    img = np.array(img).astype(np.float32) / 255.
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return img

@st.cache_resource
def load_dl_model():
    model = get_model('pspnet', 'hrnet_w40', False, 6, True)
    model.load_state_dict(torch.load('pspnet_hrnet_w40_39.37.pth', map_location='cpu'), strict=True)
    model.eval()
    return model

def dl_model_predict(img1, img2):
    model = load_dl_model()
    img1 = normalize_for_dl(img1).unsqueeze(0)
    img2 = normalize_for_dl(img2).unsqueeze(0)
    with torch.no_grad():
        out1, out2, out_bin = model(img1, img2, tta=True)
        out1 = torch.softmax(out1, dim=1)
        out1 = torch.argmax(out1, dim=1) + 1
        out1[out_bin > 0.5] = 0
        out1 = out1.squeeze().cpu().numpy().astype(np.uint8)
    return out1


st.title("🛰️ Satellite Change Detection - Enhanced Version")

# Sidebar - Parametre ayarları
st.sidebar.header("⚙️ Parametre Ayarları")

# Advanced hizalama seçeneği
use_enhanced_alignment = st.sidebar.checkbox("Gelişmiş Hizalama (ORB+FLANN)", value=True, help="ORB feature matching ve FLANN kullanarak daha iyi hizalama")

# SSIM parametreleri
if st.sidebar.checkbox("SSIM Parametrelerini Özelleştir"):
    ssim_threshold = st.sidebar.slider("SSIM Threshold", 0.1, 0.8, 0.3, 0.05)
    ssim_win_size = st.sidebar.selectbox("SSIM Window Size", [7, 11, 15], index=1)
    ssim_min_area = st.sidebar.number_input("SSIM Min Area", 50, 500, 100)
else:
    ssim_threshold = 0.3
    ssim_win_size = 11
    ssim_min_area = 100

# AbsDiff parametreleri
if st.sidebar.checkbox("AbsDiff Parametrelerini Özelleştir"):
    absdiff_threshold = st.sidebar.slider("AbsDiff Threshold (0=Otsu)", 0, 150, 60, 5)
    absdiff_min_area = st.sidebar.number_input("AbsDiff Min Area", 50, 500, 100)
else:
    absdiff_threshold = 60
    absdiff_min_area = 100

# Canny parametreleri
if st.sidebar.checkbox("Canny Parametrelerini Özelleştir"):
    canny_low = st.sidebar.slider("Canny Low Threshold", 10, 100, 50, 5)
    canny_high = st.sidebar.slider("Canny High Threshold", 100, 300, 150, 10)
    canny_min_area = st.sidebar.number_input("Canny Min Area", 25, 200, 50)
else:
    canny_low = 50
    canny_high = 150
    canny_min_area = 50

# Hibrit yöntem
enable_hybrid = st.sidebar.checkbox("Hibrit Yöntem Etkinleştir", value=True, help="Birden fazla yöntem seçildiğinde sonuçları birleştirir")

# Yöntem seçimi
st.header("🔍 Yöntem Seçimi")
methods = st.multiselect(
    "Değişiklik tespit yöntem(ler)i seçin", 
    ["SSIM", "AbsDiff", "Canny", "LLM (OpenAI)", "DL Model"],
    default=["SSIM", "AbsDiff"]
)

uploaded_before = st.file_uploader("📤 Upload BEFORE image", type=["png", "jpg", "jpeg"])
uploaded_after = st.file_uploader("📤 Upload AFTER image", type=["png", "jpg", "jpeg"])

# LLM seçiliyse API key iste
if "LLM (OpenAI)" in methods:
    api_key = st.text_input("🔑 OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        st.warning("⚠️ Lütfen bir OpenAI API anahtarı girin.")

if uploaded_before and uploaded_after and ("LLM (OpenAI)" not in methods or api_key):
    if st.button("🚀 Run Change Detection", type="primary"):
        with st.spinner("🔄 Processing images..."):
            img1 = load_image_from_upload(uploaded_before)
            img2 = load_image_from_upload(uploaded_after)
            img1, img2 = resize_images(img1, img2)
            
            # Gelişmiş hizalama kullan
            if use_enhanced_alignment:
                img2_aligned = align_images_enhanced(img1, img2)
            else:
                img2_aligned = align_images(img1, img2)
            
            base_image = img2_aligned

            output_masks = {}
            output_images = {}
            methods_results = {}
            llm_text = None
            llm_coordinates = None
            
            progress_bar = st.progress(0)
            total_methods = len(methods)
            
            for i, method in enumerate(methods):
                if method == "SSIM":
                    ssim_diff, ssim_mask, ssim_output, ssim_score = detect_changes_ssim(
                        img1, img2_aligned, base_image, threshold=ssim_threshold
                    )
                    output_masks["SSIM"] = ssim_mask
                    output_images["SSIM"] = ssim_output
                    methods_results["SSIM"] = {"mask": ssim_mask, "score": ssim_score}
                    
                elif method == "AbsDiff":
                    abs_diff, abs_mask, abs_output = detect_changes_absdiff(
                        img1, img2_aligned, base_image, threshold=absdiff_threshold
                    )
                    output_masks["AbsDiff"] = abs_mask
                    output_images["AbsDiff"] = abs_output
                    methods_results["AbsDiff"] = {"mask": abs_mask}
                    
                elif method == "Canny":
                    canny_diff, canny_mask, canny_output = detect_changes_canny(
                        img1, img2_aligned, base_image, low_thresh=canny_low, high_thresh=canny_high
                    )
                    output_masks["Canny"] = canny_mask
                    output_images["Canny"] = canny_output
                    methods_results["Canny"] = {"mask": canny_mask}
                    
                elif method == "LLM (OpenAI)":
                    llm_text, llm_coordinates = detect_changes_llm(img1, img2_aligned, api_key)
                    
                    # LLM görselleştirmesi oluştur
                    llm_visualization = create_llm_visualization(img1, img2_aligned, llm_coordinates)
                    if llm_visualization is None:
                        # Koordinat bulunamazsa heatmap oluştur
                        llm_visualization = create_llm_heatmap(img1, img2_aligned, img2_aligned)
                    
                    output_masks["LLM (OpenAI)"] = None
                    output_images["LLM (OpenAI)"] = llm_visualization
                    
                elif method == "DL Model":
                    # DL Model tahmini
                    pil_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                    pil_img2 = Image.fromarray(cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB))
                    mask = dl_model_predict(pil_img1, pil_img2)
                    cmap = color_map()
                    mask_img = Image.fromarray(mask, mode="P")
                    mask_img.putpalette(cmap)
                    output_masks["DL Model"] = mask
                    output_images["DL Model"] = mask_img
                    methods_results["DL Model"] = {"mask": mask}
                
                progress_bar.progress((i + 1) / total_methods)
            
            # Hibrit yöntem
            hybrid_mask, hybrid_output = None, None
            if enable_hybrid and len([m for m in methods if m not in ["LLM (OpenAI)", "DL Model"]]) >= 2:
                cv_methods_results = {k: v for k, v in methods_results.items() if k not in ["LLM (OpenAI)", "DL Model"]}
                hybrid_mask, hybrid_output = combine_methods(cv_methods_results, base_image)

        st.success("✅ Change detection completed.")

        # Görselleştirme: LLM, SSIM, AbsDiff, Canny sırası ile
        method_order = ["LLM (OpenAI)", "SSIM", "AbsDiff", "Canny", "DL Model"]
        
        # Hibrit sonucu göster
        if hybrid_mask is not None:
            st.markdown("### 🔀 Hibrit Yöntem Sonucu")
            st.info("📊 Birden fazla yöntemin hemfikir olduğu değişiklikler")
            cols = st.columns(3)
            with cols[0]:
                st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="BEFORE")
            with cols[1]:
                st.image(cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB), caption="AFTER")
            with cols[2]:
                st.image(cv2.cvtColor(hybrid_output, cv2.COLOR_BGR2RGB), caption="Hibrit Değişim")
        
        for method in method_order:
            if method in methods:
                st.markdown(f"### {method} Sonucu")
                
                # SSIM için ekstra bilgi
                if method == "SSIM" and "SSIM" in methods_results:
                    st.info(f"📈 SSIM Score: {methods_results['SSIM']['score']:.4f} (1.0 = identical)")
                
                cols = st.columns(3)
                with cols[0]:
                    st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="BEFORE")
                with cols[1]:
                    st.image(cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB), caption="AFTER")
                with cols[2]:
                    if method == "LLM (OpenAI)":
                        # LLM görselleştirmesini göster
                        if output_images.get("LLM (OpenAI)") is not None:
                            st.image(cv2.cvtColor(output_images["LLM (OpenAI)"], cv2.COLOR_BGR2RGB), 
                                   caption="LLM Tabanlı Değişim Gösterimi")
                    elif method == "DL Model":
                        st.image(output_images[method], caption="DL Model Mask")
                    else:
                        st.image(cv2.cvtColor(output_images[method], cv2.COLOR_BGR2RGB), caption="Değişim")
                
                # LLM için text sonuçları görsellerin altında göster
                if method == "LLM (OpenAI)":
                    st.markdown("**🤖 LLM Açıklaması:**")
                    st.write(llm_text)
                    if llm_coordinates:
                        st.markdown("**📍 Tespit Edilen Koordinatlar:**")
                        st.text(llm_coordinates)
                
                # Mask istatistikleri
                if method in output_masks and output_masks[method] is not None:
                    mask = output_masks[method]
                    changed_pixels = np.sum(mask > 127)
                    total_pixels = mask.shape[0] * mask.shape[1]
                    change_percentage = (changed_pixels / total_pixels) * 100
                    st.metric(f"{method} Değişim Yüzdesi", f"{change_percentage:.2f}%")

        # Download butonu ekle
        if len(output_images) > 0:
            st.markdown("### 💾 Sonuçları İndir")
            st.info("💡 Gelecek güncellemede sonuçları indirme özelliği eklenecek")