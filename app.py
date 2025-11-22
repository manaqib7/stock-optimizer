import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image, ImageOps 
import io
import time
import json
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions 

# ==============================================================================
# ⚙️ 1. CONFIGURATION & JSON SCHEMA
# ==============================================================================

st.set_page_config(
    page_title="Stock Genius CSV", # Updated Page Title
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adobe Stock Category ID Mapping
ADOBE_CATEGORIES = {
    "Animals": 1, "Buildings and Architecture": 2, "Business": 3,
    "Drinks": 4, "The Environment": 5, "States of Mind": 6,
    "Food": 7, "Graphic Resources": 8, "Hobbies and Leisure": 9,
    "Industry": 10, "Landscape": 11, "Lifestyle": 12,
    "People": 13, "Plants and Flowers": 14, "Culture and Religion": 15,
    "Science": 16, "Social Issues": 17, "Sports": 18,
    "Technology": 19, "Transport": 20, "Travel": 21
}

# Model list for auto-selection (Prioritized for stability)
MODEL_ARSENAL = [
    "gemini-2.5-flash", 
    "gemini-2.5-pro",   
    "gemini-1.5-flash", 
    "gemini-1.5-pro"    
]

# JSON Schema (MANDATORY for 100% valid JSON output)
STOCK_METADATA_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING", "description": "The descriptive title of the image."},
        "keywords": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "A list of relevant keywords."},
        "category_name": {"type": "STRING", "description": "The exact name of the best category from the Adobe list."},
        "commercial_score": {"type": "INTEGER", "description": "Commercial score from 1 (low) to 10 (high)."},
        "has_trademark": {"type": "BOOLEAN", "description": "True if visible logos, brands, or protected entities are present."}
    },
    "required": ["title", "keywords", "category_name", "commercial_score", "has_trademark"]
}

# Advanced SEO Prompt
SYSTEM_PROMPT_TEMPLATE = """
You are an expert stock photography metadata specialist for Adobe Stock.
Analyze the image content, lighting, style, and commercial appeal to generate optimal metadata in the STRICT JSON format provided in the schema.

GUIDELINES:
1. TITLE: Write a highly descriptive title ({t_min}-{t_max} chars). Capitalize the first letter of each major word (Title Case). Do not include vague phrases like "Image of" or "Photo of".
2. KEYWORDS: Generate exactly {kw_count} keywords. Sort them strictly by relevance (most important commercial tag first). Include niche and long-tail tags.
3. CATEGORY: Choose the single BEST category from the provided list.
4. COMMERCIAL: Provide a score (1-10) based on technical quality, market demand, and commercial appeal.
5. TRADEMARK: Set 'true' if any visible logos, brands, or protected entities are present, otherwise 'false'.
"""

# ==============================================================================
# 🎨 2. UI STYLING (Colourful, Attractive, Professional)
# ==============================================================================

st.markdown("""
    <style>
    /* Main Dark Theme */
    .stApp { background-color: #0d1117; color: #E6EDF3; }
    
    /* Custom Title Style */
    h1 {
        color: #00E6E6; /* Bright Cyan Accent */
        text-shadow: 0 0 10px rgba(0, 230, 230, 0.5); 
        font-weight: 900;
        margin-bottom: 20px;
    }

    /* Result card styling */
    .result-card {
        background-color: #161b22; 
        border: 1px solid #30363d; 
        border-radius: 12px;
        padding: 16px; 
        margin-bottom: 16px; 
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }
    .result-card:hover { 
        border-color: #00E6E6; 
        box-shadow: 0 0 15px rgba(0, 230, 230, 0.4); 
        transform: translateY(-2px);
    }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 8px; }
    
    /* Keyword tags */
    .tag-pill {
        display: inline-block; 
        padding: 3px 10px; 
        background-color: rgba(0, 230, 230, 0.15); /* More vibrant background */
        border: 1px solid rgba(0, 230, 230, 0.4); 
        border-radius: 4px; /* Square tags look more professional */
        color: #00E6E6;
        font-size: 0.75rem; 
        font-weight: 500; 
        margin: 3px 2px;
        line-height: 1.2;
    }
    /* Badges */
    .badge { padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
    .badge-tm-warn { background: rgba(248, 81, 73, 0.2); color: #F85149; border: 1px solid #F85149; }
    .badge-tm-safe { background: rgba(63, 185, 80, 0.2); color: #3FB950; border: 1px solid #3FB950; }
    
    /* Button style - Primary Button with Glow */
    div.stButton > button { 
        border-radius: 8px; 
        font-weight: 700; 
        height: 45px;
        transition: all 0.2s;
        border: 1px solid #00E6E6; 
    }
    div.stButton > button:hover {
        box-shadow: 0 0 10px rgba(0, 230, 230, 0.6);
    }

    /* Stop Button Styling */
    .stButton button[kind="secondary"] {
        background-color: #f85149;
        color: white;
        border: 1px solid #f85149;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #d73a49;
        border: 1px solid #d73a49;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 3. LOGIC ENGINE
# ==============================================================================

class StockAI_Engine:
    @staticmethod
    def check_api_key(api_key):
        """Validates the key and finds the best model."""
        genai.configure(api_key=api_key)
        for model in MODEL_ARSENAL:
            try:
                m = genai.GenerativeModel(model)
                m.generate_content("API key check")
                return {"status": "valid", "model": model}
            except google_exceptions.NotFound as e:
                st.toast(f"⚠️ Model {model} is not found or supported for your key. Trying next model...", icon="⏳")
                continue
            except Exception as e:
                if "API_KEY_INVALID" in str(e):
                    return {"status": "invalid", "model": None}
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.toast(f"⚠️ Quota limit hit for {model}. Trying next available model.", icon="⏳")
                    continue
                continue
        return {"status": "valid", "model": "gemini-2.5-flash"}

    @staticmethod
    def analyze_image_type(image):
        """Analyzes image to detect transparency/isolation."""
        is_transparent = False
        if image.mode in ('RGBA', 'LA'):
            alpha = image.split()[-1]
            if alpha.getextrema()[0] == 0:
                is_transparent = True
        return 'Transparent/Isolated Vector or PNG' if is_transparent else 'High-Quality Photo or JPEG'

    @staticmethod
    def optimize_image(image):
        """Resizes the image for efficient API transfer and creates a thumbnail."""
        thumb = image.copy()
        thumb.thumbnail((150, 150))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG" if thumb.mode=='RGBA' else "JPEG", quality=60)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        img_ai = image.copy()
        img_ai.thumbnail((1024, 1024))
        return img_ai, thumb_b64

    @staticmethod
    def process_image(file_obj, api_key, rules, model_name):
        """Processes a single image using the Native JSON Schema."""
        # Check stop flag before processing
        if st.session_state.get('stop_flag', False):
            return {"status": "skipped", "filename": file_obj.name, "thumbnail": None, "msg": "Processing stopped by user."}

        try:
            img = Image.open(file_obj)
            img_ai, thumb_b64 = StockAI_Engine.optimize_image(img)
            
            image_context = StockAI_Engine.analyze_image_type(img)
            
            prompt = SYSTEM_PROMPT_TEMPLATE.format(
                t_min=rules['t_min'], 
                t_max=rules['t_max'], 
                kw_count=rules['kw_count']
            )
            
            full_prompt = f"Image Context: {image_context}\n\n{prompt}"
            
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel(
                model_name, 
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": STOCK_METADATA_SCHEMA,
                    "temperature": 0.3 
                }
            )
            
            resp = model.generate_content([full_prompt, img_ai])
            
            data = json.loads(resp.text)
            
            keywords = [k for k in data.get("keywords", []) if k.strip()]
            title = data.get("title", "").strip()

            if not title or not keywords:
                 raise ValueError("AI returned insufficient metadata (title or keywords missing).")

            return {
                "status": "success", "filename": file_obj.name, "thumbnail": thumb_b64,
                "title": title, "keywords": keywords,
                "category_name": data.get("category_name", "Unknown"),
                "category_id": ADOBE_CATEGORIES.get(data.get("category_name", "Unknown"), 1),
                "score": data.get("commercial_score", 5), "trademark": data.get("has_trademark", False)
            }
            
        except google_exceptions.NotFound as e:
            return {"status": "error", "filename": file_obj.name, "thumbnail": thumb_b64 if 'thumb_b64' in locals() else None, "msg": f"Model Not Found (404). Please check the API key scope."}
        except Exception as e:
            return {"status": "error", "filename": file_obj.name, "thumbnail": thumb_b64 if 'thumb_b64' in locals() else None, "msg": f"AI/Processing Failed: {str(e)}"}

# ==============================================================================
# 🚀 4. MAIN APP
# ==============================================================================

def render_card(res):
    """Renders the result card for a single image."""
    if res['status'] == 'success':
        tm_badge = '⚠️ LOGO/TM' if res['trademark'] else '✅ SAFE'
        # Increased keyword display to 20 tags
        tags = "".join([f'<span class="tag-pill">{k}</span>' for k in res['keywords'][:20]]) 
        
        tm_class = 'warn' if res['trademark'] else 'safe'

        st.markdown(f"""
        <div class="result-card">
            <div style="display:flex; gap:15px;">
                <img src="data:image/jpeg;base64,{res['thumbnail']}" style="width:90px; height:90px; border-radius:8px; object-fit:cover;">
                <div style="flex-grow:1;">
                    <div style="display:flex; justify-content:space-between;">
                        <div class="card-title">{res['filename']}</div>
                        <div class="badge badge-tm-{tm_class}">{tm_badge}</div>
                    </div>
                    <div style="font-size:0.9em; color:#ccc; margin-bottom:5px;">{res['title']}</div>
                    <div style="display:flex; flex-wrap:wrap;">{tags}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif res['status'] == 'error':
        st.error(f"❌ {res['filename']}: {res['msg']}")
    elif res['status'] == 'skipped':
         st.warning(f"⏭️ {res['filename']}: {res['msg']}")


def main():
    # Session state setup
    if 'results' not in st.session_state: st.session_state.results = []
    if 'processing' not in st.session_state: st.session_state.processing = False
    if 'key_status' not in st.session_state: st.session_state.key_status = {"status": "none", "model": None}
    if 'stop_flag' not in st.session_state: st.session_state.stop_flag = False

    workers = 2 

    with st.sidebar:
        st.title("⚙️ Settings")
        
        api_key = st.text_input("Gemini API Key", type="password", key="api_input")
        
        if api_key and st.session_state.key_status["status"] == "none":
            st.session_state.key_status = StockAI_Engine.check_api_key(api_key)
            
        if st.session_state.key_status["status"] == "valid":
            st.sidebar.success(f"🔑 Key Valid. Using Model: **{st.session_state.key_status['model']}**")
        elif st.session_state.key_status["status"] == "invalid":
            st.sidebar.error("❌ Invalid API Key. Please check the key.")
        
        st.markdown("---")
        st.info(f"⚡ **Processing Mode:** Concurrent ({workers} threads) for speed, with rate limit control.")

        with st.expander("Target Rules"):
            t_range = st.slider("Title Length (Chars)", 40, 200, (60, 80))
            kw_count = st.number_input("Keywords Count", 5, 49, 49)

    # Renamed Headline as requested
    st.title("Stock Genius Csv") 
    uploaded_files = st.file_uploader("Upload Images (JPG, PNG)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

    key_invalid = st.session_state.key_status["status"] == "invalid"
    start_btn_disabled = not uploaded_files or not api_key or key_invalid or st.session_state.processing
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    start_btn = col1.button("🚀 START BATCH", disabled=start_btn_disabled, type="primary")
    
    if st.session_state.processing:
        # Show STOP button only during processing
        if col2.button("🛑 STOP PROCESSING", type="secondary"):
            st.session_state.stop_flag = True
            st.warning("Stopping process. Waiting for current threads to finish...")
            
    else:
        # Show CLEAR button only when not processing
        if col2.button("🗑️ Clear Results"):
            st.session_state.results = []
            st.session_state.processing = False
            st.session_state.key_status = {"status": "none", "model": None}
            st.session_state.stop_flag = False
            st.rerun()


    if start_btn:
        st.session_state.results = []
        st.session_state.processing = True
        st.session_state.stop_flag = False # Reset stop flag
        
        active_model = st.session_state.key_status["model"]
        
        rules = {'t_min': t_range[0], 't_max': t_range[1], 'kw_count': kw_count}
        total = len(uploaded_files)
        prog_bar = st.progress(0)
        res_area = st.container()
        
        st.info(f"Starting {total} files using **{active_model}** with **{workers}** concurrent threads...")

        # --- CONCURRENT LOOP ---
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(StockAI_Engine.process_image, f, api_key, rules, active_model): f for f in uploaded_files}
            
            for i, future in enumerate(as_completed(futures)):
                if st.session_state.stop_flag:
                    # Cancel pending tasks if stop flag is set
                    for f in futures:
                        f.cancel()
                    st.toast("Batch stopped by user.", icon="🛑")
                    break 

                res = future.result()
                st.session_state.results.append(res)
                
                # Update UI
                prog_bar.progress((i+1)/total)
                
                # Render results live during processing
                with res_area:
                    # Clear previous results container to prevent continuous growth
                    res_area.empty()
                    # Re-render all results collected so far
                    for r in st.session_state.results:
                        render_card(r)

        st.session_state.processing = False
        st.session_state.stop_flag = False # Final reset
        prog_bar.empty()
        st.success("✅ Batch Processing Complete!" if not st.session_state.stop_flag else "🛑 Processing Stopped.")


    # --- Results and CSV Export Section (PERSISTENTLY DISPLAYED) ---
    if st.session_state.results:
        # 1. Clear and re-render the final results area after processing is done
        st.markdown("---")
        st.subheader("🖼️ Processed Results")
        
        # Display the final list of results persistently
        for r in st.session_state.results:
            render_card(r)
            
        # 2. CSV Export Logic
        success = [r for r in st.session_state.results if r['status'] == 'success']
        if success:
            st.markdown("---")
            st.subheader("⬇️ Download CSV")
            
            df = pd.DataFrame(success)
            df['keywords_str'] = df['keywords'].apply(lambda x: ", ".join(x))
            
            # Adobe Stock CSV Format
            csv_df = pd.DataFrame({
                "Filename": df['filename'], "Title": df['title'], 
                "Keywords": df['keywords_str'], "Category": df['category_id'], "Releases": ""
            })
            
            csv = csv_df.to_csv(index=False).encode('utf-8')
            
            st.download_button("⬇️ Download Adobe Stock CSV", csv, "Adobe_Ready.csv", "text/csv", type="primary")

if __name__ == "__main__":
    main()