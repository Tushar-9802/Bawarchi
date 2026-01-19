"""
Bawarchi - AI-Powered Recipe Generation System

Two-Phase Substitution System:
- Phase 1 (Planning): Pre-generation ingredient swapping in Preparation tab
- Phase 2 (Adaptation): Post-generation alternatives in Recipe tab

Role-aware recipe generation with cuisine-context switching
"""

import streamlit as st
from pathlib import Path
import torch
from PIL import Image
import sys
import warnings

warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="Bawarchi",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Bawarchi - AI Recipe Generation"
    }
)

# Force light theme
st.markdown("""
<style>
    /* Force light theme globally */
    :root {
        color-scheme: light;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main {
        padding: 2rem;
        background-color: #ffffff !important;
    }
    
    /* All text dark */
    * {
        color: #1f1f1f !important;
    }
    
    h1 { font-size: 2.5rem; font-weight: 600; margin-bottom: 0.5rem; }
    h2 { font-size: 1.75rem; font-weight: 500; margin-top: 2rem; margin-bottom: 1rem; 
         border-bottom: 2px solid #d0d0d0; padding-bottom: 0.5rem; }
    h3 { font-size: 1.25rem; font-weight: 500; margin-top: 1.5rem; margin-bottom: 0.75rem; }
    
    .recipe-container {
        background: #f9f9f9 !important;
        padding: 1.5rem;
        border-radius: 6px;
        border: 1px solid #d0d0d0;
        margin-top: 1rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 4px !important;
        font-weight: 500 !important;
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border: 1px solid #d0d0d0 !important;
    }
    
    .stButton > button:hover {
        background-color: #f5f5f5 !important;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #28a745 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #218838 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #28a745 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #218838 !important;
    }
    
    /* Selectbox - complete override */
    .stSelectbox, .stSelectbox * {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d0d0d0 !important;
    }
    
    /* Dropdown menu */
    [role="listbox"], [role="listbox"] * {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    
    [role="option"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    
    [role="option"]:hover {
        background-color: #f5f5f5 !important;
    }
    
    /* Text inputs */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border: 1px solid #d0d0d0 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    /* Alerts */
    .stAlert {
        background-color: #e9ecef !important;
        color: #1f1f1f !important;
        border: 1px solid #d0d0d0 !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #1f1f1f !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detection_model():
    try:
        from ultralytics import YOLO
        import os
        
        # Show current directory for debugging
        cwd = os.getcwd()
        st.info(f"Working directory: {cwd}")
        
        # Try custom trained model first
        possible_paths = [
            Path("models/yolov8m_best.pt"),
            Path("models/detection/production/yolov8m_best.pt"),
            Path("models/detection/production/best.pt"),
            Path("runs/detect/train/weights/best.pt"),
            Path("yolov8m_best.pt"),
        ]
        
        # Check which files actually exist
        existing = [p for p in possible_paths if p.exists()]
        
        if existing:
            model_path = existing[0]
            st.success(f"✓ Found model: {model_path}")
            model = YOLO(str(model_path))
            st.success(f"✓ Loaded {len(model.names)} classes: {list(model.names.values())[:5]}...")
            return model
        else:
            st.warning(f"Custom model not found in: {[str(p) for p in possible_paths]}")
            st.info("Using pretrained YOLOv8n (generic 80-class COCO model)")
            model = YOLO("yolov8n.pt")
            st.info(f"Loaded generic model with {len(model.names)} classes")
            return model
            
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


@st.cache_resource
def load_recipe_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        base_model = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Try role-aware adapter first, fallback to final
        adapter_paths = [
            Path("models/bawarchi-adapter/role_aware"),
            Path("models/bawarchi-adapter/final")
        ]
        
        adapter_path = None
        for p in adapter_paths:
            if p.exists():
                adapter_path = p
                break
        
        if adapter_path is None:
            st.error("No recipe adapter found")
            return None, None
        
        adapter_type = "role-aware" if "role_aware" in str(adapter_path) else "base"
        st.info(f"Loading {adapter_type} recipe model...")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        st.success(f"✓ {adapter_type.title()} model loaded")
        return model, tokenizer
    except Exception as e:
        st.error(f"Recipe model error: {str(e)}")
        return None, None


@st.cache_resource
def load_substitution_system():
    try:
        from scripts.substitution_learning.substitution_ranker import SubstitutionRanker
        
        paths = {
            'yaml': Path("data/merged/data.yaml"),
            'embeddings': Path("data/embeddings/semantic_embeddings.pt"),
            'pmi': Path("data/substitution/pmi_matrix.npz"),
            'categories': Path("data/substitution/ingredient_categories.json")
        }
        
        # Check all paths exist
        missing = [k for k, v in paths.items() if not v.exists()]
        if missing:
            st.warning(f"Substitution files missing: {missing}")
            return None, []
        
        ranker = SubstitutionRanker(
            data_yaml_path=paths['yaml'],
            embeddings_path=paths['embeddings'],
            pmi_matrix_path=paths['pmi'],
            categories_path=paths['categories']
        )
        
        st.success(f"Substitution system loaded: {len(ranker.detection_classes)} ingredients")
        return ranker, ranker.detection_classes
    except Exception as e:
        st.warning(f"Substitution system error: {str(e)}")
        return None, []


def init_state():
    if 'detections' not in st.session_state:
        st.session_state.detections = []
    if 'selected_ingredients' not in st.session_state:
        st.session_state.selected_ingredients = []
    if 'recipe' not in st.session_state:
        st.session_state.recipe = None


def main():
    init_state()
    
    st.title("Bawarchi")
    st.markdown("**AI-Powered Recipe Generation**")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("### Configuration")
        confidence = st.slider("Detection Confidence", 0.10, 0.95, 0.25, 0.05)
        
        st.markdown("---")
        st.markdown("### Models")
        st.markdown("""
        **Detection:** YOLOv8m (66.51%)  
        **Substitution:** PMI+Embeddings (85%)
        - Planning phase (prep)
        - Adaptation phase (recipe)
        
        **Generation:** Llama 3.2 3B
        - Role-aware reasoning
        - Cuisine-context switching
        """)
    
    tab1, tab2 = st.tabs(["Preparation", "Recipe Generation"])
    
    with tab1:
        preparation_tab(confidence)
    
    with tab2:
        recipe_tab()


def preparation_tab(confidence):
    st.header("Ingredient Preparation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Add Ingredients")
        
        method = st.selectbox("Input method:", ["Upload Image", "Use Camera", "Manual Entry"])
        
        if method == "Upload Image":
            file = st.file_uploader("Upload photo", type=['jpg', 'png', 'jpeg'])
            
            if file:
                img = Image.open(file)
                st.image(img, use_container_width=True)
                
                if st.button("Detect", type="primary"):
                    with st.spinner("Detecting..."):
                        detect_ingredients(img, confidence)
        
        elif method == "Use Camera":
            cam = st.camera_input("Capture")
            
            if cam:
                img = Image.open(cam)
                if st.button("Detect", type="primary"):
                    with st.spinner("Detecting..."):
                        detect_ingredients(img, confidence)
        
        else:
            with st.form("manual"):
                ing = st.text_input("Ingredient name:", placeholder="e.g., chicken, tomato, onion")
                st.caption("💡 Tip: Separate multiple ingredients with commas")
                if st.form_submit_button("Add"):
                    if ing.strip():
                        # Split on commas and add each ingredient separately
                        ingredients_to_add = [i.strip().lower() for i in ing.split(',')]
                        
                        added_count = 0
                        for clean in ingredients_to_add:
                            if clean and clean not in st.session_state.selected_ingredients:
                                st.session_state.selected_ingredients.append(clean)
                                added_count += 1
                        
                        if added_count > 0:
                            st.success(f"Added {added_count} ingredient(s)")
                            st.rerun()
        
        if st.session_state.detections:
            st.markdown("---")
            st.markdown("**Detections:**")
            
            for det in st.session_state.detections:
                c1, c2, c3 = st.columns([0.6, 0.25, 0.15])
                with c1:
                    st.text(det['name'].replace('_', ' ').title())
                with c2:
                    st.text(f"{det['confidence']:.0%}")
                with c3:
                    if st.button("Add", key=f"add_{det['name']}"):
                        if det['name'] not in st.session_state.selected_ingredients:
                            st.session_state.selected_ingredients.append(det['name'])
                            st.rerun()
    
    with col2:
        st.subheader("Selected Ingredients")
        
        if st.session_state.selected_ingredients:
            # Display selected with inline substitutions
            for i, ing in enumerate(st.session_state.selected_ingredients):
                col_a, col_b = st.columns([0.75, 0.25])
                with col_a:
                    st.text(f"{i+1}. {ing.replace('_', ' ').title()}")
                with col_b:
                    if st.button("Remove", key=f"rm_{i}"):
                        st.session_state.selected_ingredients.pop(i)
                        st.rerun()
                
                # Phase 1: Planning substitutions (inline, per ingredient)
                show_planning_substitutions(ing, i)
            
        else:
            st.info("No ingredients selected")


def show_planning_substitutions(ingredient, index):
    """
    Phase 1: Planning substitutions
    Shows inline under each ingredient with [Use] buttons
    """
    result = load_substitution_system()
    
    if not result or result[0] is None:
        return
    
    ranker, available = result
    
    # Match ingredient to available classes
    matched = None
    if ingredient in available:
        matched = ingredient
    elif ingredient.replace(' ', '_') in available:
        matched = ingredient.replace(' ', '_')
    elif ingredient.replace(' ', '_').title() in available:
        matched = ingredient.replace(' ', '_').title()
    else:
        for av in available:
            if av.lower() == ingredient.lower():
                matched = av
                break
    
    if not matched:
        return
    
    try:
        subs = ranker.get_substitutions(matched, top_k=3, use_category_filter=True)
        
        if subs and len(subs) > 0:
            with st.expander(f"ℹ️ Don't have {ingredient.replace('_', ' ').title()}?", expanded=False):
                st.markdown("**Alternative ingredients:**")
                for sub_name, score in subs:
                    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
                    with c1:
                        st.text(sub_name.replace('_', ' ').title())
                    with c2:
                        st.text(f"Match: {score:.2f}")
                    with c3:
                        if st.button("Use", key=f"use_{index}_{sub_name}"):
                            st.session_state.selected_ingredients[index] = sub_name.lower()
                            st.success(f"Swapped to {sub_name.replace('_', ' ').title()}")
                            st.rerun()
    except Exception as e:
        pass


def show_recipe_substitutions(ingredients):
    """
    Phase 2: Recipe adaptation substitutions
    Shows comprehensive list after recipe generation
    """
    result = load_substitution_system()
    
    if not result or result[0] is None:
        return
    
    ranker, available = result
    
    # CRITICAL FIX: Split comma-separated ingredients
    split_ingredients = []
    for ing in ingredients:
        if ',' in ing:
            # Split and clean
            split_ingredients.extend([i.strip() for i in ing.split(',')])
        else:
            split_ingredients.append(ing)
    
    ingredients = split_ingredients
    
    st.markdown("---")
    st.markdown("### 📝 Ingredient Alternatives for This Recipe")
    st.markdown("*Missing an ingredient while cooking? Here are tested substitutions:*")
    st.markdown("")
    
    # DEBUG INFO
    with st.expander("🔍 Debug Info (click to see why substitutions might not appear)", expanded=False):
        st.write(f"**Your ingredients:** {ingredients}")
        st.write(f"**Available classes in system:** {len(available)} ingredients")
        st.write(f"**Sample classes:** {available[:15]}")
        
        # Show matching attempts
        st.markdown("**Matching attempts:**")
        for ing in ingredients:
            attempts = []
            attempts.append(f"Exact: '{ing}' → {ing in available}")
            attempts.append(f"Underscore: '{ing.replace(' ', '_')}' → {ing.replace(' ', '_') in available}")
            attempts.append(f"Title: '{ing.replace(' ', '_').title()}' → {ing.replace(' ', '_').title() in available}")
            
            st.text(f"{ing}:")
            for attempt in attempts:
                st.text(f"  {attempt}")
    
    has_substitutions = False
    
    for ing in ingredients:
        # Match ingredient
        matched = None
        if ing in available:
            matched = ing
        elif ing.replace(' ', '_') in available:
            matched = ing.replace(' ', '_')
        elif ing.replace(' ', '_').title() in available:
            matched = ing.replace(' ', '_').title()
        else:
            for av in available:
                if av.lower() == ing.lower():
                    matched = av
                    break
        
        if matched:
            try:
                subs = ranker.get_substitutions(matched, top_k=3, use_category_filter=True)
                
                if subs and len(subs) > 0:
                    has_substitutions = True
                    
                    # Display ingredient header
                    st.markdown(f"**{ing.replace('_', ' ').title()}**")
                    
                    # Show substitutions as bullet points
                    for sub_name, score in subs:
                        display_name = sub_name.replace('_', ' ').title()
                        st.markdown(f"→ {display_name} (match: {score:.2f})")
                    
                    st.markdown("")  # spacing
                else:
                    st.caption(f"ℹ️ {ing.title()}: No substitutions found in database")
            except Exception as e:
                st.error(f"Error for {ing}: {str(e)}")
        else:
            st.caption(f"⚠️ {ing.title()}: Not in ingredient database")
    
    if not has_substitutions:
        st.info("No common substitutions available for these ingredients")



def recipe_tab():
    st.header("Recipe Generation")
    
    if not st.session_state.selected_ingredients:
        st.warning("Add ingredients in Preparation tab first")
        return
    
    st.markdown("**Using:**")
    st.write(", ".join([i.replace('_', ' ').title() for i in st.session_state.selected_ingredients]))
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        cuisine = st.selectbox("Cuisine", ["Indian", "Mexican", "Italian", "Asian", "Fusion", "General"])
    
    with c2:
        difficulty = st.select_slider("Difficulty", ["Easy", "Medium", "Hard"])
    
    st.markdown("")
    
    if st.button("Generate Recipe", type="primary", use_container_width=True):
        with st.spinner("Generating (20-30s)..."):
            generate_recipe(st.session_state.selected_ingredients, cuisine.lower(), difficulty.lower())
    
    if st.session_state.recipe:
        display_recipe(st.session_state.recipe)


def detect_ingredients(image, conf):
    model = load_detection_model()
    if not model:
        st.error("Model not loaded")
        return
    
    # Show model info
    st.info(f"Model: {model.model.yaml.get('nc', 'unknown')} classes")
    st.info(f"Classes: {list(model.names.values())[:10]}... (showing first 10)")
    
    try:
        results = model(image, conf=conf)
        dets = []
        
        for r in results:
            for box in r.boxes:
                dets.append({
                    'name': model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0])
                })
        
        st.session_state.detections = dets
        
        if len(dets) == 0:
            st.warning(f"No detections above {conf:.0%} confidence. Try lowering the threshold.")
        else:
            st.success(f"Detected {len(dets)} ingredients")
        
        st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")


def generate_recipe(ingredients, cuisine, difficulty):
    model, tokenizer = load_recipe_model()
    if not model:
        st.error("Model not loaded")
        return
    
    try:
        ing_fmt = [i.replace('_', ' ') for i in ingredients]
        ing_str = ", ".join(ing_fmt)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Create a {difficulty}-level {cuisine} recipe using: {ing_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        recipe = response.split("assistant")[-1].strip() if "assistant" in response else response
        
        st.session_state.recipe = recipe
        st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")


def display_recipe(text):
    st.markdown("---")
    st.subheader("Generated Recipe")
    
    # Clean up the text - convert markdown to HTML properly
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Convert markdown bold (**text**) to HTML bold
        line = line.replace('**', '')
        
        # Handle bullet points (lines starting with *)
        if line.startswith('* '):
            line = '• ' + line[2:]
        
        # Make section headers bold
        if any(h in line.lower() for h in ['ingredients:', 'instructions:', 'servings:', 'cook time:', 'chutney:', 'masala:']):
            formatted_lines.append(f"<strong>{line}</strong>")
        # Make numbered steps bold (just the number)
        elif line and line[0].isdigit() and '.' in line[:3]:
            parts = line.split('.', 1)
            if len(parts) == 2:
                formatted_lines.append(f"<strong>{parts[0]}.</strong>{parts[1]}")
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    formatted = '<br>'.join(formatted_lines)
    
    st.markdown(f'<div class="recipe-container">{formatted}</div>', unsafe_allow_html=True)
    st.markdown("")
    
    # Phase 2: Recipe adaptation substitutions
    show_recipe_substitutions(st.session_state.selected_ingredients)
    
    st.markdown("")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("Regenerate", use_container_width=True):
            st.session_state.recipe = None
            st.rerun()
    
    with c2:
        if st.button("Clear All", use_container_width=True):
            st.session_state.recipe = None
            st.session_state.selected_ingredients = []
            st.session_state.detections = []
            st.rerun()
    
    with c3:
        st.download_button("Download", text, "recipe.txt", "text/plain", use_container_width=True)


if __name__ == "__main__":
    main()