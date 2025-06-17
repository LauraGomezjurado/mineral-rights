import streamlit as st
import os
import tempfile
from document_classifier import DocumentProcessor

# Page config
st.set_page_config(
    page_title="Mineral Rights Analyzer",
    page_icon="⛏️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E4057;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .no-reservation {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .has-reservation {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.processing_error = None

def initialize_processor():
    """Initialize the document processor"""
    try:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            st.error("❌ ANTHROPIC_API_KEY environment variable not set")
            st.info("Please set your Anthropic API key in the Streamlit Cloud settings.")
            return False
        
        st.session_state.processor = DocumentProcessor(api_key=api_key)
        return True
    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"❌ Failed to initialize: {e}")
        return False

def process_document(uploaded_file):
    """Process the uploaded document"""
    tmp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process document
        with st.spinner('🔍 Analyzing document...'):
            result = st.session_state.processor.process_document(
                tmp_path,
                max_samples=5,
                confidence_threshold=0.7
            )
        
        return result
        
    except Exception as e:
        raise e
    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignore cleanup errors

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">⛏️ Mineral Rights Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Upload a PDF document to analyze mineral rights reservations</p>', unsafe_allow_html=True)
    
    # Initialize processor if not already done
    if st.session_state.processor is None and st.session_state.processing_error is None:
        with st.spinner('🔧 Initializing document processor...'):
            initialize_processor()
    
    # Show initialization error if any
    if st.session_state.processing_error:
        st.error(f"❌ Initialization failed: {st.session_state.processing_error}")
        if st.button("🔄 Retry Initialization"):
            st.session_state.processing_error = None
            st.session_state.processor = None
            st.rerun()
        return
    
    # Check if processor is ready
    if st.session_state.processor is None:
        st.warning("⏳ Initializing processor...")
        return
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📄 Choose a PDF file",
        type=['pdf'],
        help="Upload a legal document to analyze for mineral rights reservations"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Show file info
        st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Process button
        if st.button("🔍 Analyze Document", type="primary"):
            try:
                result = process_document(uploaded_file)
                
                # Display results
                st.markdown("## 📊 Analysis Results")
                
                # Main classification result
                classification = result['classification']
                confidence = result['confidence']
                
                if classification == 0:
                    st.markdown(f'''
                    <div class="result-box no-reservation">
                        <h3>✅ NO MINERAL RIGHTS RESERVATIONS DETECTED</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p>This document appears to be a clean transfer without mineral rights reservations.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="result-box has-reservation">
                        <h3>⚠️ MINERAL RIGHTS RESERVATIONS DETECTED</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p>This document contains language that may reserve mineral rights.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Additional details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Analysis Details")
                    st.write(f"**Samples analyzed:** {result['samples_used']}")
                    st.write(f"**Pages processed:** {result['pages_processed']}")
                    if result.get('early_stopped'):
                        st.write("**Early stopped:** Yes")
                    if result.get('stopped_at_chunk'):
                        st.write(f"**Stopped at page:** {result['stopped_at_chunk']}")
                
                with col2:
                    st.markdown("### 🗳️ Vote Distribution")
                    votes = result['votes']
                    for vote_class, vote_confidence in votes.items():
                        label = "No Reservations" if vote_class == 0 else "Has Reservations"
                        st.write(f"**{label}:** {vote_confidence:.1%}")
                
                # Show reasoning if available
                if 'detailed_samples' in result and result['detailed_samples']:
                    with st.expander("🔍 Detailed Reasoning"):
                        for i, sample in enumerate(result['detailed_samples'][:3]):  # Show top 3
                            st.markdown(f"**Sample {i+1} (Confidence: {sample['confidence_score']:.1%})**")
                            st.write(sample['reasoning'])
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.info("💡 Try uploading a different PDF or check if the file is corrupted.")

if __name__ == "__main__":
    main() 