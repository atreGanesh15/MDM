import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os

try:
    from symptom_checker_complete import SymptomChecker, SEVERITY_COLORS
except:
    SEVERITY_COLORS = {
        'emergency': '🔴',
        'high': '🟠',
        'moderate': '🟡',
        'mild': '🟢',
        'unknown': '⚪'
    }

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .result-box {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .emergency-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
        font-weight: bold;
    }
    .symptom-chip {
        display: inline-block;
        background-color: #e3f2fd;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.9rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f1f3f4;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================
if 'checker' not in st.session_state:
    st.session_state.checker = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []
if 'conversation_stage' not in st.session_state:
    st.session_state.conversation_stage = 'greeting'

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        if os.path.exists('symptom_checker_model.pkl'):
            checker = SymptomChecker.load_model('symptom_checker_model.pkl')
            return checker, None
        else:
            return None, "Model file not found. Please train the model first."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# ==================== HELPER FUNCTIONS ====================
def add_to_chat(role, message):
    """Add message to chat history"""
    st.session_state.chat_history.append({
        'role': role,
        'message': message,
        'timestamp': datetime.now().strftime("%H:%M")
    })

def display_chat_history():
    """Display chat history with styling"""
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <small>{chat['timestamp']}</small><br>
                <strong>You:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <small>{chat['timestamp']}</small><br>
                <strong>🏥 Health Assistant:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True)

def format_disease_result(result):
    """Format prediction result with rich display"""
    if result.get('error'):
        st.error(f"⚠️ {result['error']}")
        return
    
    disease = result['disease']
    confidence = result.get('confidence', 0)
    severity = result.get('severity', 'unknown')
    
    # Determine box style based on severity
    if severity == 'emergency':
        box_class = "emergency-box"
    else:
        box_class = "result-box"
    
    # Display main result
    st.markdown(f"""
    <div class="{box_class}">
        <h3>{SEVERITY_COLORS.get(severity, '⚪')} Predicted Condition: {disease.title()}</h3>
        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        <p><strong>Severity:</strong> {severity.upper()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display urgency
    st.markdown(f"### ⏰ Urgency")
    st.info(result['urgency'])
    
    # Display medical action
    st.markdown(f"### 💊 Recommended Actions")
    st.success(result['action'])
    
    # Display recognized symptoms
    if result['recognized_symptoms']:
        st.markdown("### ✅ Symptoms Analyzed")
        symptoms_html = "".join([f'<span class="symptom-chip">{s}</span>' 
                                for s in result['recognized_symptoms']])
        st.markdown(symptoms_html, unsafe_allow_html=True)
    
    # Display alternative diagnoses
    if result.get('alternative_diagnoses') and len(result['alternative_diagnoses']) > 1:
        st.markdown("### 🔍 Other Possible Conditions")
        alt_df = pd.DataFrame(
            result['alternative_diagnoses'][1:],  # Skip first as it's the main prediction
            columns=['Condition', 'Probability (%)']
        )
        alt_df['Probability (%)'] = alt_df['Probability (%)'].round(1)
        st.dataframe(alt_df, use_container_width=True)
    
    # Suggest more information if needed
    if result.get('needs_more_info'):
        st.warning("💡 The confidence level is below 70%. Consider providing more symptoms for a more accurate assessment.")

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Health Assistant</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ MEDICAL DISCLAIMER:</strong><br>
        This is an AI-powered tool for educational and informational purposes only. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or qualified health provider with any questions 
        you may have regarding a medical condition. In case of emergency, call 911 immediately.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    checker, error = load_model()
    
    if error:
        st.error(error)
        st.info("Please run the training script first: `python symptom_checker_complete.py`")
        return
    
    st.session_state.checker = checker
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Options")
        
        mode = st.radio(
            "Select Mode:",
            ["💬 Chat Mode", "📋 Quick Check", "📚 Browse Symptoms"],
            index=0
        )
        
        st.markdown("---")
        
        if st.button("🔄 Start New Consultation"):
            st.session_state.chat_history = []
            st.session_state.selected_symptoms = []
            st.session_state.conversation_stage = 'greeting'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        st.metric("Total Symptoms in Database", len(checker.get_all_symptoms()))
        st.metric("Consultations Today", len(st.session_state.chat_history) // 2)
        
        st.markdown("---")
        st.markdown("""
        ### 🆘 Emergency Numbers
        - **Emergency:** 911
        - **Suicide Prevention:** 988
        - **Poison Control:** 1-800-222-1222
        """)
    
    # Main Content Area
    if mode == "💬 Chat Mode":
        chat_mode(checker)
    elif mode == "📋 Quick Check":
        quick_check_mode(checker)
    else:
        browse_symptoms_mode(checker)

# ==================== CHAT MODE ====================
def chat_mode(checker):
    st.header("💬 Chat with Health Assistant")
    
    # Display chat history
    if st.session_state.chat_history:
        display_chat_history()
    else:
        add_to_chat('bot', "Hello! I'm your AI Health Assistant. I can help you understand your symptoms and provide guidance on what actions to take. What symptoms are you experiencing today?")
        display_chat_history()
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your symptoms or message:",
            key="chat_input",
            placeholder="e.g., I have fever and headache..."
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input:
        # Add user message
        add_to_chat('user', user_input)
        
        # Parse symptoms from input
        input_lower = user_input.lower()
        all_symptoms = checker.get_all_symptoms()
        found_symptoms = [s for s in all_symptoms if s in input_lower]
        
        if found_symptoms:
            st.session_state.selected_symptoms.extend(found_symptoms)
            st.session_state.selected_symptoms = list(set(st.session_state.selected_symptoms))
            
            # Get prediction
            result = checker.predict(st.session_state.selected_symptoms)
            
            # Create response
            response = f"I've identified the following symptoms: {', '.join(found_symptoms)}. "
            
            if result.get('disease'):
                response += f"\n\nBased on your symptoms, you may have: **{result['disease'].title()}**"
                if result.get('confidence'):
                    response += f" (Confidence: {result['confidence']:.1f}%)"
                response += f"\n\n**Recommended Action:** {result['action']}"
                response += f"\n\n**Urgency:** {result['urgency']}"
                
                if result.get('needs_more_info'):
                    response += "\n\nWould you like to provide more symptoms for a more accurate assessment?"
            
            add_to_chat('bot', response)
        else:
            response = "I didn't recognize any specific symptoms in your message. Could you please describe your symptoms? For example: 'fever', 'headache', 'cough', etc."
            add_to_chat('bot', response)
        
        st.rerun()
    
    # Quick symptom buttons
    if st.session_state.selected_symptoms:
        st.markdown("### Selected Symptoms:")
        cols = st.columns(4)
        for idx, symptom in enumerate(st.session_state.selected_symptoms):
            with cols[idx % 4]:
                if st.button(f"❌ {symptom}", key=f"remove_{symptom}"):
                    st.session_state.selected_symptoms.remove(symptom)
                    st.rerun()
        
        if st.button("🔍 Get Diagnosis", use_container_width=True, type="primary"):
            result = checker.predict(st.session_state.selected_symptoms)
            st.markdown("---")
            format_disease_result(result)

# ==================== QUICK CHECK MODE ====================
def quick_check_mode(checker):
    st.header("📋 Quick Symptom Check")
    st.write("Select your symptoms from the list below:")
    
    # Get all symptoms
    all_symptoms = checker.get_all_symptoms()
    
    # Search box
    search = st.text_input("🔍 Search symptoms:", placeholder="Type to search...")
    
    if search:
        filtered_symptoms = [s for s in all_symptoms if search.lower() in s.lower()]
    else:
        filtered_symptoms = all_symptoms
    
    # Display symptoms in columns with checkboxes
    st.markdown("### Available Symptoms:")
    
    cols = st.columns(3)
    selected = []
    
    for idx, symptom in enumerate(filtered_symptoms[:100]):  # Limit to 100 for performance
        with cols[idx % 3]:
            if st.checkbox(symptom.title(), key=f"symptom_{symptom}"):
                selected.append(symptom)
    
    if len(filtered_symptoms) > 100:
        st.info(f"Showing 100 of {len(filtered_symptoms)} symptoms. Use search to find more.")
    
    # Predict button
    st.markdown("---")
    if st.button("🔍 Analyze Symptoms", type="primary", disabled=len(selected) == 0):
        if selected:
            result = checker.predict(selected)
            st.markdown("---")
            format_disease_result(result)
        else:
            st.warning("Please select at least one symptom.")

# ==================== BROWSE SYMPTOMS MODE ====================
def browse_symptoms_mode(checker):
    st.header("📚 Browse Symptoms Database")
    
    all_symptoms = checker.get_all_symptoms()
    
    st.write(f"**Total symptoms in database:** {len(all_symptoms)}")
    
    # Search and filter
    search = st.text_input("🔍 Search symptoms:")
    
    if search:
        filtered = [s for s in all_symptoms if search.lower() in s.lower()]
    else:
        filtered = all_symptoms
    
    # Display in a nice format
    st.markdown(f"**Showing {len(filtered)} symptoms:**")
    
    # Create a dataframe for better display
    df = pd.DataFrame(filtered, columns=['Symptom'])
    df['Symptom'] = df['Symptom'].str.title()
    
    st.dataframe(df, use_container_width=True, height=500)
    
    # Download option
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Symptom List",
        data=csv,
        file_name="symptoms_list.csv",
        mime="text/csv"
    )

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
