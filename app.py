import streamlit as st
import numpy as np
import pickle
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Depression Prediction Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

def check_model_file():
    """Check if the model file exists and is accessible"""
    model_path = Path('model/depression_model.pkl')
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path.absolute()}")
        st.info("Please ensure the model file is in the correct location.")
        return False
    return True

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the model with proper error handling"""
    try:
        if not check_model_file():
            return None
            
        with open('model/depression_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Validate model data structure
        required_keys = ['model', 'scaler', 'label_encoders', 'feature_columns', 'sleep_duration_map']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            st.error(f"❌ Model file is missing required components: {missing_keys}")
            return None
            
        logger.info("Model loaded successfully")
        return model_data
        
    except FileNotFoundError:
        st.error("❌ Model file not found. Please check the file path.")
        return None
    except pickle.UnpicklingError:
        st.error("❌ Error loading model file. The file may be corrupted.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {str(e)}")
        logger.error(f"Model loading error: {e}")
        return None

def validate_inputs(age, academic_pressure, study_hours, financial_stress):
    """Validate user inputs and return error messages"""
    errors = []
    
    # Validate data types
    if not isinstance(age, (int, float)):
        errors.append("Age must be a number")
    elif age < 10 or age > 100:
        errors.append("Age must be between 10 and 100")
    
    if not isinstance(academic_pressure, (int, float)):
        errors.append("Academic pressure must be a number")
    elif academic_pressure < 1 or academic_pressure > 10:
        errors.append("Academic pressure must be between 1 and 10")
    
    if not isinstance(study_hours, (int, float)):
        errors.append("Study hours must be a number")
    elif study_hours < 0 or study_hours > 20:
        errors.append("Study hours must be between 0 and 20")
    
    if not isinstance(financial_stress, (int, float)):
        errors.append("Financial stress must be a number")
    elif financial_stress < 1 or financial_stress > 10:
        errors.append("Financial stress must be between 1 and 10")
    
    return errors

def main():
    st.markdown('<h1 class="main-header">🧠 Depression Prediction Tool</h1>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    sleep_duration_map = model_data['sleep_duration_map']
    

    
    # Main form
    with st.form("prediction_form"):
        st.header("📝 Enter Your Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
            age = st.number_input("Age", min_value=0, max_value=100, value=21, help="Enter your current age")
            academic_pressure = st.slider("Academic Pressure (1-10)", 1, 10, 5, 
                                        help="Rate your academic pressure level")
            study_satisfaction = st.selectbox("Study Satisfaction", label_encoders['Study Satisfaction'].classes_)
            sleep_duration = st.selectbox("Sleep Duration", list(sleep_duration_map.keys()))
        
        with col2:
            dietary_habits = st.selectbox("Dietary Habits", label_encoders['Dietary Habits'].classes_)
            study_hours = st.number_input("Study Hours per Day", min_value=0, max_value=20, value=5, 
                                        help="Average hours spent studying per day")
            financial_stress = st.slider("Financial Stress (1-10)", 1, 10, 5, 
                                       help="Rate your financial stress level")
            family_history = st.selectbox("Family History of Mental Illness", 
                                        label_encoders['Family History of Mental Illness'].classes_)
            suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", 
                                           label_encoders['Have you ever had suicidal thoughts ?'].classes_)
        
        submitted = st.form_submit_button("🔮 Get Prediction", use_container_width=True)
    
    if submitted:
        # Validate inputs
        validation_errors = validate_inputs(age, academic_pressure, study_hours, financial_stress)
        
        if validation_errors:
            st.error("❌ Please correct the following errors:")
            for error in validation_errors:
                st.error(f"• {error}")
            return
        
        # Show processing indicator
        with st.spinner("🤖 Analyzing your data..."):
            try:
                def encode_feature(col_name, val):
                    le = label_encoders.get(col_name)
                    if le is None:
                        st.error(f"❌ Label encoder not found for column: {col_name}")
                        return None
                    try:
                        return le.transform([val])[0]
                    except ValueError as e:
                        st.error(f"❌ Invalid value '{val}' for {col_name}. Available values: {list(le.classes_)}")
                        return None
                    except Exception as e:
                        st.error(f"❌ Error encoding {col_name}: {str(e)}")
                        return None
                
                # Prepare input data with better error handling
                sleep_duration_value = sleep_duration_map.get(sleep_duration)
                if sleep_duration_value is None:
                    st.error(f"❌ Invalid sleep duration: '{sleep_duration}'. Available options: {list(sleep_duration_map.keys())}")
                    return
                
                # Validate sleep duration value range
                if not isinstance(sleep_duration_value, (int, float)) or sleep_duration_value < 0:
                    st.error(f"❌ Invalid sleep duration value: {sleep_duration_value}")
                    return
                
                input_data = {
                    'Gender': encode_feature('Gender', gender),
                    'Age': age,
                    'Academic Pressure': academic_pressure,
                    'Study Satisfaction': encode_feature('Study Satisfaction', study_satisfaction),
                    'Sleep Duration': sleep_duration_value,
                    'Dietary Habits': encode_feature('Dietary Habits', dietary_habits),
                    'Study Hours': study_hours,
                    'Financial Stress': financial_stress,
                    'Family History of Mental Illness': encode_feature('Family History of Mental Illness', family_history),
                    'Have you ever had suicidal thoughts ?': encode_feature('Have you ever had suicidal thoughts ?', suicidal_thoughts)
                }
                
                # Check for encoding errors (simplified since encode_feature already shows errors)
                if None in input_data.values():
                    # Error messages already shown by encode_feature function
                    return
                
                # Create feature array with validation
                try:
                    feature_values = [input_data[col] for col in feature_columns]
                    if len(feature_values) != len(feature_columns):
                        st.error("❌ Mismatch between feature values and expected columns")
                        return
                    
                    X = np.array(feature_values).reshape(1, -1)
                    X_scaled = scaler.transform(X)
                except Exception as e:
                    st.error(f"❌ Error creating feature array: {str(e)}")
                    logger.error(f"Feature array error: {e}")
                    return
                
                # Make prediction with error handling
                try:
                    prediction = model.predict(X_scaled)[0]
                    probas = model.predict_proba(X_scaled)
                    
                    # Handle both string and numeric predictions
                    if prediction in ['Yes', 1, '1']:
                        prediction_binary = 1
                        prediction_label = "High Risk"
                    elif prediction in ['No', 0, '0']:
                        prediction_binary = 0
                        prediction_label = "Low Risk"
                    else:
                        st.error(f"❌ Invalid prediction value: {prediction}. Expected Yes/No or 0/1.")
                        return
                    
                    proba = probas[0, prediction_binary]
                except Exception as e:
                    st.error(f"❌ Error during model prediction: {str(e)}")
                    logger.error(f"Model prediction error: {e}")
                    return
                
                # Display results
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                if prediction_binary == 1:
                    st.markdown(f"""
                    <div class="prediction-box error-box">
                        <h3>⚠️ High Risk of Depression</h3>
                        <p><strong>Confidence:</strong> {proba:.1%}</p>
                        <p><em>Please consider seeking professional help and support.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box success-box">
                        <h3>✅ Low Risk of Depression</h3>
                        <p><strong>Confidence:</strong> {proba:.1%}</p>
                        <p><em>Continue maintaining healthy habits and seek support if needed.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("---")
                st.header("💡 Insights & Recommendations")
                
                if prediction_binary == 1:
                    st.warning("**High Risk Indicators Detected:**")
                    if academic_pressure >= 7:
                        st.write("• High academic pressure detected")
                    if financial_stress >= 7:
                        st.write("• High financial stress detected")
                    if suicidal_thoughts == 'Yes':
                        st.write("• Suicidal thoughts history detected")
                    
                    st.info("**Recommendations:**")
                    st.write("• Consider speaking with a mental health professional")
                    st.write("• Reach out to trusted friends or family members")
                    st.write("• Contact a crisis helpline if needed")
                    st.write("• Practice self-care and stress management techniques")
                else:
                    st.success("**Positive Factors:**")
                    if academic_pressure <= 5:
                        st.write("• Manageable academic pressure")
                    if financial_stress <= 5:
                        st.write("• Manageable financial stress")
                    # Fix: Handle both string and numeric study satisfaction values
                    if study_satisfaction in ['4.0', '5.0'] or study_satisfaction in [4.0, 5.0] or str(study_satisfaction) in ['4.0', '5.0']:
                        st.write("• Good study satisfaction")
                    
                    st.info("**Continue these healthy practices:**")
                    st.write("• Maintain regular sleep patterns")
                    st.write("• Keep balanced study hours")
                    st.write("• Stay connected with support networks")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                logger.error(f"Prediction error: {e}")
                st.info("Please try again or contact support if the issue persists.")

if __name__ == "__main__":
    main()
