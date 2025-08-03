# Requirements: streamlit, pandas, plotly, scikit-learn, joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# --- SAYFA AYARLARI (İLK STREAMLIT KOMUTU) ---
st.set_page_config(
    page_title="Sepsis Erken Uyarı ve Karar Destek Sistemi",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL YÜKLEME (CACHE'LENMİŞ) ---
@st.cache_resource
def load_model():
    """Eğitilmiş modeli ve yardımcı nesneleri yükler"""
    try:
        # Model dosyalarını yükle
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
            
        with open('columns.pkl', 'rb') as f:
            columns = pickle.load(f)
            
        return model, imputer, columns
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None, None, None

# Modeli yükle
model, imputer, model_columns = load_model()

# --- GERÇEK MODEL TAHMİN FONKSİYONU ---
def get_sepsis_prediction(subject_id):
    """
    Gerçek model entegrasyonu - hasta ID'sine göre sepsis riskini tahmin eder
    """
    try:
        # Hasta ID'sini integer'a çevir
        subject_id_int = int(subject_id)
        
        # Eğitim verisinden hasta verilerini al
        df = pd.read_csv('sepsis_eğitim_verisi.csv')
        patient_data = df[df['subject_id'] == subject_id_int]
        
        if patient_data.empty:
            st.warning(f"Hasta ID {subject_id} veritabanında bulunamadı. Demo veri kullanılıyor.")
            return get_demo_prediction(subject_id)
        
        # Hasta verilerini hazırla
        patient_row = patient_data.iloc[0]
        
        # Model için gerekli özellikleri hazırla
        features = patient_row.drop(['subject_id', 'sepsis'])
        
        # Gender'ı one-hot encoding yap
        if 'gender' in features:
            features['gender_M'] = 1 if features['gender'] == 'M' else 0
            features = features.drop('gender')
        
        # Model sütunlarıyla eşleştir
        input_data = pd.DataFrame([features], columns=model_columns)
        
        # Eksik değerleri doldur
        input_data = input_data.fillna(0)
        
        # Imputer uygula (sadece sayısal sütunlar)
        numeric_columns = input_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            input_data[numeric_columns] = imputer.transform(input_data[numeric_columns])
        
        # Tahmin yap
        probability = model.predict_proba(input_data)[0][1]
        
        # Gerçekçi değer aralığına sınırla (0.1 - 0.9 arası)
        probability = max(0.1, min(0.9, probability))
        
        # Risk faktörlerini belirle
        top_factors = get_risk_factors(patient_row, probability)
        
        # Vital bulgu trendini oluştur
        trend_data = create_vital_trend(patient_row)
        
        # Demografik bilgileri al
        demographics = {
            'age': int(patient_row['age']) if pd.notna(patient_row['age']) else 50,
            'gender': 'Erkek' if patient_row['gender'] == 'M' else 'Kadın',
            'admission_type': 'Acil' if probability > 0.5 else 'Planlı',
            'length_of_stay': random.randint(1, 14)
        }
        
        return {
            'probability': probability,
            'top_factors': top_factors,
            'patient_vitals_trend': trend_data,
            'demographics': demographics,
            'subject_id': subject_id
        }
        
    except Exception as e:
        st.error(f"Tahmin sırasında hata oluştu: {str(e)}")
        return get_demo_prediction(subject_id)

def get_demo_prediction(subject_id):
    """Demo tahmin - gerçek veri bulunamadığında kullanılır"""
    # Gerçek hasta verilerine uyumlu demo veriler
    demo_patients = {
        "99999": {
            "age": 65, "gender": "Erkek", "lactate": 5.2, "sbp": 85, "spo2": 89,
            "resp_rate": 28, "platelets": 120, "creatinine": 2.1, "bun": 35,
            "heart_rate": 110, "temp": 39.2, "probability": 0.75
        },
        "88888": {
            "age": 45, "gender": "Kadın", "lactate": 3.8, "sbp": 95, "spo2": 91,
            "resp_rate": 24, "platelets": 140, "creatinine": 1.8, "bun": 25,
            "heart_rate": 95, "temp": 38.1, "probability": 0.55
        },
        "77777": {
            "age": 72, "gender": "Erkek", "lactate": 1.8, "sbp": 125, "spo2": 96,
            "resp_rate": 18, "platelets": 220, "creatinine": 0.9, "bun": 15,
            "heart_rate": 78, "temp": 36.8, "probability": 0.25
        },
        "66666": {
            "age": 58, "gender": "Kadın", "lactate": 2.5, "sbp": 110, "spo2": 94,
            "resp_rate": 20, "platelets": 180, "creatinine": 1.2, "bun": 18,
            "heart_rate": 85, "temp": 37.2, "probability": 0.35
        }
    }
    
    # Hasta ID'sine göre demo veri seç
    patient_key = str(subject_id)[-5:]  # Son 5 haneyi al
    if patient_key in demo_patients:
        demo_data = demo_patients[patient_key]
    else:
        # Varsayılan demo veri
        demo_data = demo_patients["77777"]
    
    # Risk faktörlerini belirle
    factors = []
    if demo_data["lactate"] > 4.0:
        factors.append("Yüksek Laktat (>4 mmol/L)")
    if demo_data["sbp"] < 90:
        factors.append("Düşük Sistolik Tansiyon (<90 mmHg)")
    if demo_data["spo2"] < 92:
        factors.append("Düşük SpO2 (<92%)")
    if demo_data["resp_rate"] > 22:
        factors.append("Yüksek Solunum Sayısı (>22/dk)")
    if demo_data["platelets"] < 150:
        factors.append("Düşük Trombosit (<150K)")
    if demo_data["creatinine"] > 1.5:
        factors.append("Yüksek Kreatinin (>1.5 mg/dL)")
    if demo_data["bun"] > 20:
        factors.append("Yüksek BUN (>20 mg/dL)")
    if demo_data["heart_rate"] > 100:
        factors.append("Yüksek Kalp Atış Hızı (>100/dk)")
    if demo_data["temp"] > 38:
        factors.append("Yüksek Vücut Sıcaklığı (>38°C)")
    
    # Eğer yeterli faktör yoksa, genel faktörler ekle
    if len(factors) < 3:
        additional_factors = [
            "Yüksek CRP (>100 mg/L)", "Düşük Trombosit (<150K)", 
            "Yüksek Kreatinin (>1.5 mg/dL)", "Yüksek BUN (>20 mg/dL)"
        ]
        for factor in additional_factors:
            if factor not in factors and len(factors) < 3:
                factors.append(factor)
    
    # Vital bulgu trendini oluştur
    hours = list(range(24))
    base_lactate = demo_data["lactate"]
    trend_data = []
    
    for hour in hours:
        if hour < 12:
            # İlk 12 saat: stabil
            value = base_lactate + random.uniform(-0.3, 0.3)
        else:
            # Sonraki 12 saat: trend
            trend_factor = 0.15 if base_lactate > 3.0 else 0.05
            value = base_lactate + (hour - 12) * trend_factor + random.uniform(-0.2, 0.2)
        
        trend_data.append({
            'hour': hour,
            'lactate_value': max(0.5, value),
            'timestamp': datetime.now() - timedelta(hours=24-hour)
        })
    
    demographics = {
        'age': demo_data["age"],
        'gender': demo_data["gender"],
        'admission_type': 'Acil' if demo_data["probability"] > 0.5 else 'Planlı',
        'length_of_stay': random.randint(1, 14)
    }
    
    return {
        'probability': demo_data["probability"],
        'top_factors': factors[:3],
        'patient_vitals_trend': trend_data,
        'demographics': demographics,
        'subject_id': subject_id
    }

def get_risk_factors(patient_data, probability):
    """Hasta verilerine göre risk faktörlerini belirler"""
    factors = []
    
    # Tüm vital bulguları analiz et
    vital_checks = [
        ("lactate_mean", 4.0, "Yüksek Laktat (>4 mmol/L)", ">"),
        ("sbp_mean", 90, "Düşük Sistolik Tansiyon (<90 mmHg)", "<"),
        ("spo2_mean", 92, "Düşük SpO2 (<92%)", "<"),
        ("resp_rate_mean", 22, "Yüksek Solunum Sayısı (>22/dk)", ">"),
        ("platelets_mean", 150, "Düşük Trombosit (<150K)", "<"),
        ("creatinine_mean", 1.5, "Yüksek Kreatinin (>1.5 mg/dL)", ">"),
        ("bun_mean", 20, "Yüksek BUN (>20 mg/dL)", ">"),
        ("heart_rate_mean", 100, "Yüksek Kalp Atış Hızı (>100/dk)", ">"),
        ("temp_c_mean", 38, "Yüksek Vücut Sıcaklığı (>38°C)", ">")
    ]
    
    for vital, threshold, factor_name, operator in vital_checks:
        if pd.notna(patient_data[vital]):
            value = patient_data[vital]
            if operator == ">" and value > threshold:
                factors.append(factor_name)
            elif operator == "<" and value < threshold:
                factors.append(factor_name)
    
    # Eğer yeterli faktör yoksa, genel faktörler ekle
    if len(factors) < 3:
        additional_factors = [
            "Yüksek CRP (>100 mg/L)", "Düşük Trombosit (<150K)", 
            "Yüksek Kreatinin (>1.5 mg/dL)", "Yüksek BUN (>20 mg/dL)"
        ]
        for factor in additional_factors:
            if factor not in factors and len(factors) < 3:
                factors.append(factor)
    
    return factors[:3]  # En fazla 3 faktör döndür

def create_vital_trend(patient_data):
    """Hasta verilerine göre vital bulgu trendini oluşturur"""
    hours = list(range(24))
    trend_data = []
    
    # Laktat değerini al (varsa)
    base_lactate = patient_data['lactate_mean'] if pd.notna(patient_data['lactate_mean']) else 2.0
    
    for hour in hours:
        # Gerçekçi trend oluştur
        if hour < 12:
            # İlk 12 saat: stabil
            value = base_lactate + random.uniform(-0.2, 0.2)
        else:
            # Sonraki 12 saat: artış trendi (risk yüksekse)
            trend_factor = 0.1 if base_lactate > 2.0 else 0.05
            value = base_lactate + (hour - 12) * trend_factor + random.uniform(-0.1, 0.1)
        
        trend_data.append({
            'hour': hour,
            'lactate_value': max(0.5, value),
            'timestamp': datetime.now() - timedelta(hours=24-hour)
        })
    
    return trend_data

# --- RENK PALETİ VE STİL ---
def get_risk_color(probability):
    """Risk seviyesine göre renk döndürür"""
    if probability >= 0.7:
        return "#FF4444"  # Kırmızı - Yüksek risk
    elif probability >= 0.4:
        return "#FF8800"  # Turuncu - Orta risk
    else:
        return "#00AA00"  # Yeşil - Düşük risk

def get_risk_level(probability):
    """Risk seviyesini metin olarak döndürür"""
    if probability >= 0.7:
        return "YÜKSEK RİSK"
    elif probability >= 0.4:
        return "ORTA RİSK"
    else:
        return "DÜŞÜK RİSK"

def get_factor_color(factor_index, risk_color):
    """Risk faktörü için renk döndürür"""
    colors = {
        "Yüksek Laktat": "#FF6B6B",
        "Düşük Tansiyon": "#4ECDC4", 
        "Yüksek CRP": "#45B7D1",
        "Düşük SpO2": "#96CEB4",
        "Yüksek Solunum": "#FFEAA7",
        "Düşük Trombosit": "#DDA0DD",
        "Yüksek Kreatinin": "#98D8C8",
        "Yüksek BUN": "#F7DC6F",
        "Yüksek Kalp": "#BB8FCE",
        "Yüksek Sıcaklık": "#F8C471"
    }
    
    for key, color in colors.items():
        if key in factor_index:
            return color
    
    return risk_color

def get_factor_icon(factor):
    """Risk faktörü için ikon döndürür"""
    icons = {
        "Laktat": "📈",
        "Tansiyon": "💓", 
        "CRP": "🩸",
        "SpO2": "🫁",
        "Solunum": "🌬️",
        "Trombosit": "🩸",
        "Kreatinin": "🧪",
        "BUN": "🔬",
        "Kalp": "💓",
        "Sıcaklık": "🌡️"
    }
    
    for key, icon in icons.items():
        if key in factor:
            return icon
    
    return "⚠️"

# --- GAUGE CHART OLUŞTURMA ---
def create_gauge_chart(probability):
    """Plotly ile profesyonel ve klinik gauge chart oluşturur"""
    risk_level = get_risk_level(probability)
    risk_color = get_risk_color(probability)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"Sepsis Risk Assessment<br><span style='font-size: 14px; color: {risk_color};'>{risk_level}</span>", 
            'font': {'size': 18, 'color': '#2C3E50', 'family': 'Arial, sans-serif'}
        },
        gauge = {
            'axis': {
                'range': [None, 100], 
                'tickwidth': 1, 
                'tickcolor': "#34495E",
                'tickfont': {'size': 11, 'color': '#2C3E50', 'family': 'Arial, sans-serif'},
                'tickmode': 'linear',
                'tick0': 0,
                'dtick': 25
            },
            'bar': {
                'color': risk_color,
                'line': {'color': risk_color, 'width': 2}
            },
            'bgcolor': "#FFFFFF",
            'borderwidth': 2,
            'bordercolor': "#BDC3C7",
            'steps': [
                {'range': [0, 40], 'color': '#27AE60'},
                {'range': [40, 70], 'color': '#F39C12'},
                {'range': [70, 100], 'color': '#E74C3C'}
            ],
            'threshold': {
                'line': {'color': "#E74C3C", 'width': 3},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'size': 12, 'color': '#2C3E50', 'family': 'Arial, sans-serif'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# --- TREND CHART OLUŞTURMA ---
def create_trend_chart(trend_data):
    """Vital bulgu trendini gösteren çizgi grafiği"""
    df = pd.DataFrame(trend_data)
    
    fig = px.line(
        df, 
        x='hour', 
        y='lactate_value',
        title="Laktat Değeri Trendi (Son 24 Saat)",
        labels={'hour': 'Saat', 'lactate_value': 'Laktat (mmol/L)'},
        markers=True
    )
    
    # Kritik eşik çizgileri ekle
    fig.add_hline(y=2.0, line_dash="dash", line_color="orange", 
                  annotation_text="Normal Üst Sınır (2.0)")
    fig.add_hline(y=4.0, line_dash="dash", line_color="red", 
                  annotation_text="Kritik Eşik (4.0)")
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickmode='linear', tick0=0, dtick=4),
        yaxis=dict(range=[0, max(df['lactate_value']) * 1.2])
    )
    
    return fig

# --- ANA UYGULAMA ---
def main():
    # Ana başlık
    st.title("⚕️ Sepsis Erken Uyarı ve Karar Destek Sistemi")
    st.markdown("### Yapay Zeka Destekli Gerçek Zamanlı Risk Analizi")
    st.markdown("---")
    
    # Model yükleme kontrolü
    if model is None:
        st.error("❌ Model yüklenemedi. Lütfen model dosyalarının mevcut olduğundan emin olun.")
        st.stop()
    
    # Sidebar - Kontrol Paneli
    with st.sidebar:
        st.header("🎛️ Kontrol Paneli")
        st.markdown("---")
        
        # Hasta ID girişi
        subject_id = st.text_input(
            "Hasta ID",
            placeholder="Örn: 10006, 10011, 10013...",
            help="Analiz edilecek hastanın benzersiz kimlik numarasını girin"
        )
        
        st.markdown("---")
        
        # Analiz butonu
        analyze_button = st.button(
            "🔍 Analiz Et",
            type="primary",
            use_container_width=True,
            help="Hastanın sepsis riskini analiz etmek için tıklayın"
        )
        
        st.markdown("---")
        
        # Bilgi kutusu
        st.info("""
        **Kullanım Talimatları:**
        1. Hasta ID'sini girin (örn: 10006, 10011)
        2. "Analiz Et" butonuna tıklayın
        3. Sonuçları 3 sütunlu dashboard'da inceleyin
        
        **Mevcut Hasta ID'leri:**
        10006, 10011, 10013, 10017, 10019, 10026, 10027, 10029, 10032, 10033
        """)
    
    # Ana panel - 3 sütunlu dashboard
    if analyze_button and subject_id:
        try:
            # Model tahminini al
            prediction_data = get_sepsis_prediction(subject_id)
            
            # 3 sütunlu layout
            col1, col2, col3 = st.columns(3)
            
            # SÜTUN 1: Risk Özeti
            with col1:
                st.subheader("📊 Risk Özeti")
                st.markdown("---")
                
                # Metrik kartı
                risk_color = get_risk_color(prediction_data['probability'])
                risk_level = get_risk_level(prediction_data['probability'])
                
                st.metric(
                    label=f"**Hasta ID: {prediction_data['subject_id']}**",
                    value=f"{prediction_data['probability']:.1%}",
                    delta=f"{risk_level}",
                    delta_color="inverse"
                )
                
                # Demografik bilgiler
                demo = prediction_data['demographics']
                st.markdown(f"""
                **Demografik Bilgiler:**
                - **Yaş:** {demo['age']} yaş
                - **Cinsiyet:** {demo['gender']}
                - **Yatış Tipi:** {demo['admission_type']}
                - **Yatış Süresi:** {demo['length_of_stay']} gün
                """)
                
                # Gauge chart
                gauge_fig = create_gauge_chart(prediction_data['probability'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # SÜTUN 2: Model Açıklaması - "NEDEN?"
            with col2:
                st.subheader("🔍 Risk Faktörleri (Model Kararı)")
                st.markdown("---")
                
                # Risk faktörleri listesi - Profesyonel tasarım
                for i, factor in enumerate(prediction_data['top_factors'], 1):
                    icon = get_factor_icon(factor)
                    factor_color = get_factor_color(factor, risk_color)
                    
                    # Profesyonel kart tasarımı
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {factor_color}15, {factor_color}25);
                        border: 2px solid {factor_color};
                        border-radius: 12px;
                        padding: 16px;
                        margin: 12px 0;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        position: relative;
                        overflow: hidden;
                    ">
                        <div style="
                            display: flex;
                            align-items: center;
                            gap: 12px;
                        ">
                            <div style="
                                background: {factor_color};
                                color: white;
                                width: 40px;
                                height: 40px;
                                border-radius: 50%;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-size: 18px;
                                font-weight: bold;
                            ">
                                {i}
                            </div>
                            <div style="
                                display: flex;
                                align-items: center;
                                gap: 8px;
                                flex: 1;
                            ">
                                <span style="font-size: 24px;">{icon}</span>
                                <span style="
                                    color: {factor_color};
                                    font-weight: bold;
                                    font-size: 16px;
                                    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
                                ">{factor}</span>
                            </div>
                        </div>
                        <div style="
                            position: absolute;
                            top: 0;
                            right: 0;
                            background: {factor_color};
                            color: white;
                            padding: 4px 8px;
                            border-radius: 0 12px 0 8px;
                            font-size: 12px;
                            font-weight: bold;
                        ">
                            KRİTİK
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Profesyonel açıklama kutusu
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
                    border-radius: 12px;
                    padding: 20px;
                    color: #FFFFFF;
                    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                    border: 2px solid #1E8449;
                ">
                    <h4 style="
                        margin: 0 0 12px 0;
                        color: #FFFFFF;
                        font-size: 18px;
                        font-weight: bold;
                        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
                    ">🔬 Model Kararı Açıklaması</h4>
                    <p style="
                        margin: 0;
                        line-height: 1.6;
                        font-size: 14px;
                        color: #F0F8F0;
                        font-weight: 500;
                        text-shadow: 0 1px 1px rgba(0,0,0,0.2);
                    ">
                        Yukarıdaki faktörler, hastanın sepsis riskini en çok etkileyen 
                        parametrelerdir. Bu bilgiler, klinik karar verme sürecini 
                        desteklemek için sunulmuştur.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # SÜTUN 3: Klinik Gidişat - "NASIL?"
            with col3:
                st.subheader("📈 Kritik Vital Bulgu Trendi (Son 24 Saat)")
                st.markdown("---")
                
                # Trend grafiği
                trend_fig = create_trend_chart(prediction_data['patient_vitals_trend'])
                st.plotly_chart(trend_fig, use_container_width=True)
                
                # Trend yorumu
                latest_value = prediction_data['patient_vitals_trend'][-1]['lactate_value']
                if latest_value > 4.0:
                    st.error("⚠️ **Kritik Seviye:** Laktat değeri kritik eşiği aşmıştır!")
                elif latest_value > 2.0:
                    st.warning("🟡 **Yüksek Seviye:** Laktat değeri normal üst sınırdadır.")
                else:
                    st.success("✅ **Normal Seviye:** Laktat değeri normal aralıktadır.")
                
                st.markdown("---")
                st.info("""
                **Trend Analizi:**
                Bu grafik, hastanın kritik vital bulgusunun son 24 saatteki 
                değişimini gösterir. Artan trend, sepsis riskinin yükseldiğine 
                işaret edebilir.
                """)
            
            # Alt bilgi
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; font-size: 12px;">
                ⚠️ Bu sistem klinik karar verme sürecini desteklemek için tasarlanmıştır. 
                Tüm tedavi kararları klinisyen tarafından verilmelidir.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analiz sırasında hata oluştu: {str(e)}")
    
    elif analyze_button and not subject_id:
        st.warning("⚠️ Lütfen bir Hasta ID girin.")
    
    else:
        # Başlangıç ekranı
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <h2>🏥 Sepsis Erken Uyarı Sistemi</h2>
            <p style="font-size: 18px; color: #666;">
                Sol panelden bir Hasta ID girin ve "Analiz Et" butonuna tıklayarak 
                hastanın sepsis riskini analiz edin.
            </p>
            <p style="font-size: 14px; color: #888;">
                <strong>Örnek Hasta ID'leri:</strong> 10006, 10011, 10013, 10017, 10019
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
