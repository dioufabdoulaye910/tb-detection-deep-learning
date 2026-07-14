import os
import tempfile
from io import BytesIO

import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from utils import load_model, predict, preprocess_image
from database import (
    init_db,
    save_analysis,
    get_analyses,
    get_patients,
    get_analyses_by_patient,
    delete_all_analyses
)
from gradcam import make_gradcam_heatmap, overlay_heatmap


st.set_page_config(
    page_title="Détection Tuberculose IA",
    page_icon="🫁",
    layout="wide"
)

init_db()


st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #12355B;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}
.card-success {
    background-color: #E8F8EE;
    border: 2px solid #2EAD5B;
}
.card-warning {
    background-color: #FFF4D6;
    border: 2px solid #F5A623;
}
.card-danger {
    background-color: #FFE5E5;
    border: 2px solid #E53935;
}
.metric-title {
    color: #666;
    font-size: 15px;
}
.metric-value {
    color: #111;
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model("tb_model.keras")


model = get_model()


def niveau_risque(score):
    if score < 0.30:
        return "Risque faible", "🟢", "card-success"
    elif score < 0.60:
        return "Risque modéré", "🟠", "card-warning"
    else:
        return "Risque élevé", "🔴", "card-danger"


def metric_card(title, value, subtitle):
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def gauge_chart(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={"text": "Risque estimé de tuberculose"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#12355B"},
                "steps": [
                    {"range": [0, 30], "color": "#D9F2E3"},
                    {"range": [30, 60], "color": "#FFF1C2"},
                    {"range": [60, 100], "color": "#FFD6D6"},
                ],
            },
        )
    )
    return fig


def generate_pdf(patient_name, prediction, result, risk_level, threshold, image=None, gradcam_image=None):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "Rapport d'analyse - Détection Tuberculose IA")
    y -= 35

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, "Application d'aide pédagogique à l'analyse de radiographies thoraciques.")
    y -= 35

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Informations patient")
    y -= 25

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Patient : {patient_name}")
    y -= 20
    pdf.drawString(50, y, f"Résultat : {result}")
    y -= 20
    pdf.drawString(50, y, f"Score du modèle : {prediction:.4f}")
    y -= 20
    pdf.drawString(50, y, f"Probabilité estimée : {prediction * 100:.2f} %")
    y -= 20
    pdf.drawString(50, y, f"Niveau de risque : {risk_level}")
    y -= 20
    pdf.drawString(50, y, f"Seuil utilisé : {threshold:.2f}")
    y -= 35

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Recommandation")
    y -= 25

    pdf.setFont("Helvetica", 10)

    if prediction >= threshold:
        recommendation = "Suspicion de tuberculose. Une consultation médicale est fortement recommandée."
    else:
        recommendation = "Aucune tuberculose détectée par le modèle. En cas de symptômes, consulter un médecin."

    pdf.drawString(50, y, recommendation)
    y -= 40

    if image is not None:
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, "Radiographie analysée")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        pdf.drawImage(image_path, 50, y - 190, width=220, height=180)
        os.remove(image_path)

    if gradcam_image is not None:
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(310, y, "Carte Grad-CAM")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            Image.fromarray(gradcam_image).save(tmp.name)
            gradcam_path = tmp.name

        pdf.drawImage(gradcam_path, 310, y - 190, width=220, height=180)
        os.remove(gradcam_path)

    y -= 240

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        50,
        y,
        "Important : ce rapport ne remplace pas un diagnostic médical officiel."
    )

    pdf.save()
    buffer.seek(0)
    return buffer


def dataframe_from_rows(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Patient",
            "Score",
            "Résultat",
            "Risque",
            "Seuil",
            "Date"
        ]
    )


st.sidebar.title("🫁 TB Detection IA")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "🔍 Analyse",
        "👤 Dossier patient",
        "📊 Historique",
        "📈 Statistiques",
        "ℹ️ À propos"
    ]
)

st.sidebar.markdown("---")

threshold = st.sidebar.slider(
    "Seuil de décision",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

gradcam_option = st.sidebar.checkbox(
    "Activer Grad-CAM",
    value=True
)

st.sidebar.caption(f"Seuil actuel : {threshold:.2f}")

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ Outil pédagogique. Ne remplace pas un diagnostic médical.")


if page == "🏠 Accueil":
    st.markdown(
        '<div class="main-title">🫁 Détection de la Tuberculose par IA</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Analyse intelligente de radiographies thoraciques.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    rows = get_analyses()
    total = len(rows)
    positives = sum(1 for r in rows if r[3] == "Tuberculose suspectée")
    taux = (positives / total * 100) if total > 0 else 0
    patients = get_patients()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Examens analysés", total, "Nombre total d'analyses enregistrées.")
    with c2:
        metric_card("Patients", len(patients), "Nombre de patients différents.")
    with c3:
        metric_card("Cas suspects", positives, "Nombre de cas positifs selon le modèle.")
    with c4:
        metric_card("Taux de suspicion", f"{taux:.1f} %", "Proportion de cas suspects.")

    st.markdown("### 🎯 Objectif")
    st.write("""
    Cette application permet d’importer une radiographie thoracique et d’obtenir une estimation
    du risque de tuberculose grâce à un modèle de Deep Learning.
    """)

    st.info("Va dans **🔍 Analyse** pour commencer une nouvelle analyse.")


elif page == "🔍 Analyse":
    st.markdown(
        '<div class="main-title">🔍 Analyse d’une radiographie</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("1️⃣ Informations")

        patient_name = st.text_input(
            "Nom du patient / identifiant",
            placeholder="Ex : Patient 001"
        )

        uploaded_file = st.file_uploader(
            "Importer une radiographie",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            st.subheader("2️⃣ Image importée")
            st.image(image, caption="Radiographie chargée", use_container_width=True)

            if st.button("▶️ Lancer l’analyse", type="primary"):
                with st.spinner("Analyse en cours..."):
                    prediction = predict(model, image)

                risk_level, emoji, card_class = niveau_risque(prediction)

                if prediction >= threshold:
                    result = "Tuberculose suspectée"
                else:
                    result = "Aucune tuberculose détectée"

                final_patient_name = patient_name if patient_name else "Non renseigné"

                save_analysis(
                    patient_name=final_patient_name,
                    prediction=prediction,
                    result=result,
                    risk_level=risk_level,
                    threshold=threshold
                )

                gradcam_image = None

                if gradcam_option:
                    try:
                        img_array = preprocess_image(image)
                        heatmap = make_gradcam_heatmap(img_array, model)
                        gradcam_image = overlay_heatmap(heatmap, image)
                    except Exception as e:
                        st.warning(f"Grad-CAM non disponible : {e}")

                st.session_state.last_result = {
                    "patient_name": final_patient_name,
                    "prediction": prediction,
                    "result": result,
                    "risk_level": risk_level,
                    "threshold": threshold,
                    "emoji": emoji,
                    "card_class": card_class,
                    "image": image,
                    "gradcam": gradcam_image
                }

    with col2:
        st.subheader("3️⃣ Résultat")

        if "last_result" not in st.session_state:
            st.info("Le résultat apparaîtra ici après l’analyse.")
        else:
            res = st.session_state.last_result
            prediction = res["prediction"]
            score_percent = prediction * 100

            st.markdown(f"""
            <div class="card {res["card_class"]}">
                <h2>{res["emoji"]} {res["result"]}</h2>
                <p><b>Patient :</b> {res["patient_name"]}</p>
                <p><b>Score :</b> {prediction:.4f}</p>
                <p><b>Probabilité :</b> {score_percent:.2f} %</p>
                <p><b>Niveau :</b> {res["risk_level"]}</p>
                <p><b>Seuil :</b> {res["threshold"]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Jauge de risque")
            st.plotly_chart(gauge_chart(prediction), use_container_width=True)
            st.progress(min(int(score_percent), 100))

            df = pd.DataFrame({
                "Classe": ["Normal", "Tuberculose"],
                "Probabilité": [1 - prediction, prediction]
            })

            st.markdown("### 📌 Détail des probabilités")
            st.dataframe(
                df.style.format({"Probabilité": "{:.2%}"}),
                use_container_width=True
            )

            if prediction >= res["threshold"]:
                st.error("⚠️ Avis médical recommandé pour confirmation.")
            else:
                st.success("✅ Résultat rassurant selon le modèle.")

            if abs(prediction - res["threshold"]) < 0.05:
                st.warning("⚠️ Résultat proche du seuil. Interprétation prudente nécessaire.")

            if res.get("gradcam") is not None:
                st.markdown("### 🔥 Explication visuelle Grad-CAM")

                g1, g2 = st.columns(2)

                with g1:
                    st.image(
                        res["image"],
                        caption="Radiographie originale",
                        use_container_width=True
                    )

                with g2:
                    st.image(
                        res["gradcam"],
                        caption="Carte de chaleur Grad-CAM",
                        use_container_width=True
                    )

            pdf_buffer = generate_pdf(
                res["patient_name"],
                res["prediction"],
                res["result"],
                res["risk_level"],
                res["threshold"],
                image=res.get("image"),
                gradcam_image=res.get("gradcam")
            )

            st.download_button(
                "📄 Télécharger le rapport PDF",
                data=pdf_buffer,
                file_name=f"rapport_{res['patient_name'].replace(' ', '_')}.pdf",
                mime="application/pdf"
            )


elif page == "👤 Dossier patient":
    st.markdown(
        '<div class="main-title">👤 Dossier patient</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    patients = get_patients()

    if not patients:
        st.info("Aucun patient enregistré pour le moment.")
    else:
        selected_patient = st.selectbox(
            "Sélectionner un patient",
            patients
        )

        patient_rows = get_analyses_by_patient(selected_patient)

        if not patient_rows:
            st.warning("Aucune analyse trouvée pour ce patient.")
        else:
            df_patient = dataframe_from_rows(patient_rows)

            total = len(df_patient)
            positives = len(df_patient[df_patient["Résultat"] == "Tuberculose suspectée"])
            negatives = total - positives
            score_moyen = df_patient["Score"].mean()
            dernier_resultat = df_patient.iloc[0]["Résultat"]

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card("Examens", total, "Nombre d'analyses du patient.")
            with c2:
                metric_card("Suspects", positives, "Analyses positives.")
            with c3:
                metric_card("Normaux", negatives, "Analyses négatives.")
            with c4:
                metric_card("Score moyen", f"{score_moyen:.2f}", "Score moyen TB.")

            st.markdown("### Résumé patient")
            st.write(f"**Patient :** {selected_patient}")
            st.write(f"**Dernier résultat :** {dernier_resultat}")

            st.markdown("### Historique du patient")
            st.dataframe(
                df_patient.style.format({"Score": "{:.4f}", "Seuil": "{:.2f}"}),
                use_container_width=True
            )

            st.markdown("### Évolution des scores")
            st.line_chart(df_patient.sort_values("ID")["Score"])

            csv_patient = df_patient.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Télécharger l’historique du patient en CSV",
                data=csv_patient,
                file_name=f"historique_{selected_patient.replace(' ', '_')}.csv",
                mime="text/csv"
            )

            st.markdown("### Détail des examens")

            for _, row in df_patient.iterrows():
                with st.expander(f"Examen #{row['ID']} — {row['Date']} — {row['Résultat']}"):
                    st.write(f"**Patient :** {row['Patient']}")
                    st.write(f"**Score :** {row['Score']:.4f}")
                    st.write(f"**Probabilité :** {row['Score'] * 100:.2f} %")
                    st.write(f"**Risque :** {row['Risque']}")
                    st.write(f"**Seuil utilisé :** {row['Seuil']:.2f}")
                    st.write(f"**Date :** {row['Date']}")

                    if row["Résultat"] == "Tuberculose suspectée":
                        st.error("⚠️ Suspicion TB selon le modèle.")
                    else:
                        st.success("✅ Résultat rassurant selon le modèle.")



elif page == "📊 Historique":
    st.markdown(
        '<div class="main-title">📊 Historique des analyses</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    rows = get_analyses()

    if not rows:
        st.info("Aucune analyse enregistrée.")
    else:
        df = dataframe_from_rows(rows)

        patients = ["Tous"] + get_patients()
        selected_filter = st.selectbox("Filtrer par patient", patients)

        if selected_filter != "Tous":
            df = df[df["Patient"] == selected_filter]

        st.dataframe(
            df.style.format({"Score": "{:.4f}", "Seuil": "{:.2f}"}),
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Télécharger l’historique CSV",
            data=csv,
            file_name="historique_tb.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Supprimer tout l’historique"):
            delete_all_analyses()
            st.success("Historique supprimé.")
            st.rerun()



elif page == "📈 Statistiques":
    st.markdown(
        '<div class="main-title">📈 Statistiques</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    rows = get_analyses()

    if not rows:
        st.info("Aucune donnée disponible.")
    else:
        df = dataframe_from_rows(rows)

        total = len(df)
        positives = len(df[df["Résultat"] == "Tuberculose suspectée"])
        negatives = total - positives
        score_moyen = df["Score"].mean()
        patients_count = df["Patient"].nunique()

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            metric_card("Total", total, "Analyses réalisées.")
        with c2:
            metric_card("Patients", patients_count, "Patients différents.")
        with c3:
            metric_card("Suspects", positives, "Cas positifs.")
        with c4:
            metric_card("Normaux", negatives, "Cas négatifs.")
        with c5:
            metric_card("Score moyen", f"{score_moyen:.2f}", "Moyenne des scores.")

        st.markdown("### Répartition des résultats")
        st.bar_chart(df["Résultat"].value_counts())

        st.markdown("### Répartition par niveau de risque")
        st.bar_chart(df["Risque"].value_counts())

        st.markdown("### Distribution des scores")
        st.line_chart(df.sort_values("ID")["Score"])

        st.markdown("### Nombre d’analyses par patient")
        patient_counts = df["Patient"].value_counts()
        st.bar_chart(patient_counts)



elif page == "ℹ️ À propos":
    st.markdown(
        '<div class="main-title">ℹ️ À propos</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.write("""
    Cette application est un prototype d’aide à la détection de la tuberculose
    à partir de radiographies thoraciques.
    """)

    st.markdown("### Fonctionnalités")

    st.markdown("""
    - Analyse automatique d’une radiographie ;
    - Score de probabilité ;
    - Jauge de risque ;
    - Grad-CAM ;
    - Rapport PDF avec image originale et carte Grad-CAM ;
    - Historique sauvegardé dans SQLite ;
    - Dossier patient avancé ;
    - Filtrage par patient ;
    - Export CSV ;
    - Statistiques globales.
    """)

    st.error(
        "Cette application ne remplace jamais un médecin, un radiologue "
        "ou un diagnostic médical officiel."
    )