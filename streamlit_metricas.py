"""
streamlit_metricas.py
Calculadora interactiva de métricas para el pipeline ASR-RAG
Detección automatizada de cumplimiento en banca privada colombiana

Instalación de dependencias:
    pip install streamlit pandas numpy scikit-learn plotly

Ejecución:
    streamlit run streamlit_metricas.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Métricas ASR-RAG | Cumplimiento Banca Privada",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS adicional
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f0f6ff;
    border-left: 4px solid #1e508c;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 8px 0;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #1e508c; }
  .metric-label { font-size: 0.85rem; color: #555; }
  .formula-box {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    color: #000;
    padding: 10px 14px;
    font-family: monospace;
    font-size: 0.92rem;
    margin: 6px 0 12px 0;
  }
  .section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e508c;
    margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/80/1e508c/combo-chart.png", width=60)
    st.title("ASR-RAG\nCalculadora de Métricas")
    st.markdown("---")
    st.markdown("""
    **Pipeline evaluado**
    - 🎙️ ASR: WhisperX + LoRA
    - 🔍 RAG: BM25 + HNSW + Cohere Rerank
    - ⚖️ Adjudicador: Claude Sonnet 4.6
    - 📐 G-Eval: Meta Llama 3.1 70B
    """)
    st.markdown("---")
    st.markdown("**Formatos de entrada esperados**")
    st.markdown("""
    - `test_asr_pairs.csv`
    - `test_financial_terms.txt`
    - `test_diarization.csv`
    - `test_role_attribution.csv`
    - `test_rag_predictions.csv`
    - `test_ragas_scores.csv`
    - `test_retrieval.csv`
    """)
    st.markdown("---")
    st.caption("Tesis de maestría · EAFIT · 2025")


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = ""):
    delta_html = f"<br><span style='font-size:0.8rem;color:#28a745'>{delta}</span>" if delta else ""
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-value">{value}</div>
              <div class="metric-label">{label}{delta_html}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def formula_box(text: str):
    st.markdown(f'<div class="formula-box">{text}</div>', unsafe_allow_html=True)


def load_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def load_txt_lines(uploaded) -> list:
    content = uploaded.read().decode("utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]


# ─────────────────────────────────────────────
# WER / FTWER  —  cálculo por distancia de edición
# ─────────────────────────────────────────────

def levenshtein_words(ref: list, hyp: list):
    """Distancia de Levenshtein a nivel de palabras.
    Retorna (S, D, I, N) donde N = len(ref).
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],   # deletion
                                    dp[i][j - 1],   # insertion
                                    dp[i - 1][j - 1])  # substitution
    # Count operations via backtrack
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1; j -= 1
        else:
            D += 1; i -= 1
    return S, D, I, n


def compute_rrf(df: pd.DataFrame, k: int = 60) -> pd.Series:
    """Compute RRF score from severity_score, faithfulness, context_coverage."""
    rank_sev   = df['severity_score'].rank(ascending=False, method='first').astype(int)
    rank_faith = df['faithfulness'].rank(ascending=False, method='first').astype(int)
    rank_ctx   = df['context_coverage'].rank(ascending=False, method='first').astype(int)
    return (1 / (k + rank_sev) + 1 / (k + rank_faith) + 1 / (k + rank_ctx))


def compute_gain_lift(y_true: pd.Series, scores: pd.Series, D: int = 10):
    """
    Compute cumulative gain and lift curves.
    Returns DataFrame with columns: decile, pct_reviewed, n_inspected,
                                    cum_violations, cum_gain, lift.
    """
    N = len(y_true)
    total_pos = y_true.sum()
    order = scores.argsort()[::-1].values  # descending

    rows = []
    for d in range(1, D + 1):
        top_n = int(d * N / D)
        cum_pos = int(y_true.iloc[order[:top_n]].sum())
        cum_gain = cum_pos / total_pos if total_pos > 0 else 0
        lift = cum_gain / (d / D)
        rows.append({
            "Decil": d,
            "% revisado": d * 10,
            "Pares inspeccionados": top_n,
            "Infracciones capturadas": cum_pos,
            "GananciaAcum": round(cum_gain, 4),
            "Lift": round(lift, 4),
        })
    return pd.DataFrame(rows)


def compute_wer_ftwer(asr_df: pd.DataFrame, fin_terms: set):
    total_S = total_D = total_I = total_N = 0
    total_fin_errors = total_fin_ref = 0
    total_nonfin_errors = total_nonfin_ref = 0

    for _, row in asr_df.iterrows():
        ref_words = str(row["reference"]).split()
        hyp_words = str(row["hypothesis"]).split()
        S, D, I, N = levenshtein_words(ref_words, hyp_words)
        total_S += S; total_D += D; total_I += I; total_N += N

        # Align pairs for FTWER (zip only matched positions — simple greedy)
        min_len = min(len(ref_words), len(hyp_words))
        for k in range(min_len):
            r, h = ref_words[k], hyp_words[k]
            is_fin = r in fin_terms
            if is_fin:
                total_fin_ref += 1
                if r != h:
                    total_fin_errors += 1
            else:
                total_nonfin_ref += 1
                if r != h:
                    total_nonfin_errors += 1
        # Remaining reference words (deletions) count as errors
        for k in range(min_len, len(ref_words)):
            r = ref_words[k]
            if r in fin_terms:
                total_fin_ref += 1; total_fin_errors += 1
            else:
                total_nonfin_ref += 1; total_nonfin_errors += 1

    wer = (total_S + total_D + total_I) / total_N * 100 if total_N > 0 else 0
    ftwer = total_fin_errors / total_fin_ref * 100 if total_fin_ref > 0 else 0
    return {
        "WER (%)": round(wer, 2),
        "FTWER (%)": round(ftwer, 2),
        "Total palabras ref": total_N,
        "Errores totales (S+D+I)": total_S + total_D + total_I,
        "Sustituciones": total_S,
        "Eliminaciones": total_D,
        "Inserciones": total_I,
        "Términos fin. ref": total_fin_ref,
        "Errores en términos fin.": total_fin_errors,
    }


# ─────────────────────────────────────────────
# TÍTULO PRINCIPAL
# ─────────────────────────────────────────────
st.title("📊 Calculadora de Métricas — Pipeline ASR-RAG")
st.markdown(
    "Cargue los archivos de evaluación y obtenga todas las métricas del sistema de "
    "detección automatizada de cumplimiento en llamadas de banca privada colombiana."
)
st.markdown("---")

# ─────────────────────────────────────────────
# PESTAÑAS
# ─────────────────────────────────────────────
tab_asr, tab_rag, tab_ragas, tab_ret = st.tabs([
    "🎙️ ASR — Transcripción y Diarización",
    "🔍 RAG — Detección de Cumplimiento",
    "📋 RAGAS — Calidad de Generación",
    "🔎 Recuperación — Recall@K",
])


# ══════════════════════════════════════════════
#  TAB 1 — ASR
# ══════════════════════════════════════════════
with tab_asr:
    st.header("🎙️ Métricas ASR — Transcripción y Diarización")

    col_upload, col_info = st.columns([1, 1])

    with col_upload:
        st.markdown("### Archivos de entrada")

        f_asr = st.file_uploader(
            "1. Pares ASR (referencia / hipótesis)",
            type=["csv"],
            key="asr_pairs",
            help="Columnas: utterance_id, reference, hypothesis",
        )
        f_terms = st.file_uploader(
            "2. Términos financieros",
            type=["txt"],
            key="fin_terms",
            help="Un término por línea",
        )
        f_diar = st.file_uploader(
            "3. Datos de diarización",
            type=["csv"],
            key="diar",
            help="Columnas: call_id, total_ref_speech_s, false_alarm_s, missed_speech_s, speaker_confusion_s",
        )
        f_role = st.file_uploader(
            "4. Atribución de roles",
            type=["csv"],
            key="role",
            help="Columnas: segment_id, true_role, predicted_role",
        )

    with col_info:
        st.markdown("### Fórmulas de referencia")
        formula_box("WER = (S + D + I) / N")
        formula_box("FTWER = Σ(w_t · 1[t≠t̂]) / Σ(w_t),  w_t > 1 si t ∈ T_F")
        formula_box("DER = (FA + MISS + SC) / T_ref")
        formula_box("SCR = SC / T_ref")
        formula_box("RAA = Σ 1[ρ̂(s) = ρ(s)] / |S|")

        st.markdown(
            "**Nota:** FTWER se calcula sobre los términos de la lista de términos financieros. "
            "Si FTWER < WER, el modelo es proporcionalmente más preciso sobre vocabulario financiero."
        )

    st.markdown("---")

    # ── Cálculo WER / FTWER ──
    if f_asr and f_terms:
        asr_df = load_csv(f_asr)
        fin_terms_list = load_txt_lines(f_terms)
        fin_terms_set = set(fin_terms_list)

        with st.spinner("Calculando WER y FTWER (distancia de edición por enunciado)…"):
            wer_results = compute_wer_ftwer(asr_df, fin_terms_set)

        st.markdown("#### Resultados WER y FTWER")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("WER", f"{wer_results['WER (%)']:.1f}%", "↓ menor es mejor")
        with c2:
            metric_card("FTWER", f"{wer_results['FTWER (%)']:.1f}%", "↓ menor es mejor")
        with c3:
            metric_card("Palabras de referencia", f"{wer_results['Total palabras ref']:,}")
        with c4:
            metric_card("Términos financieros", f"{wer_results['Términos fin. ref']:,}")

        with st.expander("Desglose de errores ASR"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.dataframe(
                    pd.DataFrame([{
                        "Sustituciones (S)": wer_results["Sustituciones"],
                        "Eliminaciones (D)": wer_results["Eliminaciones"],
                        "Inserciones (I)": wer_results["Inserciones"],
                        "Total errores": wer_results["Errores totales (S+D+I)"],
                    }]).T.rename(columns={0: "Valor"}),
                    use_container_width=True,
                )
            with col_b:
                st.dataframe(
                    pd.DataFrame([{
                        "Términos fin. en referencia": wer_results["Términos fin. ref"],
                        "Errores en términos fin.": wer_results["Errores en términos fin."],
                        "Tasa error términos fin.": f"{wer_results['FTWER (%)']:.2f}%",
                    }]).T.rename(columns={0: "Valor"}),
                    use_container_width=True,
                )

        with st.expander("Vista previa — pares ASR (primeras 5 filas)"):
            st.dataframe(asr_df.head(), use_container_width=True)

    elif f_asr or f_terms:
        st.info("Cargue tanto el archivo de pares ASR como la lista de términos financieros para calcular WER y FTWER.")

    # ── Cálculo DER / SCR ──
    if f_diar:
        diar_df = load_csv(f_diar)
        required_cols = {"total_ref_speech_s", "false_alarm_s", "missed_speech_s", "speaker_confusion_s"}
        if required_cols.issubset(diar_df.columns):
            T_ref  = diar_df["total_ref_speech_s"].sum()
            FA     = diar_df["false_alarm_s"].sum()
            MISS   = diar_df["missed_speech_s"].sum()
            SC     = diar_df["speaker_confusion_s"].sum()
            DER    = (FA + MISS + SC) / T_ref * 100
            SCR    = SC / T_ref * 100

            st.markdown("#### Resultados DER y SCR")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("DER", f"{DER:.1f}%", "↓ menor es mejor")
            with c2:
                metric_card("SCR", f"{SCR:.1f}%", "↓ menor es mejor")
            with c3:
                metric_card("Tiempo ref. total (s)", f"{T_ref:,.0f}")
            with c4:
                metric_card("SC / DER", f"{SC/(FA+MISS+SC)*100:.1f}%", "% del error por confusión")

            with st.expander("Desglose de componentes del DER"):
                fig = go.Figure(go.Bar(
                    x=["Falsa alarma (FA)", "Speech no det. (MISS)", "Confusión (SC)"],
                    y=[FA / T_ref * 100, MISS / T_ref * 100, SC / T_ref * 100],
                    marker_color=["#e07b39", "#e0b039", "#1e508c"],
                    text=[f"{v:.2f}%" for v in [FA/T_ref*100, MISS/T_ref*100, SC/T_ref*100]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Componentes del DER (% del tiempo de referencia)",
                    yaxis_title="% tiempo de referencia",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"El archivo de diarización debe tener las columnas: {required_cols}")

    # ── Cálculo RAA ──
    if f_role:
        role_df = load_csv(f_role)
        if {"true_role", "predicted_role"}.issubset(role_df.columns):
            raa = (role_df["true_role"] == role_df["predicted_role"]).mean() * 100
            n_correct = (role_df["true_role"] == role_df["predicted_role"]).sum()

            st.markdown("#### Resultado RAA")
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("RAA", f"{raa:.1f}%", "↑ mayor es mejor")
            with c2:
                metric_card("Segmentos correctos", f"{n_correct:,}")
            with c3:
                metric_card("Total segmentos", f"{len(role_df):,}")

            with st.expander("Matriz de confusión — atribución de roles"):
                roles = sorted(role_df["true_role"].unique())
                cm = confusion_matrix(role_df["true_role"], role_df["predicted_role"], labels=roles)
                fig = px.imshow(
                    cm,
                    x=[f"Pred: {r}" for r in roles],
                    y=[f"Real: {r}" for r in roles],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title="Matriz de confusión — atribución de roles",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("El archivo de atribución de roles debe tener columnas: true_role, predicted_role")


# ══════════════════════════════════════════════
#  TAB 2 — RAG DETECCIÓN
# ══════════════════════════════════════════════
with tab_rag:
    st.header("🔍 RAG — Métricas de Detección de Cumplimiento")

    col_upload2, col_info2 = st.columns([1, 1])

    with col_upload2:
        st.markdown("### Archivo de entrada")
        f_rag = st.file_uploader(
            "Predicciones RAG por llamada y regla",
            type=["csv"],
            key="rag_pred",
            help="Columnas: call_id, rule_id, y_true, y_pred, severity_score, "
                 "routed_to_human. Opcionales para RRF: faithfulness, "
                 "context_coverage, rrf_score.",
        )

    with col_info2:
        st.markdown("### Fórmulas de referencia")
        formula_box("F1_r = 2·TP_r / (2·TP_r + FP_r + FN_r)")
        formula_box("F1_macro = (1/R) · Σ F1_r")
        formula_box("F1_weighted = Σ(sup_r · F1_r) / Σ(sup_r)")
        formula_box("AUROC = P(ŝ_i > ŝ_j | y_i=1, y_j=0)")
        formula_box("HRR = #{pares enrutados} / #{pares totales}")
        formula_box("RRF(i) = Σ_l 1/(60 + rank_l(i))  [l ∈ {G-Eval, Faith, CtxCov}]")
        formula_box("GananciaAcum(d) = Σ_{top dN/D} y_π(i)  /  n₊")
        formula_box("Lift(d) = GananciaAcum(d) / (d/D)")

    st.markdown("---")

    if f_rag:
        rag_df = load_csv(f_rag)
        required = {"rule_id", "y_true", "y_pred", "severity_score", "routed_to_human"}
        rrf_cols = {"faithfulness", "context_coverage", "severity_score"}
        has_rrf_cols = rrf_cols.issubset(rag_df.columns)
        if not required.issubset(rag_df.columns):
            st.error(f"Columnas requeridas: {required}")
        else:
            # Per-rule F1
            rule_ids = sorted(rag_df["rule_id"].unique())
            f1_list, prec_list, rec_list, sup_list = [], [], [], []

            for rid in rule_ids:
                sub = rag_df[rag_df["rule_id"] == rid]
                f1_list.append(f1_score(sub["y_true"], sub["y_pred"], zero_division=0))
                prec_list.append(precision_score(sub["y_true"], sub["y_pred"], zero_division=0))
                rec_list.append(recall_score(sub["y_true"], sub["y_pred"], zero_division=0))
                sup_list.append(int(sub["y_true"].sum())) # TP + TN 

            macro_f1    = float(np.mean(f1_list))
            weighted_f1 = float(np.average(f1_list, weights=sup_list))
            roc_auc       = roc_auc_score(rag_df["y_true"], rag_df["y_pred"])
            hrr         = rag_df["routed_to_human"].mean() * 100

            # ── Métricas globales ──
            st.markdown("#### Resultados globales")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("F₁ macro", f"{macro_f1:.4f}", "↑ mayor es mejor")
            with c2:
                metric_card("F₁ ponderado", f"{weighted_f1:.4f}", "↑ mayor es mejor")
            with c3:
                metric_card("ROC AUC", f"{roc_auc:.4f}", "↑ mayor es mejor")
            with c4:
                metric_card("Revisión humana", f"{hrr:.2f}%", "↓ menor es mejor")

            # ── Tabla per-rule ──
            st.markdown("#### Métricas por regla de cumplimiento")
            rule_table = pd.DataFrame({
                "Regla": [f"R{r}" for r in rule_ids],
                "Soporte (infracc. reales)": sup_list,
                "Precisión": [round(p, 4) for p in prec_list],
                "Recall": [round(r, 4) for r in rec_list],
                "F₁": [round(f, 4) for f in f1_list],
            })
            st.dataframe(rule_table, use_container_width=True, hide_index=True)

            # ── Gráfica F1 por regla ──
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Bar(
                x=[f"R{r}" for r in rule_ids],
                y=[round(f, 4) for f in f1_list],
                name="F₁ por regla",
                marker_color="#1e508c",
                text=[f"{f:.3f}" for f in f1_list],
                textposition="outside",
            ))
            fig_f1.add_hline(y=macro_f1, line_dash="dash", line_color="orange",
                             annotation_text=f"F₁ macro = {macro_f1:.4f}")
            fig_f1.add_hline(y=weighted_f1, line_dash="dot", line_color="green",
                             annotation_text=f"F₁ weighted = {weighted_f1:.4f}")
            fig_f1.update_layout(
                title="F₁ por regla de cumplimiento",
                yaxis=dict(title="F₁", range=[0, 1.05]),
                xaxis_title="Regla",
                height=380,
            )
            st.plotly_chart(fig_f1, use_container_width=True)

            # ── Curva ROC ──
            with st.expander("Curva ROC sobre puntuaciones de severidad G-Eval"):
                fpr, tpr, _ = roc_curve(rag_df["y_true"], rag_df["severity_score"])
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                             name=f"ROC (AUROC = {auroc:.4f})",
                                             line=dict(color="#1e508c", width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                             line=dict(dash="dash", color="gray"),
                                             name="Aleatorio (0.5)"))
                fig_roc.update_layout(
                    xaxis_title="Tasa de Falsos Positivos",
                    yaxis_title="Tasa de Verdaderos Positivos",
                    title="Curva ROC — Severidad G-Eval vs. Etiqueta binaria",
                    height=400,
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            # ── Distribución de puntuaciones de severidad ──
            with st.expander("Distribución de puntuaciones de severidad"):
                pos_scores = rag_df[rag_df["y_true"] == 1]["severity_score"]
                neg_scores = rag_df[rag_df["y_true"] == 0]["severity_score"]
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=pos_scores, name="Infracción (y=1)",
                                                opacity=0.75, marker_color="#e03030",
                                                nbinsx=30))
                fig_hist.add_trace(go.Histogram(x=neg_scores, name="Cumple (y=0)",
                                                opacity=0.75, marker_color="#1e508c",
                                                nbinsx=30))
                fig_hist.update_layout(
                    barmode="overlay",
                    title="Distribución de puntuaciones de severidad G-Eval",
                    xaxis_title="Puntuación de severidad",
                    yaxis_title="Frecuencia",
                    height=380,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # ── RRF + Gain/Lift ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📈 Curvas de Ganancia Acumulada y Lift — Ranking RRF")
            st.markdown(
                "Las curvas de ganancia y lift miden cuántas infracciones captura el sistema "
                "al revisar únicamente la fracción de mayor riesgo del ranking RRF "
                "(fusión de G-Eval severidad + Fidelidad RAGAS + Cobertura de Contexto RAGAS)."
            )

            # Compute  RRF score
            if has_rrf_cols:
                rrf_scores = compute_rrf(rag_df)
                auroc_rrf = roc_auc_score(rag_df["y_true"], rrf_scores)
                st.success(
                    f"RRF calculado a partir de `severity_score`, `faithfulness` y "
                    f"`context_coverage`. AUROC RRF = **{auroc_rrf:.4f}** "
                    f"(vs. AUROC severidad = {auroc:.4f})"
                )
            else:
                rrf_scores = rag_df["severity_score"]
                missing_cols = rrf_cols - set(rag_df.columns)
                st.warning(
                    "No se encontraron columnas completas para calcular RRF (faltan: "f"{missing_cols}). Se usará `severity_score` como ranking de riesgo."
                )

            gain_lift_df = compute_gain_lift(rag_df["y_true"].reset_index(drop=True),
                                             rrf_scores.reset_index(drop=True), D=10)

            # KPIs
            lift1 = gain_lift_df.loc[0, "Lift"]
            gain1 = gain_lift_df.loc[0, "GananciaAcum"]
            d_90pct = gain_lift_df[gain_lift_df["GananciaAcum"] >= 0.90].iloc[0]["% revisado"]

            c_l1, c_l2, c_l3 = st.columns(3)
            with c_l1:
                metric_card("Lift@10%", f"{lift1:.2f}×",
                            f"Captura {gain1*100:.1f}% de infracciones")
            with c_l2:
                metric_card("90% infracciones cubiertas al", f"{int(d_90pct)}% revisado",
                            "del ranking RRF")
            with c_l3:
                total_viol = int(rag_df["y_true"].sum())
                metric_card("Total infracciones reales", f"{total_viol:,}",
                            f"de {len(rag_df):,} pares")

            # Dual chart: Gain + Lift
            fig_gl = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Curva de Ganancia Acumulada", "Curva de Lift"),
            )

            pct_vals   = gain_lift_df["% revisado"].tolist()
            gain_vals  = gain_lift_df["GananciaAcum"].tolist()
            lift_vals  = gain_lift_df["Lift"].tolist()

            # Ganancia acumulada
            fig_gl.add_trace(go.Scatter(
                x=[0] + pct_vals, y=[0] + gain_vals,
                mode="lines+markers",
                name="RRF (modelo)",
                line=dict(color="#1e508c", width=2.5),
                marker=dict(size=7),
            ), row=1, col=1)
            fig_gl.add_trace(go.Scatter(
                x=[0, 100], y=[0, 1],
                mode="lines",
                name="Aleatorio",
                line=dict(dash="dash", color="gray", width=1.5),
                showlegend=True,
            ), row=1, col=1)

            # Lift
            fig_gl.add_trace(go.Scatter(
                x=pct_vals, y=lift_vals,
                mode="lines+markers",
                name="Lift RRF",
                line=dict(color="#1e508c", width=2.5),
                marker=dict(size=7),
                showlegend=False,
            ), row=1, col=2)
            fig_gl.add_hline(y=1, line_dash="dash", line_color="gray",
                             annotation_text="Base aleatoria (1×)", row=1, col=2)

            fig_gl.update_xaxes(title_text="% del ranking inspeccionado", row=1, col=1)
            fig_gl.update_xaxes(title_text="% del ranking inspeccionado", row=1, col=2)
            fig_gl.update_yaxes(title_text="Fracción de infracciones capturadas", row=1, col=1,
                                range=[0, 1.05])
            fig_gl.update_yaxes(title_text="Factor de mejora vs. aleatorio", row=1, col=2,
                                range=[0, max(lift_vals) * 1.1])
            fig_gl.update_layout(height=420, legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig_gl, use_container_width=True)

            # Gain/Lift table
            with st.expander("Tabla completa de Ganancia Acumulada y Lift por decil"):
                display_df = gain_lift_df.copy()
                display_df.columns = [
                    "Decil", "% revisado", "Pares inspeccionados",
                    "Infracc. capturadas", "GananciaAcum", "Lift"
                ]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Download button
                csv_bytes = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Descargar tabla (CSV)",
                    data=csv_bytes,
                    file_name="gain_lift_rrf.csv",
                    mime="text/csv",
                )

    else:
        st.info("Cargue el archivo de predicciones RAG para calcular las métricas de detección de cumplimiento.")


# ══════════════════════════════════════════════
#  TAB 3 — RAGAS
# ══════════════════════════════════════════════
with tab_ragas:
    st.header("📋 RAGAS — Métricas de Calidad de Generación")

    col_upload3, col_info3 = st.columns([1, 1])

    with col_upload3:
        st.markdown("### Archivo de entrada")
        f_ragas = st.file_uploader(
            "Puntuaciones RAGAS por muestra",
            type=["csv"],
            key="ragas_scores",
            help="Columnas: sample_id, faithfulness, context_coverage",
        )

    with col_info3:
        st.markdown("### Fórmulas de referencia")
        formula_box(
            "Faith = (1/n) · Σ_{muestra} #{afirmaciones respaldadas} / #{afirmaciones totales}"
        )
        formula_box(
            "CtxCov = (1/n) · Σ_{muestra} |R(q) ∩ C| / |R(q)|"
        )
        st.markdown("""
        **Faithfulness** mide si las afirmaciones del hallazgo generado están ancladas
        en los fragmentos recuperados (sin alucinación).

        **Context Coverage** mide si el retriever capturó la información normativa
        necesaria para emitir el veredicto.
        """)

    st.markdown("---")

    if f_ragas:
        ragas_df = load_csv(f_ragas)
        required_r = {"faithfulness", "context_coverage"}
        if not required_r.issubset(ragas_df.columns):
            st.error(f"Columnas requeridas: {required_r}")
        else:
            faith_mean = ragas_df["faithfulness"].mean()
            ctx_mean   = ragas_df["context_coverage"].mean()
            faith_std  = ragas_df["faithfulness"].std()
            ctx_std    = ragas_df["context_coverage"].std()
            n_samples  = len(ragas_df)

            # IC 95% con normal (muestra grande)
            z = 1.96
            faith_ci_lo = faith_mean - z * faith_std / np.sqrt(n_samples)
            faith_ci_hi = faith_mean + z * faith_std / np.sqrt(n_samples)
            ctx_ci_lo   = ctx_mean - z * ctx_std / np.sqrt(n_samples)
            ctx_ci_hi   = ctx_mean + z * ctx_std / np.sqrt(n_samples)

            st.markdown("#### Resultados RAGAS")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Fidelidad RAGAS", f"{faith_mean:.4f}", "↑ mayor es mejor")
            with c2:
                metric_card("Cobertura de Contexto", f"{ctx_mean:.4f}", "↑ mayor es mejor")
            with c3:
                metric_card("IC 95% Fidelidad", f"[{faith_ci_lo:.3f}, {faith_ci_hi:.3f}]")
            with c4:
                metric_card("IC 95% Cobertura", f"[{ctx_ci_lo:.3f}, {ctx_ci_hi:.3f}]")

            st.markdown(f"*Estadísticas sobre n = {n_samples} muestras.*")

            # ── Distribuciones ──
            col_f, col_c = st.columns(2)
            with col_f:
                fig_faith = px.histogram(
                    ragas_df, x="faithfulness", nbins=20,
                    title=f"Distribución de Fidelidad RAGAS (media={faith_mean:.4f})",
                    color_discrete_sequence=["#1e508c"],
                )
                fig_faith.add_vline(x=faith_mean, line_dash="dash", line_color="orange",
                                    annotation_text=f"μ={faith_mean:.3f}")
                fig_faith.update_layout(height=350)
                st.plotly_chart(fig_faith, use_container_width=True)

            with col_c:
                fig_ctx = px.histogram(
                    ragas_df, x="context_coverage", nbins=20,
                    title=f"Distribución de Cobertura de Contexto (media={ctx_mean:.4f})",
                    color_discrete_sequence=["#28a745"],
                )
                fig_ctx.add_vline(x=ctx_mean, line_dash="dash", line_color="orange",
                                  annotation_text=f"μ={ctx_mean:.3f}")
                fig_ctx.update_layout(height=350)
                st.plotly_chart(fig_ctx, use_container_width=True)

            with st.expander("Scatter: Fidelidad vs. Cobertura de Contexto"):
                fig_sc = px.scatter(
                    ragas_df, x="context_coverage", y="faithfulness",
                    opacity=0.6,
                    title="Fidelidad vs. Cobertura de Contexto por muestra",
                    labels={"context_coverage": "Cobertura de Contexto",
                            "faithfulness": "Fidelidad"},
                    color_discrete_sequence=["#1e508c"],
                )
                fig_sc.update_layout(height=400)
                st.plotly_chart(fig_sc, use_container_width=True)

            with st.expander("Vista previa de datos RAGAS"):
                st.dataframe(ragas_df.describe().T, use_container_width=True)
    else:
        st.info("Cargue el archivo de puntuaciones RAGAS para calcular Fidelidad y Cobertura de Contexto.")


# ══════════════════════════════════════════════
#  TAB 4 — RECUPERACIÓN
# ══════════════════════════════════════════════
with tab_ret:
    st.header("🔎 Métricas de Recuperación — Recall@K")

    col_upload4, col_info4 = st.columns([1, 1])

    with col_upload4:
        st.markdown("### Archivo de entrada")
        f_ret = st.file_uploader(
            "Resultados de recuperación por consulta",
            type=["csv"],
            key="retrieval",
            help="Columnas: query_id, k, relevant_in_top_k (1 si hay al menos un fragmento relevante en top-K, 0 si no)",
        )

    with col_info4:
        st.markdown("### Fórmulas de referencia")
        formula_box("R@K = (1/|Q|) · Σ_{q ∈ Q}  1[R(q) ∩ top-K(q) ≠ ∅]")
        st.markdown("""
        **R@K** indica la proporción de consultas de auditoría para las cuales
        el retriever devuelve al menos un fragmento relevante entre los $K$ mejores resultados.

        El sistema usa $K = 10$ con el índice HNSW de OpenSearch (parámetros: $m = 16$,
        $ef_{\\text{construction}} = 128$).
        """)

    st.markdown("---")

    if f_ret:
        ret_df = load_csv(f_ret)
        required_ret = {"relevant_in_top_k"}
        if not required_ret.issubset(ret_df.columns):
            st.error("Columnas requeridas: query_id, k, relevant_in_top_k")
        else:
            # Compute per-K if multiple K values exist
            k_vals = sorted(ret_df["k"].unique()) if "k" in ret_df.columns else [10]
            recall_global = ret_df["relevant_in_top_k"].mean()
            n_queries     = len(ret_df)
            n_relevant    = int(ret_df["relevant_in_top_k"].sum())

            st.markdown("#### Resultados Recall@K")
            c1, c2, c3 = st.columns(3)
            with c1:
                k_label = f"Recall@{k_vals[0]}" if len(k_vals) == 1 else "Recall@K (global)"
                metric_card(k_label, f"{recall_global:.4f}", "↑ mayor es mejor")
            with c2:
                metric_card("Consultas con fragmento relevante", f"{n_relevant}/{n_queries}")
            with c3:
                metric_card("Consultas sin fragmento relevante", f"{n_queries - n_relevant}/{n_queries}")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recall_global * 100,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": f"Recall@{k_vals[0] if len(k_vals)==1 else 'K'}", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1e508c"},
                    "steps": [
                        {"range": [0, 70],  "color": "#fde8e8"},
                        {"range": [70, 85], "color": "#fdf3cd"},
                        {"range": [85, 100], "color": "#d4edda"},
                    ],
                    "threshold": {
                        "line": {"color": "orange", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Si hay columna k con múltiples valores, mostrar curva Recall@K
            if "k" in ret_df.columns and len(k_vals) > 1:
                recall_per_k = ret_df.groupby("k")["relevant_in_top_k"].mean().reset_index()
                fig_rk = go.Figure()
                fig_rk.add_trace(go.Scatter(
                    x=recall_per_k["k"], y=recall_per_k["relevant_in_top_k"],
                    mode="lines+markers",
                    line=dict(color="#1e508c", width=2),
                    marker=dict(size=8),
                    name="Recall@K",
                ))
                fig_rk.update_layout(
                    title="Curva Recall@K",
                    xaxis_title="K (número de documentos recuperados)",
                    yaxis_title="Recall@K",
                    yaxis=dict(range=[0, 1.05]),
                    height=380,
                )
                st.plotly_chart(fig_rk, use_container_width=True)

            with st.expander("Vista previa de datos de recuperación"):
                st.dataframe(ret_df.head(20), use_container_width=True)
    else:
        st.info("Cargue el archivo de resultados de recuperación para calcular Recall@K.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.82rem'>"
    "Pipeline ASR-RAG · Detección de cumplimiento regulatorio · Banca privada colombiana · "
    "WhisperX + pyannote + OpenSearch + Claude Sonnet 4.6"
    "</div>",
    unsafe_allow_html=True,
)
