"""
============================================================================
APPLICATION STREAMLIT — ANALYSE DE SURVIE : CONTRACEPTION MODERNE
EDS Cameroun 2018
Auteure : Ariane Kamguen Yolong Sonita
Superviseur : Pr. Nguefack Tsague Georges

CHARGEMENT DES MODÈLES :
  Les fichiers cox_model.pkl et rsf_model.pkl doivent être produits par le
  script ml_survie_contraception.py et placés dans le même dossier que ce
  fichier app.py.  Aucun téléchargement GitHub n'est nécessaire.

CHARGEMENT DES DONNÉES :
  Le fichier donnees_contraception_survie1.csv doit se trouver dans le même
  dossier.  Si l'utilisateur ne charge pas de fichier via la sidebar,
  l'application lit automatiquement ce CSV local.
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io, pickle, os
import warnings
warnings.filterwarnings("ignore")

from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================
st.set_page_config(
    page_title="Survie — Contraception Moderne · EDS Cameroun 2018",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Style CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 1.8rem; font-weight: 700; color: #1B4F72;
    border-bottom: 3px solid #2A9D8F; padding-bottom: 8px; margin-bottom: 0;
  }
  .subtitle { font-size: 1rem; color: #555; margin-top: 4px; }
  .metric-card {
    background: #f0f4f8; border-radius: 8px; padding: 14px 18px;
    border-left: 4px solid #2A9D8F; margin-bottom: 8px;
  }
  .metric-val { font-size: 1.6rem; font-weight: 700; color: #1B4F72; }
  .metric-lbl { font-size: 0.85rem; color: #666; }
  .section-header {
    background: linear-gradient(90deg,#1B4F72,#2A9D8F);
    color: white; padding: 7px 14px; border-radius: 5px;
    font-weight: 600; margin: 18px 0 10px;
  }
  .info-box {
    background: #e8f4f8; border: 1px solid #a8d8ea;
    border-radius: 6px; padding: 10px 14px; font-size: 0.9rem;
  }
  .success-box {
    background: #e8f8f0; border: 1px solid #a8e6c3;
    border-radius: 6px; padding: 10px 14px; font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTES
# ============================================================================
FEATURES = [
    'groupe_age', 'statut_matrimonial', 'niveau_instruction',
    'quintile_richesse', 'milieu_residence', 'region', 'religion_code',
    'ecoute_radio', 'vision_television', 'lecture_journaux',
    'nombre_enfants_vivants', 'nombre_naissances', 'occupation', 'type_union'
]
LABELS = {
    'groupe_age'              : "Groupe d'âge",
    'statut_matrimonial'      : "Statut matrimonial",
    'niveau_instruction'      : "Niveau instruction",
    'quintile_richesse'       : "Quintile richesse",
    'milieu_residence'        : "Milieu résidence",
    'region'                  : "Région",
    'religion_code'           : "Religion",
    'ecoute_radio'            : "Écoute radio",
    'vision_television'       : "Télévision",
    'lecture_journaux'        : "Journaux",
    'nombre_enfants_vivants'  : "Nb enfants vivants",
    'nombre_naissances'       : "Nb naissances",
    'occupation'              : "Occupation",
    'type_union'              : "Type d'union",
}

# Chemin du dossier de l'application (même dossier que app.py)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_COX  = os.path.join(APP_DIR, "cox_model.pkl")
PATH_RSF  = os.path.join(APP_DIR, "rsf_model.pkl")
PATH_DATA = os.path.join(APP_DIR, "donnees_contraception_survie1.csv")

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_resource(show_spinner=False)
def charger_modele_local(chemin: str, nom: str) -> dict | None:
    """Charge un modèle .pkl depuis le disque local (même dossier que app.py)."""
    if not os.path.exists(chemin):
        st.error(
            f"❌ Fichier introuvable : **{chemin}**\n\n"
            f"Assurez-vous d'avoir exécuté `ml_survie_contraception.py` pour "
            f"générer `{os.path.basename(chemin)}` dans le même dossier que `app.py`."
        )
        return None
    try:
        with open(chemin, "rb") as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de {nom} : {e}")
        return None


@st.cache_data(show_spinner=False)
def charger_donnees_local() -> pd.DataFrame | None:
    """Charge le CSV local (même dossier que app.py)."""
    if not os.path.exists(PATH_DATA):
        return None
    return pd.read_csv(PATH_DATA, sep=";", encoding="latin1")


@st.cache_data(show_spinner=False)
def charger_donnees_upload(fichier_upload) -> pd.DataFrame:
    """Charge le CSV uploadé via la sidebar."""
    return pd.read_csv(fichier_upload, sep=";", encoding="latin1")


def preparer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Recodage des variables catégorielles."""
    df = df.copy()
    df['milieu_f']      = df['milieu_residence'].map({1: "Urbain", 2: "Rural"})
    df['instruction_f'] = df['niveau_instruction'].map(
        {0: "Aucun", 1: "Primaire", 2: "Secondaire", 3: "Supérieur"})
    df['quintile_f']    = df['quintile_richesse'].map(
        {1: "Q1-Très pauvre", 2: "Q2-Pauvre", 3: "Q3-Moyen",
         4: "Q4-Riche",       5: "Q5-Très riche"})
    df['groupe_age_f']  = df['groupe_age'].map(
        {1: "15-19", 2: "20-24", 3: "25-29", 4: "30-34",
         5: "35-39", 6: "40-44", 7: "45-49"})
    df['radio_f']       = df['ecoute_radio'].map({0: "Non", 1: "Oui"})
    df['tv_f']          = df['vision_television'].map({0: "Non", 1: "Oui"})
    return df


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Navigation")

    onglet_choisi = st.radio(
        "Section",
        ["🏠 Accueil",
         "📈 Kaplan-Meier",
         "⚕️ Modèle de Cox",
         "🤖 Machine Learning",
         "🔮 Prédiction individuelle"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Données (optionnel)**")
    st.markdown(
        "<small>Si votre CSV est déjà dans le dossier de l'application, "
        "aucun upload n'est nécessaire.</small>",
        unsafe_allow_html=True
    )
    fichier = st.file_uploader(
        "Charger un autre CSV (séparateur ;)",
        type=["csv"],
        help="Laissez vide pour utiliser le fichier local donnees_contraception_survie1.csv"
    )

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================
if fichier is not None:
    # Priorité à l'upload utilisateur
    df_raw = charger_donnees_upload(fichier)
    source_donnees = f"CSV uploadé : **{fichier.name}**"
else:
    # Fallback : CSV local dans le dossier de l'app
    df_raw = charger_donnees_local()
    if df_raw is None:
        st.markdown(
            '<p class="main-title">Analyse de survie — Temps avant la première '
            'adoption de contraception moderne</p>'
            '<p class="subtitle">EDS Cameroun 2018 · M1 Statistique · '
            'Institut Saint Jean, Yaoundé</p>',
            unsafe_allow_html=True
        )
        st.error(
            "❌ Aucune donnée trouvée.\n\n"
            f"Placez **donnees_contraception_survie1.csv** dans le même dossier que `app.py` "
            f"(`{APP_DIR}`), ou uploadez votre fichier via la sidebar."
        )
        st.stop()
    source_donnees = f"Fichier local : **donnees_contraception_survie1.csv** ({APP_DIR})"

df      = preparer_donnees(df_raw)
n_total = len(df)
n_evt   = int(df['evenement'].sum())
n_cens  = n_total - n_evt

# ============================================================================
# ONGLET : ACCUEIL
# ============================================================================
if onglet_choisi == "🏠 Accueil":
    st.markdown(
        '<p class="main-title">Analyse de survie — Contraception moderne</p>'
        '<p class="subtitle">EDS Cameroun 2018</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="success-box">✅ Données chargées — {source_donnees}</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{n_total:,}</div>'
                    f'<div class="metric-lbl">Femmes incluses</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{n_evt:,}</div>'
                    f'<div class="metric-lbl">Événements (adoption moderne)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{n_cens:,}</div>'
                    f'<div class="metric-lbl">Censurées</div></div>', unsafe_allow_html=True)
    with c4:
        taux = round(n_evt / n_total * 100, 1)
        st.markdown(f'<div class="metric-card"><div class="metric-val">{taux}%</div>'
                    f'<div class="metric-lbl">Taux d\'adoption</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Aperçu des données</div>', unsafe_allow_html=True)
    col_affich = ['age_femme', 'age_premier_rapport', 'temps_survie', 'evenement',
                  'milieu_f', 'instruction_f', 'quintile_f', 'religion_label']
    st.dataframe(df[[c for c in col_affich if c in df.columns]].head(10), use_container_width=True)

    st.markdown('<div class="section-header">Distribution du temps de survie</div>', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df['temps_survie'], bins=40, edgecolor='white',
                color='#2A9D8F', alpha=0.85)
        ax.set_xlabel("Temps de survie (années)")
        ax.set_ylabel("Effectif")
        ax.set_title("Distribution du temps de survie")
        st.pyplot(fig)
        plt.close()
    with col_h2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ['#2A9D8F', '#E76F51']
        ax.bar(['Adoption\n(événement=1)', 'Censurée\n(événement=0)'],
               [n_evt, n_cens], color=colors, edgecolor='white')
        for i, v in enumerate([n_evt, n_cens]):
            ax.text(i, v + 30, f'{v:,}\n({v/n_total*100:.1f}%)',
                    ha='center', fontsize=9)
        ax.set_title("Répartition événement / censure")
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="info-box">📌 <strong>Variable cible :</strong> '
                '<code>temps_survie</code> = durée (en années) entre le premier rapport '
                'sexuel et l\'adoption d\'une contraception moderne. '
                'L\'événement est l\'adoption (<code>evenement=1</code>). '
                'Les femmes sans contraception moderne au moment de l\'enquête '
                'sont censurées à droite (<code>evenement=0</code>).</div>',
                unsafe_allow_html=True)

# ============================================================================
# ONGLET : KAPLAN-MEIER
# ============================================================================
elif onglet_choisi == "📈 Kaplan-Meier":
    st.markdown('<p class="main-title">Estimateur de Kaplan-Meier</p>', unsafe_allow_html=True)

    col_km, _ = st.columns([1, 2])
    with col_km:
        variable_km = st.selectbox(
            "Comparer par :",
            ["Global", "Milieu de résidence", "Niveau d'instruction",
             "Quintile de richesse", "Groupe d'âge",
             "Écoute radio", "Télévision"]
        )

    mapping_km = {
        "Global"               : (None, None),
        "Milieu de résidence"  : ("milieu_f", ["Urbain", "Rural"]),
        "Niveau d'instruction" : ("instruction_f", ["Aucun", "Primaire", "Secondaire", "Supérieur"]),
        "Quintile de richesse" : ("quintile_f", ["Q1-Très pauvre","Q2-Pauvre","Q3-Moyen","Q4-Riche","Q5-Très riche"]),
        "Groupe d'âge"         : ("groupe_age_f", ["15-19","20-24","25-29","30-34","35-39","40-44","45-49"]),
        "Écoute radio"         : ("radio_f", ["Non","Oui"]),
        "Télévision"           : ("tv_f", ["Non","Oui"]),
    }
    col_var, groupes = mapping_km[variable_km]

    palette = ['#2A9D8F','#E63946','#457B9D','#F4A261','#264653','#E9C46A','#A8DADC']

    fig, ax = plt.subplots(figsize=(9, 5))

    if col_var is None:
        kmf = KaplanMeierFitter()
        kmf.fit(df['temps_survie'], df['evenement'], label="Global")
        kmf.plot_survival_function(ax=ax, ci_show=True, color='#2A9D8F', linewidth=2)
        t50 = kmf.median_survival_time_
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.axvline(t50, color='gray', linestyle=':', linewidth=0.8)
        st.success(f"**Temps médian de survie :** {t50:.2f} années")
    else:
        groupes_presents = [g for g in groupes if g in df[col_var].dropna().unique()]
        for i, grp in enumerate(groupes_presents):
            mask = df[col_var] == grp
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, 'temps_survie'], df.loc[mask, 'evenement'], label=grp)
            kmf.plot_survival_function(ax=ax, ci_show=False,
                                       color=palette[i % len(palette)], linewidth=2)

        if len(groupes_presents) == 2:
            g0 = df[col_var] == groupes_presents[0]
            g1 = df[col_var] == groupes_presents[1]
            lr = logrank_test(
                df.loc[g0, 'temps_survie'], df.loc[g1, 'temps_survie'],
                df.loc[g0, 'evenement'],    df.loc[g1, 'evenement']
            )
            p = lr.p_value
        else:
            lr = multivariate_logrank_test(df['temps_survie'], df[col_var], df['evenement'])
            p = lr.p_value

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(0.98, 0.98, f"Log-rank p = {p:.4f} {sig}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Test log-rank p-value", f"{p:.4f}")
        col_r2.metric("Signification", sig)
        col_r3.metric("Groupes comparés", len(groupes_presents))

    ax.set_xlabel("Temps depuis premier rapport (années)", fontsize=11)
    ax.set_ylabel("Probabilité de non-adoption S(t)", fontsize=11)
    ax.set_title(f"Courbe de Kaplan-Meier — {variable_km}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
    st.download_button("📥 Télécharger la courbe",
                        data=fig_to_bytes(fig),
                        file_name=f"km_{variable_km.lower().replace(' ', '_')}.png",
                        mime="image/png")
    plt.close()

# ============================================================================
# ONGLET : MODÈLE DE COX
# ============================================================================
elif onglet_choisi == "⚕️ Modèle de Cox":
    st.markdown('<p class="main-title">Modèle de Cox (régression à risques proportionnels)</p>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">Ajustement du modèle</div>', unsafe_allow_html=True)

    df_cox = df[[
        'temps_survie', 'evenement', 'milieu_residence', 'niveau_instruction',
        'quintile_richesse', 'groupe_age', 'ecoute_radio', 'vision_television',
        'nombre_enfants_vivants', 'statut_matrimonial', 'religion_code'
    ]].dropna()

    with st.spinner("Ajustement du modèle de Cox..."):
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_cox, duration_col='temps_survie', event_col='evenement')

    st.success(f"✅ Modèle ajusté sur {len(df_cox):,} femmes | C-index : {cph.concordance_index_:.4f}")

    st.markdown('<div class="section-header">Résumé des coefficients</div>', unsafe_allow_html=True)
    summary = cph.summary.copy()
    summary.columns = [c.replace('coef lower 95%','IC 2.5%').replace('coef upper 95%','IC 97.5%')
                         .replace('exp(coef) lower 95%','HR IC 2.5%').replace('exp(coef) upper 95%','HR IC 97.5%')
                       for c in summary.columns]
    st.dataframe(summary[['coef', 'exp(coef)', 'HR IC 2.5%', 'HR IC 97.5%', 'p']].round(4),
                 use_container_width=True)

    st.markdown('<div class="section-header">Forest Plot — Hazard Ratios</div>', unsafe_allow_html=True)
    fig_fp, ax_fp = plt.subplots(figsize=(8, 6))
    cph.plot(ax=ax_fp)
    ax_fp.set_title("Forest Plot — Modèle de Cox multivarié", fontweight='bold')
    ax_fp.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    st.pyplot(fig_fp)
    st.download_button("📥 Forest Plot",
                        data=fig_to_bytes(fig_fp),
                        file_name="forest_plot_cox.png", mime="image/png")
    plt.close()

    st.markdown('<div class="section-header">Courbes de survie prédites (profils contrastés)</div>',
                unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Profil A — Urbain / Instruit / Riche**")
        milieu_A = st.selectbox("Milieu A", [1, 2], format_func=lambda x: "Urbain" if x == 1 else "Rural", key="mA")
        instr_A  = st.selectbox("Instruction A", [0,1,2,3], format_func=lambda x: ["Aucun","Primaire","Secondaire","Supérieur"][x], index=2, key="iA")
        quint_A  = st.selectbox("Quintile A", [1,2,3,4,5], index=4, key="qA")
    with col_p2:
        st.markdown("**Profil B — Rural / Non instruit / Pauvre**")
        milieu_B = st.selectbox("Milieu B", [1,2], format_func=lambda x: "Urbain" if x==1 else "Rural", index=1, key="mB")
        instr_B  = st.selectbox("Instruction B", [0,1,2,3], format_func=lambda x: ["Aucun","Primaire","Secondaire","Supérieur"][x], index=0, key="iB")
        quint_B  = st.selectbox("Quintile B", [1,2,3,4,5], index=0, key="qB")

    profils_df = pd.DataFrame({
        'milieu_residence'      : [milieu_A,   milieu_B],
        'niveau_instruction'    : [instr_A,    instr_B],
        'quintile_richesse'     : [quint_A,    quint_B],
        'groupe_age'            : [2,          5],
        'ecoute_radio'          : [1,          0],
        'vision_television'     : [1,          0],
        'nombre_enfants_vivants': [1,          4],
        'statut_matrimonial'    : [1,          1],
        'religion_code'         : [1,          3],
    })

    fig_pred, ax_pred = plt.subplots(figsize=(8, 4.5))
    cph.predict_survival_function(profils_df).plot(
        ax=ax_pred, color=['#2A9D8F','#E63946'], linewidth=2,
        label=['Profil A (Urbain/Instruit/Riche)', 'Profil B (Rural/Non instruit/Pauvre)']
    )
    ax_pred.set_xlabel("Temps (années)"); ax_pred.set_ylabel("S(t)")
    ax_pred.set_title("Courbes de survie prédites — Cox")
    ax_pred.legend(fontsize=9); ax_pred.grid(True, alpha=0.25)
    st.pyplot(fig_pred)
    st.download_button("📥 Courbes prédites",
                        data=fig_to_bytes(fig_pred),
                        file_name="cox_courbes_predites.png", mime="image/png")
    plt.close()

# ============================================================================
# ONGLET : MACHINE LEARNING
# ============================================================================
elif onglet_choisi == "🤖 Machine Learning":
    st.markdown('<p class="main-title">Machine Learning — Modèles de survie</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">ℹ️ Les modèles sont chargés depuis les fichiers <code>cox_model.pkl</code> '
        'et <code>rsf_model.pkl</code> générés par <code>ml_survie_contraception.py</code>. '
        f'Répertoire attendu : <code>{APP_DIR}</code></div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])

    with col_btn1:
        charger_cox = st.button("📂 Charger modèle Cox", use_container_width=True)
    with col_btn2:
        charger_rsf = st.button("📂 Charger modèle RSF", use_container_width=True)

    # ── Cox ──────────────────────────────────────────────────────────────────
    if charger_cox or st.session_state.get('cox_loaded'):
        cox_pkg = charger_modele_local(PATH_COX, "Cox PH")
        if cox_pkg:
            st.session_state['cox_loaded'] = True
            st.session_state['cox_pkg']    = cox_pkg
            cox_mod = cox_pkg['model']
            c_train = cox_pkg.get('c_index_train', 'N/A')
            c_test  = cox_pkg.get('c_index_test',  'N/A')

            st.markdown('<div class="section-header">Cox PH — Résultats</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("C-index (train)", f"{c_train:.4f}")
            c2.metric("C-index (test)",  f"{c_test:.4f}")

            coefs = pd.Series(cox_mod.coef_, index=FEATURES)
            hr_df = pd.DataFrame({
                'Variable'        : [LABELS[f] for f in FEATURES],
                'β (coefficient)' : coefs.values.round(4),
                'HR'              : np.exp(coefs.values).round(4),
                'Effet'           : ['↑ Risque' if hr > 1 else '↓ Risque'
                                     for hr in np.exp(coefs.values)]
            }).sort_values('HR', ascending=False)
            st.dataframe(hr_df, use_container_width=True)

            # Forest plot
            fig_cox, ax_cox = plt.subplots(figsize=(7, 5))
            hr_sorted = hr_df.sort_values('HR')
            colors = ['#E63946' if hr > 1 else '#2A9D8F' for hr in hr_sorted['HR']]
            ax_cox.barh(hr_sorted['Variable'], np.log(hr_sorted['HR']), color=colors, alpha=0.8)
            ax_cox.axvline(0, color='black', linewidth=1.2)
            ax_cox.set_xlabel("log(HR)")
            ax_cox.set_title("Hazard Ratios — Cox PH (ML)")
            st.pyplot(fig_cox)
            plt.close()

    # ── RSF ──────────────────────────────────────────────────────────────────
    if charger_rsf or st.session_state.get('rsf_loaded'):
        rsf_pkg = charger_modele_local(PATH_RSF, "RSF")
        if rsf_pkg:
            st.session_state['rsf_loaded'] = True
            st.session_state['rsf_pkg']    = rsf_pkg
            rsf_mod = rsf_pkg['model']
            c_train = rsf_pkg.get('c_index_train', 'N/A')
            c_test  = rsf_pkg.get('c_index_test',  'N/A')

            st.markdown('<div class="section-header">RSF — Résultats</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("C-index (train)", f"{c_train:.4f}")
            c2.metric("C-index (test)",  f"{c_test:.4f}")

            # Importance des variables (depuis permutation importance stockée dans le pkl)
            # Le pkl stocke "permutation_importance" si disponible, sinon feature_importances_
            if 'permutation_importance' in rsf_pkg:
                imp = pd.Series(rsf_pkg['permutation_importance'], index=FEATURES).sort_values(ascending=False)
                imp_label = "Permutation importance"
            else:
                # Fallback : feature_importances_ n'existe pas dans sksurv RSF —
                # on recalcule la permutation importance depuis X_shap si disponible
                imp = None

            if imp is not None:
                fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                imp_labeled = pd.Series(imp.values, index=[LABELS[f] for f in imp.index])
                colors = ['#E63946' if i == 0 else '#457B9D' for i in range(len(imp))]
                imp_labeled.sort_values().plot(kind='barh', ax=ax_imp, color=colors[::-1])
                ax_imp.axvline(imp.mean(), color='orange', linestyle='--',
                               linewidth=1.2, label=f'Moyenne ({imp.mean():.4f})')
                ax_imp.set_xlabel("Importance (baisse du C-index)")
                ax_imp.set_title(f"Importance des variables — Random Survival Forest ({imp_label})")
                ax_imp.legend()
                st.pyplot(fig_imp)
                st.download_button("📥 Graphique importance RSF",
                                    data=fig_to_bytes(fig_imp),
                                    file_name="rsf_importance.png", mime="image/png")
                plt.close()

            # SHAP (si présents dans le pkl)
            if 'shap_values' in rsf_pkg and 'X_shap' in rsf_pkg:
                st.markdown('<div class="section-header">Interprétabilité SHAP</div>', unsafe_allow_html=True)
                import shap
                shap_values = rsf_pkg['shap_values']
                X_shap      = rsf_pkg['X_shap']
                X_shap_lbl  = X_shap.copy()
                X_shap_lbl.columns = [LABELS[f] for f in FEATURES]
                fig_shap, _ = plt.subplots(figsize=(9, 5))
                shap.summary_plot(shap_values, X_shap_lbl, show=False, max_display=14)
                plt.title("SHAP — Impact des variables (RSF)")
                st.pyplot(fig_shap)
                st.download_button("📥 Graphique SHAP",
                                    data=fig_to_bytes(fig_shap),
                                    file_name="shap_summary.png", mime="image/png")
                plt.close()

    if not st.session_state.get('cox_loaded') and not st.session_state.get('rsf_loaded'):
        st.info("Cliquez sur un bouton ci-dessus pour charger et afficher les résultats d'un modèle.")

# ============================================================================
# ONGLET : PRÉDICTION INDIVIDUELLE
# ============================================================================
elif onglet_choisi == "🔮 Prédiction individuelle":
    st.markdown('<p class="main-title">Prédiction individuelle</p>', unsafe_allow_html=True)
    st.markdown("Renseignez le profil d'une femme pour estimer sa probabilité de ne pas encore "
                "avoir adopté une contraception moderne à différents horizons.")

    # Chargement automatique du RSF depuis le fichier local
    rsf_pkg = st.session_state.get('rsf_pkg', None)
    if rsf_pkg is None:
        with st.spinner("Chargement du modèle RSF (fichier local)..."):
            rsf_pkg = charger_modele_local(PATH_RSF, "RSF")
        if rsf_pkg:
            st.session_state['rsf_pkg'] = rsf_pkg

    if rsf_pkg is None:
        st.error(
            f"❌ Modèle RSF introuvable.\n\n"
            f"Exécutez d'abord `ml_survie_contraception.py` pour générer "
            f"`rsf_model.pkl` dans `{APP_DIR}`."
        )
        st.stop()

    rsf_mod = rsf_pkg['model']

    st.markdown('<div class="section-header">Profil individuel</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        ga    = st.selectbox("Groupe d'âge", [1,2,3,4,5,6,7],
                              format_func=lambda x: {1:"15-19",2:"20-24",3:"25-29",4:"30-34",
                                                      5:"35-39",6:"40-44",7:"45-49"}[x])
        instr = st.selectbox("Niveau d'instruction", [0,1,2,3],
                              format_func=lambda x: ["Aucun","Primaire","Secondaire","Supérieur"][x])
        quint = st.selectbox("Quintile de richesse", [1,2,3,4,5],
                              format_func=lambda x: f"Q{x}")
        milieu = st.selectbox("Milieu", [1,2], format_func=lambda x: "Urbain" if x==1 else "Rural")
    with col2:
        sm    = st.selectbox("Statut matrimonial", [0,1,2,3,4,5],
                              format_func=lambda x: {0:"Jamais marié",1:"Marié(e)",
                                                      2:"En union",3:"Veuf/ve",
                                                      4:"Divorcé(e)",5:"Non en union"}[x])
        relig = st.selectbox("Religion (code)", [1,2,3,4,5,6],
                              format_func=lambda x: {1:"Catholique",2:"Protestante",
                                                      3:"Musulmane",4:"Autre chrét.",
                                                      5:"Animiste",6:"Autre/Sans"}[x])
        region = st.selectbox("Région (code)", list(range(1,13)))
        typ_union = st.selectbox("Type d'union", [0,1,2,3])
    with col3:
        radio = st.selectbox("Écoute radio", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        tv    = st.selectbox("Télévision", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        jrnx  = st.selectbox("Journaux", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        nb_enf = st.slider("Nb enfants vivants", 0, 10, 2)
        nb_naiss = st.slider("Nb naissances", 0, 12, 2)
        occ   = st.selectbox("Occupation", list(range(0, 12)))

    profil_saisie = pd.DataFrame([[
        ga, sm, instr, quint, milieu, region, relig,
        radio, tv, jrnx, nb_enf, nb_naiss, occ, typ_union
    ]], columns=FEATURES)

    if st.button("🔮 Calculer la prédiction", type="primary", use_container_width=True):
        surv_fn = rsf_mod.predict_survival_function(profil_saisie)[0]

        horizons = [5, 10, 15, 20, 25]
        probs = {}
        for h in horizons:
            idx = np.searchsorted(surv_fn.x, h)
            idx = min(idx, len(surv_fn.y) - 1)
            probs[h] = surv_fn.y[idx]

        st.markdown('<div class="section-header">Résultats</div>', unsafe_allow_html=True)

        cols = st.columns(len(horizons))
        for i, h in enumerate(horizons):
            p_adoption = 1 - probs[h]
            cols[i].metric(f"À {h} ans", f"{p_adoption*100:.1f}%", "prob. adoption")

        fig_ind, ax_ind = plt.subplots(figsize=(8, 4))
        ax_ind.step(surv_fn.x, surv_fn.y, where='post', color='#2A9D8F', linewidth=2.5)
        ax_ind.fill_between(surv_fn.x, surv_fn.y, step='post', alpha=0.15, color='#2A9D8F')
        for h in horizons:
            idx = min(np.searchsorted(surv_fn.x, h), len(surv_fn.y)-1)
            ax_ind.plot(h, surv_fn.y[idx], 'o', color='#E63946', markersize=7, zorder=5)
            ax_ind.text(h, surv_fn.y[idx]+0.02, f'{surv_fn.y[idx]*100:.0f}%',
                        ha='center', fontsize=8, color='#E63946')
        ax_ind.set_xlabel("Temps depuis premier rapport (années)", fontsize=11)
        ax_ind.set_ylabel("Probabilité de non-adoption S(t)", fontsize=11)
        ax_ind.set_title("Courbe de survie prédite — Profil individuel", fontsize=12, fontweight='bold')
        ax_ind.grid(True, alpha=0.25); ax_ind.set_ylim(0, 1.05)
        st.pyplot(fig_ind)
        st.download_button("📥 Télécharger la courbe",
                            data=fig_to_bytes(fig_ind),
                            file_name="prediction_individuelle.png", mime="image/png")
        plt.close()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<center><small>EDS Cameroun 2018 · M1 Statistique & Analyse des données · "
    "Institut Saint Jean, Yaoundé · Superviseur : Pr. Nguefack Tsague Georges</small></center>",
    unsafe_allow_html=True
)
