"""
Script à exécuter UNE FOIS sur ta machine pour régénérer les modèles pkl.
Place ce fichier dans le même dossier que donnees_contraception_survie.csv
puis exécute : python generer_modeles.py
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

# ── Chemin du CSV ─────────────────────────────────────────────────────────────
DOSSIER = os.path.dirname(os.path.abspath(__file__))
CSV     = os.path.join(DOSSIER, "donnees_contraception_survie1.csv")

print("=" * 60)
print("GÉNÉRATION DES MODÈLES ML — SURVIE CONTRACEPTION")
print("=" * 60)

# ── Chargement ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV, sep=";", encoding="latin1")
print(f"Données : {df.shape[0]} femmes × {df.shape[1]} variables")

# ── Variables ─────────────────────────────────────────────────────────────────
FEATURES = [
    'age_premier_rapport',
    'groupe_age',
    'statut_matrimonial',
    'niveau_instruction',
    'quintile_richesse',
    'milieu_residence',
    'region',
    'religion_code',
    'ecoute_radio',
    'vision_television',
    'lecture_journaux',
    'nombre_enfants_vivants',
    'nombre_naissances',
    'occupation',
    'type_union'
]
LABELS = {
    'age_premier_rapport'    : "Âge au 1er rapport (ans)",
    'groupe_age'             : "Groupe d'âge",
    'statut_matrimonial'     : "Statut matrimonial",
    'niveau_instruction'     : "Niveau instruction",
    'quintile_richesse'      : "Quintile richesse",
    'milieu_residence'       : "Milieu résidence",
    'region'                 : "Région",
    'religion_code'          : "Religion",
    'ecoute_radio'           : "Écoute radio",
    'vision_television'      : "Télévision",
    'lecture_journaux'       : "Journaux",
    'nombre_enfants_vivants' : "Nb enfants vivants",
    'nombre_naissances'      : "Nb naissances",
    'occupation'             : "Occupation",
    'type_union'             : "Type d'union",
}

df_ml = df[FEATURES + ['evenement', 'temps_survie']].dropna()
print(f"Après suppression NA : {df_ml.shape[0]} femmes | "
      f"{int(df_ml['evenement'].sum())} événements")

X = df_ml[FEATURES].astype(float)
y = Surv.from_dataframe('evenement', 'temps_survie', df_ml)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train : {len(X_train)} | Test : {len(X_test)}")

# ── Cox PH ────────────────────────────────────────────────────────────────────
print("\n[1/3] Entraînement Cox PH...")
cox = CoxPHSurvivalAnalysis(alpha=0.1, ties='efron')
cox.fit(X_train, y_train)
c_train_cox = cox.score(X_train, y_train)
c_test_cox  = cox.score(X_test,  y_test)
print(f"      C-index train : {c_train_cox:.4f}")
print(f"      C-index test  : {c_test_cox:.4f}")

cox_path = os.path.join(DOSSIER, "cox_model.pkl")
with open(cox_path, 'wb') as f:
    pickle.dump({
        'model'          : cox,
        'features'       : FEATURES,
        'labels'         : LABELS,
        'c_index_train'  : c_train_cox,
        'c_index_test'   : c_test_cox,
    }, f)
print(f"      ✓ Sauvegardé : {cox_path}")

# ── RSF ───────────────────────────────────────────────────────────────────────
print("\n[2/3] Entraînement RSF (quelques minutes)...")
rsf = RandomSurvivalForest(
    n_estimators    = 30,
    min_samples_split = 15,
    min_samples_leaf  = 8,
    max_depth       = 6,
    n_jobs          = 1,      # 1 pour éviter les erreurs Windows
    random_state    = 42
)
rsf.fit(X_train, y_train)
c_train_rsf = rsf.score(X_train, y_train)
c_test_rsf  = rsf.score(X_test,  y_test)
print(f"      C-index train : {c_train_rsf:.4f}")
print(f"      C-index test  : {c_test_rsf:.4f}")

# ── Permutation importance ────────────────────────────────────────────────────
print("\n[3/3] Calcul de la permutation importance (patience)...")
perm = permutation_importance(
    rsf, X_test, y_test,
    n_repeats  = 5,
    random_state = 42,
    n_jobs     = 1
)
importance = pd.Series(perm.importances_mean, index=FEATURES) \
               .sort_values(ascending=False)
print("      Variables les plus importantes :")
for feat, val in importance.head(5).items():
    print(f"        {LABELS[feat]:<35s} : {val:.4f}")

rsf_path = os.path.join(DOSSIER, "rsf_model.pkl")
with open(rsf_path, 'wb') as f:
    pickle.dump({
        'model'         : rsf,
        'features'      : FEATURES,
        'labels'        : LABELS,
        'c_index_train' : c_train_rsf,
        'c_index_test'  : c_test_rsf,
        'importance'    : importance,
        'X_train'       : X_train,
        'X_test'        : X_test,
        'y_train'       : y_train,
        'y_test'        : y_test,
    }, f)
print(f"      ✓ Sauvegardé : {rsf_path}")

print("\n" + "=" * 60)
print("TERMINÉ — deux fichiers créés dans le même dossier :")
print(f"  • cox_model.pkl")
print(f"  • rsf_model.pkl")
print("Lancez maintenant : streamlit run app.py")
print("=" * 60)
