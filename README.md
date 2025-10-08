# Resume Bias Analysis in Data Science

## 🌟 Projektübersicht

Dieses Projekt untersucht mögliche **Biases** in Lebensläufen (Resumes), insbesondere solche, die auf Geschlecht, Ausdrucksweise oder Stil basieren könnten — mit Fokus auf Rollen im Bereich Data Science. Ziel ist es, systematische Muster zu erkennen, die in automatisierten Auswahlverfahren zu Verzerrungen führen könnten, und Handlungsempfehlungen ableiten zu können.

---

## 📁 Verzeichnisstruktur

```

├── data
│   ├── processed
│   │   ├── exports
│   │   ├── images
│   │   └── applicants_with_demographics.csv
│   └── raw
│       └── AI_Resume_Screening.csv
├── notebooks
│   ├── bias_analysis.ipynb
│   ├── datascience.ipynb
│   ├── fairness_metrics.py
│   ├── fairness_quickstart.py
│   └── openai_test.ipynb
├── .gitignore
├── README.md
└── structure.py
```

---

## ✅ Voraussetzungen & Installation

Folge diesen Schritten, um das Projekt lokal lauffähig zu machen:

```bash
# 1. Repository klonen
git clone https://github.com/spasegirl/resume_bias_analysis_datascience.git
cd resume_bias_analysis_datascience

# 2. Virtuelle Umgebung einrichten (optional, aber empfohlen)
python3 -m venv venv
source venv/bin/activate      # auf macOS / Linux
# venv\Scripts\activate       # auf Windows

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Jupyter Notebook starten
jupyter notebook
```

Stelle sicher, dass alle Pakete wie `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, ggf. `fairlearn` etc. installiert sind.

---

## Ablauf der Analyse

1. **Datenexploration & -bereinigung**  
   Untersuchung der Verteilungen, fehlende Werte und Inkonsistenzen in den Daten.

2. **Feature Engineering**  
   Extrahieren der Textmerkmale (z. B. Wortanzahl, bestimmte Keywords, Stilmetriken).

3. **Modelltraining & Bias-Messung**  
   Trainieren der Modelle zur Vorhersage (z. B. ob ein Lebenslauf „attractiv“ ist). Es werden anschließend Fairness-Metriken wie Demographic Parity, Equalized Odds angewandt.

4. **Vergleich & Interpretation**  
   Analyse der Unterschiede zwischen Gruppen (z. B. Geschlecht) und Interpretation, welche Merkmale am stärksten mit Abweichungen korrelieren.

5. **Visualisierung & Berichterstattung**  
   Erstellen von aussagekräftigen Grafiken und Summary der Ergebnisse in einem narrativen Bericht.

---

## Erkenntnisse

- Hohe Korrelation zwischen AI-Score und Hire-Entscheidung:
    Der AI-Score (0–100) ist fast direkt mit der Einstellung (Hire) verknüpft. Modelle, die diesen Score verwenden, erreichen deshalb extrem hohe Accuracy- und AUC-Werte, was auf eine mögliche Label-Leakage oder Score-Kopplung hindeutet.

- Erfahrung wirkt sich stark positiv aus:
    Bewerber:innen mit mehr Berufsjahren werden deutlich häufiger als „Hire“ klassifiziert. Zwischen 0 und 8 Jahren steigt die Einstellungswahrscheinlichkeit fast monoton.

- Geschlechtsverteilung ausgewogen, aber geringe Stichprobe „unknown“:
    Etwa gleich viele männliche und weibliche Profile, jedoch wenige „unknown“. Diese kleine Gruppe zeigt stärkere Streuung in den Scores (höhere Varianz).

- Numerische Merkmale korrelieren unterschiedlich:
Erfahrung (Years) und AI-Score zeigen starke Korrelation (~0.8), während Projektezahl und Gehalt nur schwach beitragen.

- Kaum fehlende Werte oder Datenprobleme:
Nullraten liegen bei 0 % für alle zentralen Merkmale – das Dataset ist vollständig.

- Fairness-Metriken deuten geringe, aber messbare Unterschiede:
Die Analyse von Equalized Odds und Demographic Parity zeigt kleine Gaps zwischen Gruppen, besonders bei höheren Thresholds (t > 0.5).

- Kalibrierung verbessert Fairness leicht:
Durch Isotonic bzw. Platt-Scaling wurden Score-Verteilungen über Gruppen angeglichen, was TPR/FPR-Differenzen reduzierte.

- Modelle ohne AI-Score verlieren stark an Performance:
Entfernt man den AI-Score als Feature, sinkt Accuracy deutlich – Hinweis, dass der Score das dominante Merkmal ist.

- Tree-basierte Modelle (z. B. Gradient Boosting) liefern stabilere, aber weniger interpretierbare Ergebnisse als logistische Regression.

- Bootstrap- und Threshold-Analysen bestätigen Robustheit:
Gruppenmetriken bleiben bei resampling stabil, allerdings nehmen Unterschiede bei extremen Schwellenwerten zu

---
