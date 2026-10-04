"""Zentrale Konstanten der Pipeline. Alle Stellschrauben stehen hier an einer Stelle."""

# --- Daten -------------------------------------------------------------------
FS = 360          # Abtastrate der MIT-BIH Arrhythmia Database in Hz
LEAD = "MLII"     # modifizierte Ableitung II, in allen verwendeten Records vorhanden

# Inter-Patienten-Aufteilung nach de Chazal et al. (2004): Training (DS1) und Test (DS2)
# enthalten unterschiedliche Patienten. Die vier Records mit Schrittmacherschlägen
# (102, 104, 107, 217) bleiben nach AAMI-Empfehlung außen vor.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# Zuordnung der MIT-BIH-Schlagsymbole zu den AAMI-Klassen (ANSI/AAMI EC57)
AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",   # normal, Schenkelblock, Ersatzschläge
    "A": "S", "a": "S", "J": "S", "S": "S",             # supraventrikuläre Extrasystolen
    "V": "V", "E": "V",                                 # ventrikuläre Extrasystolen (VEB)
    "F": "F",                                           # Fusionsschlag
    "/": "Q", "f": "Q", "Q": "Q",                       # Schrittmacher, nicht klassifizierbar
}

# --- Vorverarbeitung und Merkmale --------------------------------------------
BANDPASS_HZ = (0.5, 40.0)   # entfernt Grundlinienschwankung und hochfrequentes Rauschen
FILTER_ORDER = 3
WIN_PRE_S = 0.25            # Schlagfenster: 250 ms vor ...
WIN_POST_S = 0.40           # ... bis 400 ms nach der R-Zacke
BIN_S = 0.025               # Morphologie: Mittelwert je 25-ms-Abschnitt
WIDTH_HALF_S = 0.15         # Suchfenster für die QRS-Breite: ±150 ms
RR_LOCAL_BEATS = 10         # lokaler Rhythmus: Median der letzten 10 RR-Intervalle
REF_BEATS = 30              # Referenz für Breite und Amplitude: Median der letzten 30 Schläge

# --- Auswertung --------------------------------------------------------------
QRS_TOL_S = 0.15            # Toleranz beim Abgleich detektierter R-Zacken (AAMI: 150 ms)
SEED = 0                    # fester Zufallsstartwert für reproduzierbare Ergebnisse
