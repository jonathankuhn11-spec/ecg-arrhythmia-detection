# Ergebnisbericht

Automatisch erzeugt von `run_pipeline.py`. Regelwerk und Modell wurden ausschließlich auf DS1
eingestellt. Alle Kennzahlen stammen von der Testmenge DS2, sofern nicht anders angegeben.

## Daten

| Menge | Records | Schläge | davon VEB | Dauer in h |
| --- | --- | --- | --- | --- |
| DS1 (Training) | 22 | 50995 | 3788 | 11,03 |
| DS2 (Test) | 22 | 49685 | 3220 | 11,03 |

## VEB-Erkennung auf DS2

| Detektor | Sensitivität | pos. Prädiktivität | Fehlalarme | Fehlalarme pro h | verpasste VEB |
| --- | --- | --- | --- | --- | --- |
| Regelwerk (relativ) | 85,3 % | 91,7 % | 250 | 22,7 | 474 |
| Modell, Schwelle 0,5 | 90,9 % | 97,4 % | 79 | 7,2 | 294 |

Das Modell meldet 68,4 % weniger Fehlalarme als das Regelwerk
und erkennt zugleich mehr VEB (90,9 % gegenüber 85,3 %).

## Regelvarianten und Vergleich bei gleicher Sensitivität

Beide Varianten haben dieselbe Struktur: (vorzeitig UND verbreitert) ODER stark verbreitert.
Sie messen die QRS-Breite relativ zu den letzten 30 Schlägen oder absolut in Millisekunden.
Die Schwellen stammen aus einer Gittersuche auf DS1 (Zielgröße F1). Als Hauptvariante gilt `relativ`,
weil sie auf DS1 den höheren F1-Wert erreicht.

| Variante | F1 auf DS1 | DS1: Se / +P | DS2: Se / +P | Fehlalarme DS2 | Modell bei gleicher Se: Fehlalarme | Differenz |
| --- | --- | --- | --- | --- | --- | --- |
| relativ | 0,817 | 74,2 % / 90,7 % | 85,3 % / 91,7 % | 250 | 31 (Schwelle 0,619) | −87,6 % |
| absolut | 0,786 | 72,9 % / 85,2 % | 79,9 % / 98,8 % | 30 | 8 (Schwelle 0,697) | −73,3 % |

Die Modellschwelle für den Vergleich bei gleicher Sensitivität wurde auf DS2 abgelesen.
Das ist eine Beschreibung der Trennschärfe, kein vorab festgelegter Betriebspunkt.

Kalibrierte Schwellen:

- `relativ`: vorzeitig bei `rr_ratio_pre < 0,92`, verbreitert bei `qrs_width_rel > 1,15`, stark verbreitert bei `qrs_width_rel > 1,80`
- `absolut`: vorzeitig bei `rr_ratio_pre < 0,88`, verbreitert bei `qrs_width_ms > 40 ms`, stark verbreitert bei `qrs_width_ms > 80 ms`

## Fehlalarme nach wahrer Klasse (Regelwerk `relativ` und Modell)

| Detektor | N (normal) | S (supraventrikulär) | F (Fusion) |
| --- | --- | --- | --- |
| Regelwerk | 131 | 58 | 61 |
| Modell | 28 | 28 | 23 |

## Fehler je Record

| Record | VEB | Fehlalarme Regelwerk | Fehlalarme Modell | verpasst Regelwerk | verpasst Modell |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | 1 | 0 | 0 | 0 |
| 103 | 0 | 1 | 0 | 0 | 0 |
| 105 | 41 | 27 | 18 | 30 | 28 |
| 111 | 1 | 24 | 2 | 0 | 0 |
| 113 | 0 | 0 | 0 | 0 | 0 |
| 117 | 0 | 0 | 0 | 0 | 0 |
| 121 | 1 | 1 | 1 | 1 | 0 |
| 123 | 3 | 0 | 0 | 0 | 0 |
| 200 | 826 | 2 | 0 | 159 | 45 |
| 202 | 19 | 11 | 4 | 13 | 1 |
| 210 | 195 | 13 | 2 | 56 | 35 |
| 212 | 0 | 0 | 0 | 0 | 0 |
| 213 | 220 | 60 | 21 | 7 | 30 |
| 214 | 256 | 5 | 1 | 132 | 67 |
| 219 | 64 | 13 | 0 | 6 | 8 |
| 221 | 396 | 1 | 0 | 0 | 1 |
| 222 | 0 | 82 | 17 | 0 | 0 |
| 228 | 362 | 5 | 0 | 0 | 0 |
| 231 | 2 | 0 | 0 | 0 | 1 |
| 232 | 0 | 2 | 9 | 0 | 0 |
| 233 | 830 | 2 | 4 | 70 | 78 |
| 234 | 3 | 0 | 0 | 0 | 0 |

## Wichtigste Merkmale des Modells

| Merkmal | Bedeutung | Anteil |
| --- | --- | --- |
| `rr_ratio_pre` | Vorzeitigkeit: RR-Intervall vor dem Schlag relativ zum lokalen Rhythmus | 14,9 % |
| `rr_ratio_post` | Pause danach: RR-Intervall nach dem Schlag relativ zum Intervall davor | 13,0 % |
| `qrs_width_rel` | QRS-Breite relativ zu den letzten 30 Schlägen | 11,9 % |
| `qrs_width_ms` | QRS-Breite absolut (Halbwertsbreite in ms) | 9,1 % |
| `rr_pre` | RR-Intervall vor dem Schlag in s | 8,0 % |
| `m07` | Schlagform: Mittelwert eines 25-ms-Abschnitts | 5,2 % |
| `m10` | Schlagform: Mittelwert eines 25-ms-Abschnitts | 4,9 % |
| `m11` | Schlagform: Mittelwert eines 25-ms-Abschnitts | 3,5 % |

Anteil: mittlere Verringerung der Unreinheit im Random Forest (Summe über alle Merkmale 100 %).

## R-Zacken-Detektion (NeuroKit2)

Über alle 44 Records: Sensitivität 98,8 %, positive Prädiktivität 98,1 % (99507 Treffer, 1896 Fehldetektionen, 1226 verpasste Schläge, Toleranz 150 ms).

Die fünf schwächsten Records:

| Record | Sensitivität | pos. Prädiktivität | Fehldetektionen | verpasst |
| --- | --- | --- | --- | --- |
| 113 | 99,9 % | 63,3 % | 1039 | 1 |
| 207 | 72,0 % | 96,6 % | 47 | 520 |
| 108 | 88,7 % | 89,6 % | 181 | 200 |
| 231 | 100,0 % | 79,3 % | 410 | 0 |
| 203 | 91,9 % | 98,9 % | 30 | 241 |
