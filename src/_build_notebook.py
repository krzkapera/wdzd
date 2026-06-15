"""Buduje notebook src/lol.ipynb: analiza nietypowych lotow (OpenSky).

Skrypt pomocniczy/reprodukowalny -- generuje wszystkie komorki markdown i kod.
Uruchom: .venv/bin/python src/_build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- TYTUL / WSTEP
md(r"""
# Wykrywanie nietypowych lotow na podstawie trajektorii (OpenSky)

**Cel projektu.** Zidentyfikowac *nietypowe loty* na podstawie parametrow trajektorii:
wysokosci, predkosci, zmian kursu i dlugosci trasy. Kazdy lot opisujemy **wektorem cech**,
a nastepnie szukamy lotow, ktore odstaja od reszty ruchu lotniczego.

**Dane.** Probka [OpenSky Network](https://opensky-network.org/) -- *state vectors* z jednej
godziny ruchu lotniczego (`2022-06-27`, godz. `23:00` UTC). Jeden rekord = stan jednego
samolotu w jednej sekundzie (pozycja, predkosc, kurs, wysokosc...).

**Plan analizy:**
1. Wczytanie i czyszczenie danych.
2. Przeglad geograficzny ruchu (mapa interaktywna).
3. Inzynieria cech -- zamiana sekwencji pomiarow w **wektor cech jednego lotu**.
4. Eksploracja cech (rozklady, korelacje).
5. **Redukcja wymiaru** (PCA, UMAP, PaCMAP, TriMAP, t-SNE) -- wizualizacja lotow w 2D.
6. **Wykrywanie anomalii** (Isolation Forest, LOF, One-Class SVM) + konsensus.
7. **Interpretacja anomalii** -- jak konkretnie odstaja (wartosci vs reszta, trajektorie).
8. **Klastrowanie / typy lotow** (element rozszerzony).
9. Wnioski i kierunki rozwoju.

> Wszystkie wykresy sa **interaktywne** (Plotly) -- najedz kursorem na punkt, aby zobaczyc
> identyfikator i parametry konkretnego lotu; mozna przyblizac, filtrowac legende i zapisywac.
""")

# ---------------------------------------------------------------- IMPORTY / SETUP
md(r"""
## 0. Konfiguracja srodowiska

Importujemy biblioteki i ustawiamy wspolny styl wykresow Plotly. Definiujemy tez slownik
opisow cech (uzywany w podpowiedziach hover) oraz katalog, do ktorego pod koniec wyeksportujemy
interaktywne wykresy na potrzeby prezentacji HTML.
""")

code(r"""
import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
RANDOM_STATE = 42
DATA_PATH = "../data/states_2022-06-27-23.json"

# katalog na wyeksportowane figury (dla prezentacji reveal.js)
EXPORT_DIR = os.path.join("..", "presentation", "figures")
os.makedirs(EXPORT_DIR, exist_ok=True)


def save_fig(fig, name):
    # Zapisuje interaktywna figure jako samodzielny plik HTML (osadzany w prezentacji)
    path = os.path.join(EXPORT_DIR, name + ".html")
    fig.write_html(path, include_plotlyjs="cdn", full_html=True,
                   config={"displaylogo": False, "responsive": True})
    return path


# opisy i ladne nazwy cech lotu (uzywane w hover oraz tabelach)
FEATURE_INFO = {
    "alt_mean": "Srednia wysokosc [m]",
    "alt_max": "Maksymalna wysokosc [m]",
    "alt_min": "Minimalna wysokosc [m]",
    "alt_std": "Odchylenie wysokosci [m]",
    "alt_range": "Zakres wysokosci (max-min) [m]",
    "vel_mean": "Srednia predkosc [m/s]",
    "vel_max": "Maks. predkosc [m/s]",
    "vel_std": "Zmiennosc predkosci [m/s]",
    "vert_mean": "Srednia predkosc pionowa [m/s]",
    "vert_max": "Maks. wznoszenie [m/s]",
    "vert_min": "Maks. opadanie [m/s]",
    "vert_std": "Zmiennosc pred. pionowej [m/s]",
    "heading_mean": "Sredni kurs [stopnie]",
    "heading_std": "Zmiennosc kursu [stopnie]",
    "lat_range": "Zasieg szer. geogr. [stopnie]",
    "lon_range": "Zasieg dlug. geogr. [stopnie]",
    "measurements_count": "Liczba pomiarow",
    "time_span": "Czas obserwacji [s]",
}
print("Plotly gotowe. Figury beda zapisywane do:", os.path.abspath(EXPORT_DIR))
""")

# ---------------------------------------------------------------- WCZYTANIE
md(r"""
## 1. Wczytanie danych

Wczytujemy godzinna probke *state vectors*. Kazdy wiersz to stan jednego samolotu
(`icao24` = unikalny identyfikator transpondera) w danej sekundzie. Najwazniejsze kolumny:

| kolumna | znaczenie |
|---|---|
| `time` | znacznik czasu (UNIX) |
| `icao24` | 24-bitowy identyfikator transpondera samolotu |
| `callsign` | znak wywolawczy lotu (linia + numer) |
| `lat`, `lon` | szerokosc / dlugosc geograficzna |
| `velocity` | predkosc wzgledem ziemi [m/s] |
| `heading` | kurs (kat od polnocy) [stopnie] |
| `vertrate` | predkosc pionowa [m/s] (+ wznoszenie, - opadanie) |
| `baroaltitude`, `geoaltitude` | wysokosc barometryczna / GPS [m] |
| `onground` | czy samolot jest na ziemi |
| `squawk` | kod transpondera (np. 7500/7600/7700 = sytuacje awaryjne) |

> **Uwaga:** plik (`~800 MB`) jest celowo w `.gitignore`. Jesli go brakuje, zobacz `data/README.txt`
> oraz `README.md` -- pobierany jest z `opensky-network.org/datasets/states/`.
""")

code(r"""
df = pd.read_json(DATA_PATH)
print(f"Wczytano {len(df):,} rekordow (state vectors) x {df.shape[1]} kolumn")
df.head()
""")

# ---------------------------------------------------------------- CZYSZCZENIE
md(r"""
## 2. Jakosc i czyszczenie danych

Czesc rekordow ma braki (np. samolot bez nadajnika GPS nie poda `geoaltitude`, a `squawk`
bywa nieprzypisany). Sprawdzmy skale brakow, a nastepnie usunmy niekompletne rekordy, bo do
budowy cech trajektorii potrzebujemy kompletu pomiarow (pozycja, predkosc, wysokosc).
""")

code(r"""
missing = pd.DataFrame({
    "braki": df.isna().sum(),
    "% brakow": (df.isna().mean() * 100).round(2),
}).sort_values("% brakow", ascending=False)
rows_with_na = df.isna().any(axis=1).sum()
print(f"Rekordy z jakimkolwiek brakiem: {rows_with_na:,} "
      f"({rows_with_na / len(df) * 100:.1f}% wszystkich)")
missing
""")

code(r"""
df = df.dropna().reset_index(drop=True)
# callsign bywa dopelniony spacjami ("DAL595  ") -- czyscimy, by spojnie sklejac flight_id
df["callsign"] = df["callsign"].str.strip()
df["flight_id"] = df["icao24"] + "/" + df["callsign"]
print(f"Po usunieciu brakow: {len(df):,} rekordow, "
      f"{df['flight_id'].nunique():,} unikalnych lotow (icao24/callsign)")
df.head()
""")

# ---------------------------------------------------------------- MAPA
md(r"""
## 3. Przeglad geograficzny ruchu lotniczego

Zanim policzymy cechy, zobaczmy *jak wyglada godzina ruchu lotniczego*. Ponizsza mapa
pokazuje probke pozycji samolotow, pokolorowanych wedlug wysokosci. Wyraznie widac korytarze
nad Ameryka Polnocna i Europa oraz nisko lecace samoloty (kolor ciemny) w rejonach lotnisk.

*Interakcja:* najedz na punkt (znak wywolawczy, wysokosc, predkosc), przyblizaj, obracaj glob.
""")

code(r"""
sample_map = df.sample(min(20000, len(df)), random_state=RANDOM_STATE)
fig_map = px.scatter_geo(
    sample_map, lat="lat", lon="lon",
    color="geoaltitude", color_continuous_scale="Turbo",
    hover_name="flight_id",
    hover_data={"callsign": True, "velocity": ":.0f", "geoaltitude": ":.0f",
                "lat": False, "lon": False},
    projection="natural earth",
    title="Ruch lotniczy 2022-06-27 23:00 UTC (probka 20 tys. pozycji, kolor = wysokosc)",
)
fig_map.update_traces(marker=dict(size=3, opacity=0.6))
fig_map.update_layout(height=550, coloraxis_colorbar_title="Wys. [m]")
save_fig(fig_map, "01_mapa_ruchu")
fig_map.show()
""")

# ---------------------------------------------------------------- CECHY
md(r"""
## 4. Inzynieria cech -- lot jako wektor

Surowe dane to *sekwencje* pomiarow. Aby porownywac loty, agregujemy wszystkie pomiary jednego
lotu (`icao24` + `callsign`) w **jeden wektor cech**. Grupujemy cechy w bloki tematyczne:

- **Wysokosc** (`alt_*`): srednia, min, max, odchylenie, zakres -- profil pionowy lotu.
- **Predkosc** (`vel_*`): srednia, max, zmiennosc -- czy lot byl rejsowy, czy zmienny.
- **Pionowa** (`vert_*`): srednia/min/max/odchylenie pred. pionowej -- ile wznoszenia/opadania.
- **Kurs** (`heading_*`): srednia i **zmiennosc kursu** -- duza zmiennosc = manewry/krazenie.
- **Zasieg trasy** (`lat_range`, `lon_range`): rozpietosc geograficzna -- dlugosc trasy.
- **Pokrycie** (`measurements_count`, `time_span`): jak dlugo lot byl obserwowany.

> Uwaga metodologiczna: `heading_std` liczymy jako zwykle odchylenie stopni, wiec lot przecinajacy
> kierunek 0/360 stopni moze miec zawyzona wartosc -- traktujemy je jako *zgrubny* wskaznik
> zmiennosci kursu, wystarczajacy do wykrywania anomalii.
""")

code(r"""
flights_df = df.groupby(["icao24", "callsign"]).agg(
    alt_mean=("geoaltitude", "mean"),
    alt_max=("geoaltitude", "max"),
    alt_min=("geoaltitude", "min"),
    alt_std=("geoaltitude", "std"),
    alt_range=("geoaltitude", np.ptp),
    vel_mean=("velocity", "mean"),
    vel_max=("velocity", "max"),
    vel_std=("velocity", "std"),
    vert_mean=("vertrate", "mean"),
    vert_max=("vertrate", "max"),
    vert_min=("vertrate", "min"),
    vert_std=("vertrate", "std"),
    heading_mean=("heading", "mean"),
    heading_std=("heading", "std"),
    lat_range=("lat", np.ptp),
    lon_range=("lon", np.ptp),
    measurements_count=("time", "count"),
    time_span=("time", np.ptp),
)
flights_df.reset_index(inplace=True)
flights_df["flight_id"] = flights_df["icao24"] + "/" + flights_df["callsign"]
flights_df.drop(columns=["icao24", "callsign"], inplace=True)
flights_df.set_index("flight_id", inplace=True)
flights_df.fillna(0, inplace=True)  # std/ptp = NaN dla lotow z 1 pomiarem -> 0

FEATURES = list(FEATURE_INFO.keys())
print(f"Macierz cech: {flights_df.shape[0]:,} lotow x {flights_df.shape[1]} cech")
flights_df.head()
""")

code(r"""
flights_df.describe().T.round(2)
""")

# ---------------------------------------------------------------- EDA
md(r"""
## 5. Eksploracja cech

### 5.1 Rozklady wybranych cech
Wiekszosc lotow to typowe przeloty rejsowe (wysoko, szybko, prosto), ale rozklady maja dlugie
ogony -- to wlasnie kandydaci na anomalie. Histogramy sa interaktywne (hover = liczebnosc).
""")

code(r"""
from plotly.subplots import make_subplots

cols_to_plot = ["alt_mean", "vel_mean", "heading_std", "alt_range",
                "vert_std", "measurements_count"]
fig_hist = make_subplots(rows=2, cols=3,
                         subplot_titles=[FEATURE_INFO[c] for c in cols_to_plot])
for i, c in enumerate(cols_to_plot):
    r, cc = i // 3 + 1, i % 3 + 1
    fig_hist.add_trace(go.Histogram(x=flights_df[c], nbinsx=40, name=c,
                                    marker_color="#4C78A8"), row=r, col=cc)
fig_hist.update_layout(height=600, showlegend=False,
                       title_text="Rozklady wybranych cech lotow")
save_fig(fig_hist, "02_rozklady_cech")
fig_hist.show()
""")

md(r"""
### 5.2 Korelacje miedzy cechami
Mapa cieplna korelacji pokazuje, ktore cechy niosa podobna informacje (np. `alt_mean` i
`vel_mean` sa skorelowane -- wyzej znaczy szybciej). To uzasadnia uzycie **redukcji wymiaru**:
18 cech da sie sciesnic do 2 wymiarow bez utraty glownej struktury.
""")

code(r"""
corr = flights_df[FEATURES].corr()
fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto",
                     title="Korelacja miedzy cechami lotu")
fig_corr.update_layout(height=700, width=820)
save_fig(fig_corr, "03_korelacje")
fig_corr.show()
""")

md(r"""
### 5.3 Cechy wrazliwe na anomalie
Trzy cechy szczegolnie roznicuja loty nietypowe: **zmiennosc kursu** (krazenie/manewry),
**zakres wysokosci** (pelne wznoszenie+opadanie vs sam przelot) i **srednia predkosc**
(smiglowce/GA vs odrzutowce). Box-ploty pokazuja mediane i wartosci odstajace (punkty).
""")

code(r"""
box_cols = ["heading_std", "alt_range", "vel_mean"]
fig_box = make_subplots(rows=1, cols=3,
                        subplot_titles=[FEATURE_INFO[c] for c in box_cols])
for i, c in enumerate(box_cols):
    fig_box.add_trace(go.Box(y=flights_df[c], name=FEATURE_INFO[c],
                             boxpoints="outliers", marker_color="#E45756"),
                      row=1, col=i + 1)
fig_box.update_layout(height=420, showlegend=False,
                      title_text="Rozproszenie cech wrazliwych na anomalie")
save_fig(fig_box, "04_boxploty")
fig_box.show()
""")

# ---------------------------------------------------------------- STANDARYZACJA + HELPER
md(r"""
## 6. Redukcja wymiaru -- loty w przestrzeni 2D

Cechy maja rozne jednostki i skale, wiec najpierw je **standaryzujemy** (`StandardScaler`).
Nastepnie rzutujemy 18-wymiarowe wektory na plaszczyzne kilkoma metodami:

| metoda | co zachowuje |
|---|---|
| **PCA** | strukture *globalna* (kierunki najwiekszej wariancji), liniowa |
| **UMAP** | strukture *lokalna* i globalna, sasiedztwa |
| **PaCMAP** | balans struktury lokalnej i globalnej |
| **TriMAP** | strukture globalna przez trojki punktow |
| **t-SNE** (FIt-SNE / openTSNE) | bardzo dobre rozdzielenie *lokalnych* skupisk |

Wymog projektu (min. 2 metody) jest spelniony z naddatkiem. Kazdy wykres jest interaktywny:
**kolor = srednia wysokosc**, **rozmiar = liczba pomiarow**, a hover pokazuje parametry lotu.
""")

code(r"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(flights_df[FEATURES])
print("Macierz do redukcji wymiaru:", X.shape)

HOVER_COLS = ["alt_mean", "vel_mean", "vert_std", "heading_std",
              "alt_range", "measurements_count", "time_span"]


def embedding_figure(emb, color, title, color_label, discrete=False, size_col="measurements_count"):
    # Buduje interaktywny scatter 2D z bogatym hover (flight_id + parametry lotu)
    plot_df = flights_df.reset_index()[["flight_id"] + HOVER_COLS].copy()
    plot_df["x"], plot_df["y"] = emb[:, 0], emb[:, 1]
    plot_df["color"] = list(color)
    size = plot_df[size_col].clip(lower=1)
    common = dict(
        x="x", y="y", hover_name="flight_id",
        hover_data={c: ":.1f" for c in HOVER_COLS},
        size=size, size_max=14, opacity=0.75,
        render_mode="svg",  # bez WebGL -> dziala wszedzie (projektory, zdalny pulpit)
    )
    if discrete:
        fig = px.scatter(plot_df, color=plot_df["color"].astype(str),
                         labels={"color": color_label}, **common)
    else:
        fig = px.scatter(plot_df, color="color",
                         color_continuous_scale="Viridis",
                         labels={"color": color_label}, **common)
    fig.update_layout(title=title, height=560,
                      xaxis_title="wymiar 1", yaxis_title="wymiar 2")
    return fig
""")

md(r"""
### 6.1 PCA
PCA jest liniowa i szybka. Wykres osypiska (scree) pokazuje, ile wariancji wyjasniaja kolejne
skladowe -- pierwsze 2-3 skladowe tlumacza wiekszosc zroznicowania lotow.
""")

code(r"""
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X)
evr = pca.explained_variance_ratio_

fig_scree = go.Figure()
fig_scree.add_bar(x=[f"PC{i+1}" for i in range(len(evr))], y=evr, name="wariancja")
fig_scree.add_scatter(x=[f"PC{i+1}" for i in range(len(evr))], y=np.cumsum(evr),
                      mode="lines+markers", name="skumulowana")
fig_scree.update_layout(title="PCA -- wyjasniona wariancja", height=380,
                        yaxis_title="udzial wariancji")
save_fig(fig_scree, "05_pca_scree")
fig_scree.show()
""")

code(r"""
fig_pca = embedding_figure(X_pca, flights_df["alt_mean"],
                           "PCA -- loty w 2D (kolor = srednia wysokosc)", "Wys. [m]")
save_fig(fig_pca, "06_pca")
fig_pca.show()
""")

md(r"""
### 6.2 UMAP
""")
code(r"""
import umap

X_umap = umap.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
fig_umap = embedding_figure(X_umap, flights_df["alt_mean"],
                            "UMAP -- loty w 2D (kolor = srednia wysokosc)", "Wys. [m]")
save_fig(fig_umap, "07_umap")
fig_umap.show()
""")

md(r"""
### 6.3 PaCMAP
""")
code(r"""
import pacmap

X_pacmap = pacmap.PaCMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
fig_pacmap = embedding_figure(X_pacmap, flights_df["alt_mean"],
                              "PaCMAP -- loty w 2D (kolor = srednia wysokosc)", "Wys. [m]")
save_fig(fig_pacmap, "08_pacmap")
fig_pacmap.show()
""")

md(r"""
### 6.4 TriMAP
""")
code(r"""
import trimap

X_trimap = trimap.TRIMAP().fit_transform(X.astype(np.float32))
fig_trimap = embedding_figure(X_trimap, flights_df["alt_mean"],
                              "TriMAP -- loty w 2D (kolor = srednia wysokosc)", "Wys. [m]")
save_fig(fig_trimap, "09_trimap")
fig_trimap.show()
""")

md(r"""
### 6.5 t-SNE (FIt-SNE z fallbackiem na openTSNE)
FIt-SNE wymaga kompilacji (biblioteka FFTW). Jesli nie jest dostepny, uzywamy rownowaznego,
szybkiego `openTSNE` (ta sama rodzina metod, interpolowany t-SNE).
""")
code(r"""
X64 = np.ascontiguousarray(X, dtype=np.float64)
try:
    import fitsne
    X_tsne = fitsne.FItSNE(X64)
    tsne_label = "FIt-SNE"
except Exception as e:
    from openTSNE import TSNE
    X_tsne = np.asarray(TSNE(n_components=2, n_jobs=-1, random_state=RANDOM_STATE).fit(X64))
    tsne_label = "openTSNE (fallback)"
    print("FIt-SNE niedostepny -> uzyto openTSNE. Powod:", type(e).__name__)

fig_tsne = embedding_figure(X_tsne, flights_df["alt_mean"],
                            f"t-SNE [{tsne_label}] -- loty w 2D (kolor = srednia wysokosc)",
                            "Wys. [m]")
save_fig(fig_tsne, "10_tsne")
fig_tsne.show()
""")

md(r"""
**Porownanie metod.** PCA daje rozciagniety, ciagly uklad (gradient wysokosci/predkosci),
dobrze oddaje *globalne* trendy, ale slabiej rozdziela male grupy. UMAP, PaCMAP i t-SNE tworza
wyrazne *skupiska* (np. osobno smiglowce/GA, osobno odrzutowce rejsowe) i izolowane punkty na
obrzezach -- to czesto wlasnie anomalie. TriMAP plasuje sie posrodku. Do wykrywania pojedynczych
nietypowych lotow najlepiej czytelne sa metody nieliniowe (UMAP/t-SNE).
""")

# ---------------------------------------------------------------- ANOMALIE
md(r"""
## 7. Wykrywanie anomalii

Stosujemy trzy niezalezne, nienadzorowane detektory wartosci odstajacych i porownujemy wyniki:

- **Isolation Forest** -- izoluje punkty losowymi cieciami; anomalie izoluja sie szybciej.
- **Local Outlier Factor (LOF)** -- porownuje lokalna gestosc punktu z sasiadami.
- **One-Class SVM** -- uczy sie "obwiedni" danych normalnych.

Zakladamy `contamination = 5%`. Liczymy tez **konsensus** -- ile metod uznalo lot za anomalie
(0-3). Loty wskazane przez >=2 metody traktujemy jako najpewniejsze anomalie.
""")

code(r"""
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

CONTAMINATION = 0.05

iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
lof = LocalOutlierFactor(contamination=CONTAMINATION)
svm = OneClassSVM(nu=CONTAMINATION)

flights_df["iso_forest"] = iso.fit_predict(X)
flights_df["lof"] = lof.fit_predict(X)
flights_df["one_class_svm"] = svm.fit_predict(X)
# ciagly wynik "nietypowosci" (im wyzszy, tym bardziej anomalia)
flights_df["iso_score"] = -iso.decision_function(X)

method_cols = ["iso_forest", "lof", "one_class_svm"]
flights_df["n_methods_anom"] = (flights_df[method_cols] == -1).sum(axis=1)
flights_df["is_anomaly"] = flights_df["n_methods_anom"] >= 2

summary = pd.DataFrame({
    "Isolation Forest": [(flights_df["iso_forest"] == -1).sum()],
    "LOF": [(flights_df["lof"] == -1).sum()],
    "One-Class SVM": [(flights_df["one_class_svm"] == -1).sum()],
    "Konsensus (>=2 metody)": [int(flights_df["is_anomaly"].sum())],
}, index=["liczba anomalii"])
print("Wykryte anomalie wg metody:")
summary
""")

md(r"""
### 7.1 Zgodnosc metod
Ile lotow zostalo oznaczonych przez 0, 1, 2 lub 3 metody naraz? Im wiecej metod sie zgadza,
tym pewniejsza anomalia.
""")
code(r"""
counts = flights_df["n_methods_anom"].value_counts().sort_index()
fig_consensus = px.bar(x=counts.index.astype(str), y=counts.values,
                       labels={"x": "liczba metod wskazujacych anomalie", "y": "liczba lotow"},
                       text=counts.values, title="Zgodnosc detektorow anomalii")
fig_consensus.update_traces(marker_color="#54A24B", textposition="outside")
fig_consensus.update_layout(height=380, yaxis_type="log")
save_fig(fig_consensus, "11_konsensus")
fig_consensus.show()
""")

md(r"""
### 7.2 Anomalie na mapie redukcji wymiaru
Nanosimy wynik na rzut UMAP. Kolor = liczba metod wskazujacych anomalie. Anomalie (zolte/czerwone)
leza na obrzezach i w rzadkich rejonach przestrzeni -- zgodnie z intuicja.
""")
code(r"""
fig_anom = embedding_figure(
    X_umap, flights_df["n_methods_anom"],
    "Anomalie na rzucie UMAP (kolor = liczba metod 0-3)", "Liczba metod",
    discrete=True)
save_fig(fig_anom, "12_anomalie_umap")
fig_anom.show()
""")

# ---------------------------------------------------------------- INTERPRETACJA
md(r"""
## 8. Interpretacja anomalii -- *jak* odstaja?

Sama liczba anomalii nic nie mowi. Sprawdzmy, **ktore cechy** i **jak bardzo** odroznia loty
nietypowe od reszty. Dla kazdej cechy liczymy sredni **z-score** (o ile odchylen standardowych
populacji odstaje przecietna anomalia). Im wyzszy slupek, tym mocniej dana cecha napedza
nietypowosc.
""")
code(r"""
anom = flights_df[flights_df["is_anomaly"]]
normal = flights_df[~flights_df["is_anomaly"]]
print(f"Anomalie (konsensus): {len(anom)} lotow | Normalne: {len(normal)} lotow")

pop_mean = flights_df[FEATURES].mean()
pop_std = flights_df[FEATURES].std().replace(0, np.nan)
z_anom = ((anom[FEATURES].mean() - pop_mean) / pop_std).abs().sort_values(ascending=False)

fig_z = px.bar(x=z_anom.values, y=[FEATURE_INFO[c] for c in z_anom.index],
               orientation="h", text=z_anom.round(2).values,
               labels={"x": "sredni |z-score| anomalii", "y": ""},
               title="Ktore cechy najbardziej odroznia anomalie od reszty")
fig_z.update_traces(marker_color="#E45756", textposition="outside")
fig_z.update_layout(height=560, yaxis={"categoryorder": "total ascending"})
save_fig(fig_z, "13_zscore_cech")
fig_z.show()
""")

md(r"""
### 8.1 Tabela: anomalie vs reszta
Konkretne liczby: srednia cechy dla anomalii vs mediana populacji oraz percentyl, w ktorym
plasuja sie anomalie. Wartosci typu "p99" oznaczaja, ze przecietna anomalia jest skrajna.
""")
code(r"""
compare = pd.DataFrame({
    "cecha": [FEATURE_INFO[c] for c in FEATURES],
    "mediana (wszystkie)": flights_df[FEATURES].median().values,
    "srednia (anomalie)": anom[FEATURES].mean().values,
    "srednia (normalne)": normal[FEATURES].mean().values,
}, index=FEATURES)
# percentyl, w ktorym lezy srednia anomalia wzgledem calej populacji
compare["percentyl anomalii"] = [
    round((flights_df[c] <= anom[c].mean()).mean() * 100, 1) for c in FEATURES
]
compare.round(2).sort_values("percentyl anomalii", ascending=False)
""")

md(r"""
### 8.2 Konkretne przypadki -- co je czyni nietypowymi
Wybieramy kilka *reprezentatywnych* anomalii i pokazujemy ich wartosci na tle calej populacji
(mediana i 99. percentyl). To pozwala powiedziec wprost np. *"ten lot ma zmiennosc kursu 6x
wieksza niz typowy lot"*.
""")
code(r"""
def describe_case(flight_id, feats):
    row = flights_df.loc[flight_id]
    recs = []
    for c in feats:
        med = flights_df[c].median()
        p99 = flights_df[c].quantile(0.99)
        ratio = row[c] / med if med else np.inf
        recs.append({
            "lot": flight_id, "cecha": FEATURE_INFO[c],
            "wartosc": round(row[c], 1), "mediana": round(med, 1),
            "p99 populacji": round(p99, 1),
            "x mediany": (round(ratio, 1) if np.isfinite(ratio) else "—"),
        })
    return pd.DataFrame(recs)


# automatyczny wybor ciekawych anomalii z dluga trajektoria (do narysowania)
plottable = anom[anom["measurements_count"] >= 80]
cases = {
    "krazenie / manewry (max heading_std)": plottable["heading_std"].idxmax(),
    "pelny profil wys. (max alt_range)": plottable["alt_range"].idxmax(),
    "ekstremalna pred. pionowa (max vert_std)": plottable["vert_std"].idxmax(),
    "najwyzszy wynik Isolation Forest": plottable["iso_score"].idxmax(),
}
cases = {k: v for k, v in cases.items()}  # zachowaj kolejnosc
print("Wybrane przypadki:")
for k, v in cases.items():
    print(f"  - {k}: {v}")

key_feats = ["heading_std", "alt_range", "vert_std", "vel_mean",
             "alt_mean", "measurements_count"]
pd.concat([describe_case(fid, key_feats) for fid in dict.fromkeys(cases.values())],
          ignore_index=True)
""")

md(r"""
### 8.3 Trajektorie nietypowych lotow
Najlepszy dowod nietypowosci to ksztalt trasy. Rysujemy faktyczne sciezki lat/lon wybranych
anomalii (kolor = czas, hover = wysokosc/predkosc). Widac np. **petle/krazenie** zamiast prostego
przelotu albo nietypowe profile wysokosci.
""")
code(r"""
sel_ids = list(dict.fromkeys(cases.values()))
fig_traj = make_subplots(
    rows=2, cols=2, specs=[[{"type": "scattergeo"}] * 2] * 2,
    subplot_titles=[f"{lbl}<br>{fid}" for lbl, fid in cases.items()],
)
positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
for (lbl, fid), (r, c) in zip(cases.items(), positions):
    t = df[df["flight_id"] == fid].sort_values("time")
    fig_traj.add_trace(go.Scattergeo(
        lat=t["lat"], lon=t["lon"], mode="lines+markers",
        marker=dict(size=4, color=t["geoaltitude"], colorscale="Turbo",
                    showscale=False),
        line=dict(width=1, color="rgba(80,80,80,0.4)"),
        name=fid,
        text=[f"wys: {a:.0f} m<br>pred: {v:.0f} m/s" for a, v in
              zip(t["geoaltitude"], t["velocity"])],
        hovertemplate="%{text}<extra></extra>",
    ), row=r, col=c)
fig_traj.update_geos(fitbounds="locations", showland=True,
                     landcolor="#EAEAEA", showcountries=True)
fig_traj.update_layout(height=720, showlegend=False,
                       title_text="Trajektorie wybranych anomalii (kolor punktow = wysokosc)")
save_fig(fig_traj, "14_trajektorie_anomalii")
fig_traj.show()
""")

# ---------------------------------------------------------------- KLASTROWANIE
md(r"""
## 9. Element rozszerzony -- klastrowanie i typy lotow

Anomalie to "co odstaje". Uzupelniajaco pytamy: *jakie sa typowe rodzaje lotow?* Grupujemy loty
metoda **K-Means** (liczbe klastrow dobieramy wskaznikiem sylwetki) oraz **HDBSCAN** (gestosciowa,
sama oznacza szum = -1, co naturalnie pokrywa sie z anomaliami).
""")
code(r"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sil = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
    sil[k] = silhouette_score(X, km.labels_, sample_size=4000, random_state=RANDOM_STATE)

best_k = max(sil, key=sil.get)
fig_sil = px.line(x=list(sil.keys()), y=list(sil.values()), markers=True,
                  labels={"x": "liczba klastrow k", "y": "wskaznik sylwetki"},
                  title=f"Dobor liczby klastrow (najlepsze k = {best_k})")
fig_sil.update_layout(height=360)
save_fig(fig_sil, "15_silhouette")
fig_sil.show()
""")

code(r"""
import hdbscan

kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X)
flights_df["cluster"] = kmeans.labels_

hdb = hdbscan.HDBSCAN(min_cluster_size=50).fit(X)
flights_df["hdbscan"] = hdb.labels_
n_hdb = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
print(f"K-Means: {best_k} klastrow | HDBSCAN: {n_hdb} klastrow + "
      f"{(hdb.labels_ == -1).sum()} punktow szumu (-1)")
""")

md(r"""
### 9.1 Profile klastrow i automatyczne nazwy typow
Dla kazdego klastra liczymy srednie cechy i nadajemy **opisowy typ** na podstawie prostych regul
(wysokosc, predkosc pionowa, zmiennosc kursu, zasieg). To zamienia abstrakcyjne numery klastrow
w czytelne kategorie lotow.
""")
code(r"""
def label_cluster(p):
    if p["measurements_count"] < 60 or p["time_span"] < 600:
        return "krotkie / fragmentaryczne"
    if p["heading_std"] > 60:
        return "manewrowe / krazace"
    if p["alt_mean"] > 9000 and p["vert_std"] < 2:
        return "wysokie przeloty rejsowe"
    if p["vert_std"] > 4 or p["alt_range"] > 8000:
        return "wznoszenie / zniżanie"
    if p["vel_mean"] < 120:
        return "wolne / niskie (GA / smiglowce)"
    return "srednie przeloty"


profiles = flights_df.groupby("cluster")[FEATURES].mean()
profiles["liczba_lotow"] = flights_df.groupby("cluster").size()
profiles["% anomalii"] = (flights_df.groupby("cluster")["is_anomaly"].mean() * 100).round(1)
profiles["typ"] = profiles.apply(label_cluster, axis=1)
cluster_names = profiles["typ"].to_dict()
flights_df["cluster_name"] = flights_df["cluster"].map(cluster_names)

cols_show = ["typ", "liczba_lotow", "% anomalii", "alt_mean", "vel_mean",
             "vert_std", "heading_std", "alt_range", "time_span"]
profiles[cols_show].round(1)
""")

md(r"""
### 9.2 Klastry na rzucie UMAP
Te same loty co wczesniej, ale kolor = typ lotu. Skupiska z redukcji wymiaru pokrywaja sie z
klastrami -- co potwierdza, ze cechy faktycznie oddaja rodzaje lotow.
""")
code(r"""
fig_clusters = embedding_figure(
    X_umap, flights_df["cluster_name"],
    "Typy lotow (K-Means) na rzucie UMAP", "Typ lotu", discrete=True)
save_fig(fig_clusters, "16_klastry_umap")
fig_clusters.show()
""")

md(r"""
### 9.3 Profil klastrow -- mapa cieplna
Znormalizowane (z-score) srednie cechy w klastrach. Czytelnie widac "podpis" kazdego typu lotu:
np. typ rejsowy ma wysokie `alt_mean`/`vel_mean`, a typ manewrowy -- wysokie `heading_std`.
""")
code(r"""
prof_z = (profiles[FEATURES] - flights_df[FEATURES].mean()) / flights_df[FEATURES].std()
prof_z.index = [f"{i}: {cluster_names[i]}" for i in prof_z.index]
fig_profile = px.imshow(prof_z[FEATURES], color_continuous_scale="RdBu_r",
                        zmin=-2, zmax=2, aspect="auto", text_auto=".1f",
                        labels={"x": "cecha", "y": "klaster", "color": "z-score"},
                        title="Profile typow lotow (z-score cech)")
fig_profile.update_layout(height=420, width=950)
save_fig(fig_profile, "17_profile_klastrow")
fig_profile.show()
""")

# ---------------------------------------------------------------- WNIOSKI
md(r"""
## 10. Wnioski i kierunki rozwoju

**Wnioski.**
- 18-wymiarowy wektor cech skutecznie opisuje lot; redukcja wymiaru (UMAP/PaCMAP/t-SNE)
  ujawnia czytelne skupiska odpowiadajace **typom lotow**, a PCA dobrze oddaje globalny gradient
  wysokosc-predkosc.
- Trzy niezalezne detektory zgodnie wskazuja ~5% lotow jako nietypowe; **konsensus** (>=2 metody)
  wyostrza najpewniejsze przypadki.
- Anomalie napedzane sa glownie przez **zmiennosc kursu**, **zakres/odchylenie wysokosci** i
  **predkosc pionowa** -- czyli loty krazace, manewrowe oraz o nietypowym profilu pionowym.
- Klastrowanie nadaje strukturze interpretacje: przeloty rejsowe, wznoszenie/zniżanie, loty
  wolne/niskie (GA, smiglowce) oraz fragmentaryczne; odsetek anomalii rozni sie miedzy typami.

**Ograniczenia.**
- Tylko jedna godzina danych; loty na granicy okna sa obciete (stad cechy `*_count`/`time_span`).
- `heading_std` nie uwzglednia cyklicznosci kata 0/360 stopni.
- Cechy sa zagregowane -- traca informacje o *kolejnosci* zdarzen w locie.

**Kierunki rozwoju.**
- Cechy sekwencyjne / ksztaltu trajektorii (DTW, autoenkodery sekwencji) zamiast samych agregatow.
- Klasyfikacja *nadzorowana* typow lotow po recznym oznaczeniu probki.
- Wlaczenie `squawk` 7500/7600/7700 (porwanie / awaria radia / ogolna awaria) jako etykiet odniesienia.
- Wieksza skala (cala doba / wiele dni) i **dashboard na zywo** (np. Dash) do eksploracji.
""")

md(r"""
## 11. Eksport interaktywnych figur do prezentacji
Wszystkie kluczowe wykresy zostaly zapisane do `presentation/figures/*.html` (wywolania
`save_fig(...)` powyzej) i sa osadzone w prezentacji `presentation/index.html` (reveal.js).
""")
code(r"""
import glob
exported = sorted(glob.glob(os.path.join(EXPORT_DIR, "*.html")))
print(f"Wyeksportowano {len(exported)} interaktywnych figur:")
for p in exported:
    print("  -", os.path.basename(p))
""")

# ---------------------------------------------------------------- ZAPIS
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (wdzd)", "language": "python", "name": "wdzd-venv"},
    "language_info": {"name": "python", "version": "3.12"},
}
with open("src/lol.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Zapisano src/lol.ipynb ({len(cells)} komorek)")
