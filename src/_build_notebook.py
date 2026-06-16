"""Buduje notebook src/main.ipynb: analiza nietypowych lotow (OpenSky).

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

**Cel.** Wykryc *nietypowe loty* po parametrach trajektorii (wysokosc, predkosc, kurs, dlugosc trasy).
Kazdy lot to **wektor cech**; szukamy lotow odstajacych od reszty.

**Dane.** [OpenSky Network](https://opensky-network.org/) -- *state vectors* z jednej godziny
(`2022-06-27`, `23:00` UTC). Jeden rekord = stan samolotu w jednej sekundzie.

**Plan:** czyszczenie -> mapa ruchu -> cechy -> redukcja wymiaru (PCA, UMAP, PaCMAP, TriMAP, t-SNE)
-> anomalie (Isolation Forest, LOF, One-Class SVM + konsensus) -> interpretacja -> klastrowanie.

> Wykresy sa **interaktywne** (Plotly): hover = parametry lotu, zoom, filtr legendy.
""")

# ---------------------------------------------------------------- IMPORTY / SETUP
md(r"""
## 0. Konfiguracja srodowiska

Biblioteki, styl Plotly, slownik opisow cech i katalog na eksport figur do prezentacji.
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
    "heading_std": "Zmiennosc kursu (cyrkularna) [stopnie]",
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

Godzinna probka *state vectors*. Wiersz = stan samolotu (`icao24`) w jednej sekundzie. Kolumny:

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

> **Uwaga:** plik (`~800 MB`) jest w `.gitignore` -- pobieranie opisuje `README.md`.
""")

code(r"""
df = pd.read_json(DATA_PATH)
print(f"Wczytano {len(df):,} rekordow (state vectors) x {df.shape[1]} kolumn")
df.head()
""")

# ---------------------------------------------------------------- CZYSZCZENIE
md(r"""
## 2. Jakosc i czyszczenie danych

Czesc rekordow ma braki (brak GPS, nieprzypisany `squawk`). Sprawdzamy skale brakow i usuwamy
niekompletne rekordy -- cechy wymagaja kompletu pomiarow.
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

Mapa probki pozycji (kolor = wysokosc). Widac korytarze nad Ameryka i Europa oraz nisko lecace
samoloty przy lotniskach. *Hover = parametry lotu; mozna przyblizac.*
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

Agregujemy pomiary jednego lotu (`icao24` + `callsign`) w **jeden wektor cech**, w blokach:

- **Wysokosc** (`alt_*`): srednia, min, max, odchylenie, zakres -- profil pionowy.
- **Predkosc** (`vel_*`): srednia, max, zmiennosc -- rejsowa vs zmienna.
- **Pionowa** (`vert_*`): srednia/min/max/odchylenie -- wznoszenie/opadanie.
- **Kurs** (`heading_*`): srednia i **zmiennosc kursu** -- manewry/krazenie.
- **Trasa** (`lat_range`, `lon_range`): rozpietosc geograficzna.
- **Pokrycie** (`measurements_count`, `time_span`): dlugosc obserwacji.

> Kurs jest **cykliczny** (0 = 360 stopni), wiec `heading_std` liczymy **cyrkularnie**: prosty lot
> na polnoc ma ~0, a tylko faktyczne krazenie daje wysoka wartosc.
""")

code(r"""
def circ_std_deg(angles):
    # Cyrkularne odchylenie standardowe kursu [stopnie] -- odporne na przejscie 0/360.
    r = np.deg2rad(np.asarray(angles, dtype=float))
    R = np.hypot(np.cos(r).mean(), np.sin(r).mean())
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-9)))))


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
    heading_std=("heading", circ_std_deg),
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
Wiekszosc lotow to typowe przeloty rejsowe, ale rozklady maja dlugie ogony -- to kandydaci na anomalie.
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
Skorelowane cechy (np. `alt_mean` i `vel_mean`) niosa podobna informacje -- to uzasadnia
**redukcje wymiaru** z 18 cech do 2.
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
Loty nietypowe roznicuja zwlaszcza **zmiennosc kursu**, **zakres wysokosci** i **srednia predkosc**.
Box-ploty pokazuja mediane i wartosci odstajace.
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

Cechy **standaryzujemy** (`StandardScaler`), potem rzutujemy 18 wymiarow na plaszczyzne kilkoma metodami:

| metoda | co zachowuje |
|---|---|
| **PCA** | strukture *globalna* (kierunki najwiekszej wariancji), liniowa |
| **UMAP** | strukture *lokalna* i globalna, sasiedztwa |
| **PaCMAP** | balans struktury lokalnej i globalnej |
| **TriMAP** | strukture globalna przez trojki punktow |
| **t-SNE** (FIt-SNE / openTSNE) | bardzo dobre rozdzielenie *lokalnych* skupisk |

Wymog (min. 2 metody) spelniony z naddatkiem. Na kazdym wykresie: **kolor = wysokosc**,
**rozmiar = liczba pomiarow**.
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
PCA jest liniowa i szybka. Wykres osypiska: pierwsze 2-3 skladowe tlumacza wiekszosc wariancji.
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
from sklearn.neighbors import NearestNeighbors

# UWAGA: domyslny backend KNN w TriMAP (Annoy) zwraca w tym srodowisku (numpy 2.x)
# bledne sasiedztwa -- wszystkie odleglosci wychodza 0, skala `sig` spada do zera,
# a wagi trojek rosna do ~1e6, przez co embedding rozbiega sie do milionow (na wykresie
# widac wtedy "jeden punkt"). Liczymy wiec KNN samodzielnie (sklearn) i podajemy gotowe
# sasiedztwa przez `knn_tuple`, co omija Annoy i daje stabilny wynik.
N_INLIERS = 12
n_knn = min(N_INLIERS + 50 + 1, len(X))  # +50 zapasowych sasiadow (zob. trimap), +1 na siebie
knn_dist, knn_idx = NearestNeighbors(n_neighbors=n_knn).fit(X).kneighbors(X)

np.random.seed(RANDOM_STATE)  # TriMAP losuje trojki -> ustalamy ziarno dla powtarzalnosci
X_trimap = trimap.TRIMAP(
    n_inliers=N_INLIERS, verbose=False,
    knn_tuple=(knn_idx.astype(np.int32), knn_dist.astype(np.float32)),
).fit_transform(X.astype(np.float32))
fig_trimap = embedding_figure(X_trimap, flights_df["alt_mean"],
                              "TriMAP -- loty w 2D (kolor = srednia wysokosc)", "Wys. [m]")
save_fig(fig_trimap, "09_trimap")
fig_trimap.show()
""")

md(r"""
### 6.5 t-SNE (FIt-SNE z fallbackiem na openTSNE)
FIt-SNE wymaga kompilacji (FFTW); gdy niedostepny, uzywamy rownowaznego `openTSNE`.
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
**Porownanie metod.** PCA oddaje *globalny* gradient, ale slabo rozdziela male grupy. UMAP, PaCMAP
i t-SNE tworza wyrazne *skupiska* i izoluja punkty na obrzezach (czesto anomalie); TriMAP jest
posrodku. Do pojedynczych anomalii najczytelniejsze sa metody nieliniowe.
""")

# ---------------------------------------------------------------- ANOMALIE
md(r"""
## 7. Wykrywanie anomalii

Trzy niezalezne, nienadzorowane detektory (`contamination = 5%`):

- **Isolation Forest** -- anomalie izoluja sie szybciej losowymi cieciami.
- **Local Outlier Factor (LOF)** -- porownuje lokalna gestosc z sasiadami.
- **One-Class SVM** -- uczy sie obwiedni danych normalnych.

**Konsensus** = ile metod (0-3) wskazalo lot; >=2 metody = najpewniejsze anomalie.
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
Ile lotow oznaczyly 0, 1, 2 lub 3 metody naraz. Wiecej zgodnych metod = pewniejsza anomalia.
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
Rzut UMAP, kolor = liczba metod. Anomalie leza na obrzezach i w rzadkich rejonach przestrzeni.
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

Ktore cechy odroznia anomalie od reszty? Dla kazdej liczymy sredni **z-score** anomalii
(o ile odchylen std odstaje od populacji). Wyzszy slupek = cecha mocniej napedza nietypowosc.
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
Srednia cechy anomalii vs mediana populacji oraz percentyl, w ktorym plasuja sie anomalie.
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
Kilka reprezentatywnych anomalii na tle populacji (mediana, p99) -- np. *"zmiennosc kursu 6x
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
### 8.3 Trajektorie nietypowych lotow (rzut poziomy)
Faktyczne sciezki lat/lon wybranych anomalii (kolor = wysokosc). Widac np. **petle/krazenie**
zamiast prostego przelotu.

> Mapa lat/lon pokazuje tylko ruch *poziomy*; cechy *pionowe* widac dopiero w sekcji 8.4.
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

md(r"""
### 8.4 Profile pionowe anomalii (wysokosc i predkosc pionowa w czasie)
Dla tych samych anomalii: **wysokosc** (os lewa) i **predkosc pionowa** (os prawa) w czasie.
Tu wprost widac np. gwaltowne oscylacje pred. pionowej (`max vert_std`) czy pelny profil
wznoszenia/zniżania (`max alt_range`), niewidoczny na mapie poziomej.
""")
code(r"""
fig_vprof = make_subplots(
    rows=2, cols=2, specs=[[{"secondary_y": True}] * 2] * 2,
    subplot_titles=[f"{lbl}<br>{fid}" for lbl, fid in cases.items()],
)
for (lbl, fid), (r, c) in zip(cases.items(), positions):
    t = df[df["flight_id"] == fid].sort_values("time")
    tmin = (t["time"] - t["time"].min()) / 60.0  # czas od poczatku [min]
    first = (r == 1 and c == 1)  # legenda tylko raz
    fig_vprof.add_trace(go.Scatter(
        x=tmin, y=t["geoaltitude"], mode="lines", line=dict(color="#4C78A8"),
        name="wysokosc [m]", legendgroup="alt", showlegend=first,
        hovertemplate="%{x:.1f} min<br>wys: %{y:.0f} m<extra></extra>"),
        row=r, col=c, secondary_y=False)
    fig_vprof.add_trace(go.Scatter(
        x=tmin, y=t["vertrate"], mode="lines", line=dict(color="#E45756"),
        name="pred. pionowa [m/s]", legendgroup="vert", showlegend=first,
        hovertemplate="%{x:.1f} min<br>pion: %{y:.1f} m/s<extra></extra>"),
        row=r, col=c, secondary_y=True)
fig_vprof.update_xaxes(title_text="czas [min]")
fig_vprof.update_yaxes(title_text="wys. [m]", secondary_y=False,
                       title_font_color="#4C78A8", tickfont_color="#4C78A8")
fig_vprof.update_yaxes(title_text="pion. [m/s]", secondary_y=True,
                       title_font_color="#E45756", tickfont_color="#E45756")
fig_vprof.update_layout(height=680,
                        title_text="Profile pionowe wybranych anomalii "
                                   "(niebieski = wysokosc, czerwony = predkosc pionowa)")
save_fig(fig_vprof, "14b_profile_pionowe")
fig_vprof.show()
""")

# ---------------------------------------------------------------- KLASTROWANIE
md(r"""
## 9. Element rozszerzony -- klastrowanie i typy lotow

*Jakie sa typowe rodzaje lotow?* Grupujemy **K-Means** (k z silhouette score) oraz **HDBSCAN**
(gestosciowa, szum = -1 pokrywa sie z anomaliami).
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
                  labels={"x": "liczba klastrow k", "y": "silhouette score"},
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
Dla kazdego klastra liczymy srednie cechy i nadajemy **opisowy typ** prostymi regulami --
zamiast numerow klastrow czytelne kategorie lotow.
""")
code(r"""
def label_cluster(p):
    if p["measurements_count"] < 60 or p["time_span"] < 600:
        return "krotkie / fragmentaryczne"
    if p["heading_std"] > 50:
        return "manewrowe / krazace"
    if p["alt_mean"] > 9000 and p["vert_std"] < 2:
        return "wysokie przeloty rejsowe"
    if p["vert_std"] > 4 or p["alt_range"] > 8000:
        # rozdzielamy po znaku sredniej pred. pionowej: + = wznoszenie, - = zniżanie
        return "wznoszenie" if p["vert_mean"] > 0 else "zniżanie"
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
Te same loty, kolor = typ lotu. Skupiska UMAP pokrywaja sie z klastrami -- cechy oddaja rodzaje lotow.
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
Srednie cechy (z-score) w klastrach -- "podpis" kazdego typu: rejsowy = wysokie `alt_mean`/`vel_mean`,
manewrowy = wysokie `heading_std`.
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
## 10. Wnioski

**Wnioski.**
- Wektor 18 cech opisuje lot; redukcja wymiaru ujawnia **typy lotow**, PCA -- gradient wysokosc-predkosc.
- Trzy detektory zgodnie wskazuja ~5% lotow; **konsensus** (>=2 metody) wyostrza najpewniejsze.
- Anomalie napedza **zmiennosc kursu**, **zakres wysokosci** i **predkosc pionowa** (loty krazace, manewrowe).
- Klastrowanie daje typy: rejsowe, wznoszenie/zniżanie, wolne/niskie (GA, smiglowce), fragmentaryczne.

**Ograniczenia.**
- Jedna godzina danych; loty na brzegu okna obciete.
- Cechy zagregowane -- traca kolejnosc zdarzen w locie.
""")

md(r"""
## 11. Eksport interaktywnych figur do prezentacji
Wykresy zapisane do `presentation/figures/*.html` i osadzone w `presentation/index.html` (reveal.js).
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
with open("src/main.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Zapisano src/main.ipynb ({len(cells)} komorek)")
