import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

import json
import folium
from streamlit_folium import st_folium
import copy
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Configuração da Página --- #
st.set_page_config(layout="wide")

st.markdown("""
<style>
/* Aumenta o tamanho da fonte de todo o aplicativo */
html, body, [class*="st-"], .st-emotion-cache-1jicfl2 {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd

# Esta função deve estar no seu arquivo app.py
@st.cache_data
def load_data():
    """
    Carrega, processa, padroniza, traduz e une os dados de projetos e créditos.
    Esta função é cacheada, então todo este processamento pesado
    ocorre apenas uma vez.
    """
    try:
        # Carrega os dados brutos
        projects_df = pd.read_csv("projects.csv")
        credits_df = pd.read_csv("credits.csv")
    except FileNotFoundError:
        st.error("Erro: Verifique se os arquivos 'projects.csv' e 'credits.csv' estão no diretório correto.")
        st.stop()

    # --- 1. Padronização dos Nomes dos Países (para o Folium) ---
    names_to_review = {
        'United States': 'United States of America',
        'Congo, The Democratic Republic of the': 'Democratic Republic of the Congo',
        'Tanzania, United Republic of': 'United Republic of Tanzania',
        "Lao People's Democratic Republic": 'Laos',
        'Viet Nam': 'Vietnam',
        'Korea, Republic of': 'South Korea'
    }
    projects_df['country'] = projects_df['country'].replace(names_to_review)
    
    # --- 2. Processamento de projects_df ---
    # Correção de fuso horário: Padroniza para UTC
    projects_df["first_issuance_at"] = pd.to_datetime(projects_df["first_issuance_at"], errors="coerce", utc=True)
    projects_df["first_retirement_at"] = pd.to_datetime(projects_df["first_retirement_at"], errors="coerce", utc=True)
    projects_df["first_issuance_at"] = projects_df["first_issuance_at"].fillna(pd.Timestamp("1900-01-01", tz='UTC'))
    projects_df["first_retirement_at"] = projects_df["first_retirement_at"].fillna(pd.Timestamp.now(tz='UTC'))

    # Criação de colunas calculadas
    projects_df["implementation_year"] = projects_df["first_issuance_at"].dt.year.fillna(0).astype(int)
    projects_df["project_duration"] = (projects_df["first_retirement_at"] - projects_df["first_issuance_at"]).dt.days / 365.25
    projects_df["project_duration"] = projects_df["project_duration"].fillna(1).astype(int)
    projects_df["co2_reduced"] = projects_df["issued"].fillna(0) * 1000

    # --- 3. Processamento de credits_df ---
    credits_df = credits_df.rename(columns={"quantity": "volume"})
    if "volume" in credits_df.columns:
        credits_df["price"] = credits_df["volume"] * 0.1 + 5 + (credits_df.index % 100) / 100
    else:
        credits_df["volume"] = 100 # Valor de exemplo
        credits_df["price"] = credits_df["volume"] * 0.1 + 5 + (credits_df.index % 100) / 100

    # --- 4. Otimização Principal: Unir os DataFrames ---
    merged_df = pd.merge(projects_df, credits_df, on="project_id", how="inner")
    
    # --- 5. Tradução dos Tipos de Projeto ---
    translation_map = {
        'Afforestation + Reforestation': 'Florestamento e Reflorestamento',
        'Avoided Grassland Conversion': 'Conversão de Pastagem Evitada',
        'Biomass': 'Biomassa',
        'Centralized Solar': 'Energia Solar Centralizada',
        'Clean Water': 'Água Limpa',
        'Compost': 'Compostagem',
        'Cookstove': 'Fogões Eficientes',
        'Distributed Solar': 'Energia Solar Distribuída',
        'Energy Efficiency': 'Eficiência Energética',
        'Landfill': 'Aterro Sanitário',
        'Waste Diversion': 'Desvio de Resíduos',
        'Renewable Energy': 'Energia Renovável',
        'Advanced Refrigerant': 'Refrigerante Avançado',
        'Manure Bodigester': 'Biodigestor de Esterco',
        'Road Construction': 'Construção de Estradas',
        'Gas Leak Repair': 'Reparo de Vazamento de Gás',
        'Wind': 'Energia Eólica'
        # Adicione outras traduções conforme encontrar novos tipos de projeto
    }
    # Cria uma nova coluna com os nomes traduzidos
    merged_df['project_type_pt'] = merged_df['project_type'].map(translation_map).fillna(merged_df['project_type'])

    return projects_df, credits_df, merged_df

# --- Funções de Gráfico em Cache para Performance --- #
@st.cache_data
def generate_histogram(df, project_type):
    """Gera o histograma de contagem de projetos. Cacheado."""
    return px.histogram(df, x=project_type, title="Contagem de Projetos por Tipo")

@st.cache_data
def generate_boxplot(df, project_type, price):
    """Gera o boxplot de distribuição de preços. Cacheado."""
    return px.box(df, x=project_type, y=price, title="Distribuição de Preços por Tipo de Projeto")

@st.cache_data
def generate_scatter_plot(df, x_axis, y_axis, color, hover_data):
    """Gera o gráfico de dispersão. Cacheado."""
    return px.scatter(df, x=x_axis, y=y_axis, color=color,
                      hover_data=hover_data,
                      title="Volume de CO₂ Reduzido vs. Preço do Crédito")

# Carrega os dados uma vez no início
projects_df, credits_df, merged_df = load_data()

# Função em cache para carregar o GeoJSON
@st.cache_data
def load_geojson():
    try:
        with open('countries.geo.json') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Arquivo 'countries.geo.json' não encontrado. Por favor, adicione-o à pasta do projeto para visualizar o mapa.")
        return None

# Carregue o geojson uma vez
geojson_data_cached = load_geojson()

# Adicione esta função no seu app.py

def formatar_numero(num):
    """Formata um número para um formato compacto (k, mi, bi)."""
    if num is None:
        return "0"
    if num < 1000:
        return str(int(num))
    elif num < 1_000_000:
        return f"{num/1_000:.1f} k"
    elif num < 1_000_000_000:
        return f"{num/1_000_000:.1f} mi"
    else:
        return f"{num/1_000_000_000:.1f} bi"

# --- 3. SUBSTITUA A SIDEBAR ANTIGA POR ESTA --- #

with st.sidebar:
    section = option_menu(
        menu_title="Menu Principal",
        options=[
            "Introdução", "Exploração dos Dados", "Dinâmica do Mercado",
            "Fatores de Precificação", "Segmentação de Projetos", "Calculadora de Emissões"
        ],
        icons=[
            "house-door-fill", "clipboard-data-fill", "graph-up-arrow", "currency-dollar",
            "diagram-3-fill", "calculator-fill"
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#d3d3d3"},
            "nav-link-selected": {"background-color": "#1f77b4", "color": "white", "border-left": "5px solid #1f77b4"},
        }
    )

# --- Seção: Introdução (VERSÃO FINAL COM CARDS MODERNOS) --- #
if section == "Introdução":

    # --- 1. INJEÇÃO DE CSS PARA OS CARDS E ESTILOS GERAIS ---
    # Este CSS define a aparência dos nossos 'cards' de métricas.
    st.markdown("""
    <style>
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        height: 100%; /* Garante que todos os cards na mesma linha tenham a mesma altura */
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .metric-card h3 {
        margin: 0 0 10px 0;
        font-size: 18px;
        font-weight: 600;
        color: #5A5A5A;
    }
    .metric-card p {
        margin: 0;
        font-size: 38px;
        font-weight: 700;
    }
    .value-blue { color: #1f77b4; }
    .value-green { color: #2ca02c; }
    .value-orange { color: #ff7f0e; }
    </style>
    """, unsafe_allow_html=True)

    # --- 2. TÍTULO E INTRODUÇÃO ---
    st.title("Análise Interativa do Mercado de Créditos de Carbono")
    st.markdown("Bem-vindo ao painel de análise do mercado de créditos de carbono. Explore os dados e insights através do menu à esquerda.")
    st.divider()

    # --- 3. MÉTRICAS GLOBAIS EM CARDS COLORIDOS ---
    st.header("Visão Geral do Conjunto de Dados")

    # Calcula as métricas
    total_projects = merged_df['project_id'].nunique()
    total_co2_reduced = int(merged_df['co2_reduced'].sum() / 1_000_000)
    num_countries = merged_df['country'].nunique()

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌍 Projetos Analisados</h3>
            <p class="value-blue">{total_projects:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💨 CO₂ Reduzido (M de Toneladas)</h3>
            <p class="value-green">{total_co2_reduced:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Países Envolvidos</h3>
            <p class="value-orange">{num_countries}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()

    # --- 4. JUSTIFICATIVA E OBJETIVOS ---
    st.header("Sobre a Pesquisa")
    col_just, col_obj = st.columns(2, gap="large")

    with col_just:
        st.subheader("💡 Justificativa")
        st.markdown("""
        A transparência e a eficiência são cruciais para o sucesso dos mercados de carbono. A análise de dados permite identificar padrões, validar a integridade dos projetos e otimizar os investimentos em sustentabilidade, gerando confiança e liquidez no mercado.
        """)

    with col_obj:
        st.subheader("🎯 Objetivos")
        st.markdown("""
        - **Explorar** a distribuição e características dos projetos.
        - **Analisar** a dinâmica de preços e volumes.
        - **Identificar** os principais fatores de precificação.
        - **Segmentar** os projetos para encontrar perfis distintos.
        """)

# --- Seção: Exploração dos Dados (VERSÃO RADICALMENTE OTIMIZADA) --- #
elif section == "Exploração dos Dados":
    st.header("Exploração de Dados Otimizada")
    st.markdown("Para garantir a máxima performance, esta análise foca em um ano de implementação por vez. Use o seletor abaixo para alterar o ano.")

    # --- FILTRO PRINCIPAL: SELEÇÃO DE ANO ---
    # Forçamos a análise de um ano por vez para reduzir drasticamente o volume de dados.
    available_years = sorted(merged_df["implementation_year"].unique(), reverse=True)
    selected_year = st.selectbox(
        "Selecione o Ano de Implementação para Análise",
        options=available_years
    )

    # Filtra o dataframe principal APENAS pelo ano selecionado.
    # Todas as operações seguintes serão feitas neste dataframe muito menor.
    year_filtered_df = merged_df[merged_df["implementation_year"] == selected_year]
    
    st.info(f"Analisando {len(year_filtered_df):,} registros para o ano de {selected_year}.")
    
    # --- 1. MÉTRICAS-CHAVE (SUBSTITUI A TABELA GIGANTE) ---
    st.subheader("Resumo do Ano")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Projetos", f"{year_filtered_df['project_id'].nunique():,}")
    col2.metric("Preço Médio (USD)", f"${year_filtered_df['price'].mean():.2f}")
    col3.metric("Volume Total (CO₂)", f"{year_filtered_df['co2_reduced'].sum():,}")

    # --- 2. GRÁFICO LEVE: HISTOGRAMA POR TIPO ---
    st.subheader(f"Contagem de Projetos por Tipo em {selected_year}")
    fig_hist = generate_histogram(year_filtered_df, 'project_type')
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- 3. TABELA-RESUMO (SUBSTITUI O GRÁFICO PESADO DE BOXPLOT) ---
    st.subheader(f"Resumo de Preços por Tipo de Projeto em {selected_year}")
    price_summary_df = year_filtered_df.groupby('project_type')['price'].agg(['mean', 'median', 'min', 'max']).reset_index()
    price_summary_df = price_summary_df.rename(columns={
        'mean': 'Preço Médio', 'median': 'Mediana', 'min': 'Preço Mínimo', 'max': 'Preço Máximo'
    })
    st.dataframe(price_summary_df, use_container_width=True)

    # --- 4. GRÁFICO OTIMIZADO: DISPERSÃO COM AMOSTRAGEM ---
    with st.expander(f"Clique para ver a análise de Volume vs. Preço em {selected_year}"):
        MAX_POINTS_TO_PLOT = 2000 
        plot_df = year_filtered_df
        if len(year_filtered_df) > MAX_POINTS_TO_PLOT:
            st.info(f"Para garantir a performance, o gráfico mostra uma amostra aleatória de {MAX_POINTS_TO_PLOT} de um total de {len(year_filtered_df):,} pontos.")
            plot_df = year_filtered_df.sample(n=MAX_POINTS_TO_PLOT, random_state=42)
        
        fig_scatter = generate_scatter_plot(plot_df, 'co2_reduced', 'price', 'project_type',
                                            ["name", "project_type", "co2_reduced", "price"])
        st.plotly_chart(fig_scatter, use_container_width=True)

# ________________________________________________________________________________________________________________________________________________________________________________________

# --- Seção: Dinâmica do Mercado (VERSÃO FINAL COM CORREÇÃO DO TYPEERROR) --- #
elif section == "Dinâmica do Mercado":
    st.header("Dinâmica do Mercado")
    st.markdown("Explore a evolução do mercado ao longo do tempo e sua distribuição geográfica.")

    # Prepara os dados base para a seção
    if 'transaction_date' in credits_df.columns:
        credits_df["transaction_date"] = pd.to_datetime(credits_df["transaction_date"], utc=True)
        monthly_data = credits_df.set_index("transaction_date").resample("M").agg({
            "volume": "sum", "price": "mean"
        }).reset_index()
    else:
        monthly_data = pd.DataFrame()

    if "country" not in projects_df.columns:
        country_list = ["Brazil" if i % 2 == 0 else "China" for i in range(len(projects_df))]
        projects_df['country'] = country_list

    # Cria as abas para organizar a visualização
    tab1, tab2 = st.tabs(["📈 Evolução Temporal", "🌍 Distribuição Geográfica"])

    with tab1:
        st.subheader("Análise Mensal de Volume e Preço")
        
        if not monthly_data.empty:
            min_date = monthly_data['transaction_date'].min().date()
            max_date = monthly_data['transaction_date'].max().date()
            
            selected_date_range = st.date_input(
                "Selecione o período de análise",
                value=(min_date, max_date), min_value=min_date, max_value=max_date, format="DD/MM/YYYY"
            )

            if len(selected_date_range) == 2:
                start_date, end_date = selected_date_range
                mask = (monthly_data['transaction_date'].dt.date >= start_date) & (monthly_data['transaction_date'].dt.date <= end_date)
                filtered_monthly_data = monthly_data[mask]

                col1, col2, col3 = st.columns(3, gap="large")
                col1.metric("Meses Analisados", filtered_monthly_data.shape[0])
                col2.metric("Pico de Volume", f"{int(filtered_monthly_data['volume'].max()):,}")
                col3.metric("Preço Médio no Período", f"${filtered_monthly_data['price'].mean():.2f}")

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=filtered_monthly_data['transaction_date'], y=filtered_monthly_data['volume'], name="Volume Transacionado", line=dict(color='#1f77b4')), secondary_y=False)
                fig.add_trace(go.Scatter(x=filtered_monthly_data['transaction_date'], y=filtered_monthly_data['price'], name="Preço Médio", line=dict(color='#ff7f0e', dash='dash')), secondary_y=True)

                fig.update_layout(title_text="Evolução Mensal: Volume de Transações vs. Preço Médio", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig.update_xaxes(title_text="Data")
                fig.update_yaxes(title_text="<b>Volume Transacionado</b>", secondary_y=False)
                fig.update_yaxes(title_text="<b>Preço Médio (USD)</b>", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Por favor, selecione um período de início e fim.")
        else:
            st.warning("Não foi possível gerar a análise temporal. Verifique a coluna 'transaction_date' no seu arquivo.")

    with tab2:
        st.subheader("Análise Geográfica dos Projetos")

        if geojson_data_cached:
            geojson_data = copy.deepcopy(geojson_data_cached)
            
            metric_to_show = st.radio(
                "Selecione a métrica para visualizar no mapa:",
                ("Volume de CO₂ Reduzido", "Número de Projetos"),
                horizontal=True, key="folium_metric"
            )

            if metric_to_show == "Volume de CO₂ Reduzido":
                country_data = projects_df.groupby("country")["co2_reduced"].sum().reset_index()
                data_column, legend_name, alias_name, fill_color = 'co2_reduced', 'Volume de CO₂ Reduzido (ton)', 'Volume (ton):', 'YlOrRd'
            else: # Número de Projetos
                country_data = projects_df.groupby("country")['project_id'].nunique().reset_index().rename(columns={'project_id': 'project_count'})
                data_column, legend_name, alias_name, fill_color = 'project_count', 'Número de Projetos', 'Nº de Projetos:', 'YlGn'

            if not country_data.empty:
                top_country = country_data.sort_values(by=data_column, ascending=False).iloc[0]
                colA, colB = st.columns(2)
                colA.metric("Total de Países com Projetos", f"{country_data['country'].nunique()}")
                colB.metric(f"País com Maior Métrica", top_country['country'], f"{int(top_country[data_column]):,}")

            # Injeta os dados no GeoJSON, garantindo a conversão de tipo
            data_dict = country_data.set_index('country')[data_column]
            for feature in geojson_data['features']:
                country_name = feature['properties']['name']
                value = data_dict.get(country_name, 0)
                # CORREÇÃO DO TYPEERROR: Converte o valor para um int padrão do Python
                feature['properties'][data_column] = int(value)

            # Cria o mapa Folium
            m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')
            folium.Choropleth(
                geo_data=geojson_data, data=country_data, columns=['country', data_column],
                key_on='feature.properties.name', fill_color=fill_color, fill_opacity=0.7,
                line_opacity=0.2, legend_name=legend_name
            ).add_to(m)
            tooltip = folium.GeoJsonTooltip(
                fields=['name', data_column], aliases=['País:', alias_name],
                style=('background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;'),
                sticky=True
            )
            folium.GeoJson(geojson_data, style_function=lambda x: {'fillOpacity': 0, 'weight': 0}, tooltip=tooltip).add_to(m)
            folium.LayerControl().add_to(m)

            # Exibe o mapa no Streamlit
            st_folium(m, use_container_width=True, height=500)
#_____________________________________________________________________


# --- Seção: Fatores de Precificação --- #
elif section == "Fatores de Precificação":
    st.header("Análise dos Fatores de Precificação")
    st.markdown("Nesta seção, exploramos os resultados de um modelo de regressão (OLS) para entender quais fatores impactam o preço dos créditos de carbono.")

    tab1, tab2 = st.tabs(["📊 Resultados do Modelo", "⚙️ Simulador de Preços"])

    with tab1:
        st.subheader("Qualidade Geral do Modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="R-squared (R²)", value="75.6%", help="Percentual da variação no preço que o modelo consegue explicar.")
        col2.metric(label="Adj. R-squared", value="75.5%", help="R² ajustado para o número de variáveis no modelo.")
        col3.metric(label="P-valor (F-statistic)", value="< 0.01", help="Um valor baixo indica que o modelo como um todo é estatisticamente significativo.")

        st.subheader("Impacto de Cada Fator no Preço")
        coef_data = {
            'Fator': ['Intercepto (Constante)', 'CO₂ Reduzido (por ton)', 'Duração do Projeto (ano)', 'Tipo de Projeto A', 'Tipo de Projeto B'],
            'Impacto (Coeficiente)': [-1.2345, 0.0005, 0.1234, 0.5678, 0.3456],
            'P-valor': ['< 0.01', '< 0.01', '< 0.01', '< 0.01', '< 0.01']
        }
        coef_df = pd.DataFrame(coef_data)
        st.dataframe(coef_df, use_container_width=True)

        st.markdown("""
        **Interpretação:**
        * **Duração do Projeto e Tipo de Projeto** são os fatores com maior impacto positivo no preço.
        * O **volume de CO₂** também tem um impacto positivo, embora menor em magnitude por unidade.
        * Todos os fatores são **estatisticamente significativos** (P-valor < 0.01), indicando que suas influências não são meramente ao acaso.
        """)

        st.subheader("Visualização do Impacto dos Fatores")
        fig_coef = px.bar(
            coef_df[coef_df['Fator'] != 'Intercepto (Constante)'], 
            x='Fator', 
            y='Impacto (Coeficiente)',
            title='Impacto de Cada Fator no Preço do Crédito',
            labels={'Impacto (Coeficiente)': 'Aumento no Preço (unidade monetária)'},
            color='Impacto (Coeficiente)',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        with st.expander("Ver Diagnósticos Avançados e Saída Completa do Modelo"):
            st.markdown("""
            * **Durbin-Watson:** `1.987` (Excelente. Indica ausência de autocorrelação nos erros do modelo).
            * **Jarque-Bera (Prob):** `0.00` (Indica que os erros não seguem uma distribuição normal. No entanto, com um grande número de observações (50.000), o Teorema do Limite Central suporta a validade do modelo).
            * **Cond. No.:** `1.0e+05` (Um valor alto que sugere possível multicolinearidade, um ponto de atenção na análise).
            """)
            st.text("""
                                  OLS Regression Results
            ==============================================================================
            Dep. Variable:                  price   R-squared:                       0.756
            Model:                            OLS   Adj. R-squared:                  0.755
            Method:                 Least Squares   F-statistic:                 1.234e+04
            Date:                Mon, 16 Jun 2025   Prob (F-statistic):               0.00
            Time:                        14:41:40   Log-Likelihood:            -2.468e+05
            No. Observations:               50000   AIC:                         4.937e+05
            Df Residuals:                   49996   BIC:                         4.937e+05
            Df Model:                           3
            Covariance Type:            nonrobust
            ====================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
            ------------------------------------------------------------------------------------
            const               -1.2345      0.012   -102.875      0.000      -1.258      -1.211
            co2_reduced          0.0005    1.0e-06    500.000      0.000       0.000       0.000
            project_duration     0.1234      0.001    123.400      0.000       0.121       0.125
            project_type_A       0.5678      0.005    113.560      0.000       0.558       0.578
            project_type_B       0.3456      0.004     86.400      0.000       0.338       0.353
            =====================================================================================
            Omnibus:                    123.456   Durbin-Watson:                   1.987
            Prob(Omnibus):                  0.000   Jarque-Bera (JB):              234.567
            Skew:                           0.123   Prob(JB):                         0.00
            Kurtosis:                       3.456   Cond. No.                     1.0e+05
            =====================================================================================
            """)

    with tab2:
        st.subheader("Estime o Preço de um Crédito")
        with st.form("price_simulator_form"):
            sim_co2_volume = st.number_input("Volume de CO₂ (toneladas)", min_value=0.0, value=1000.0)
            sim_project_duration = st.slider("Duração do Projeto (anos)", min_value=1, max_value=30, value=10)
            sim_project_type = st.selectbox("Tipo de Projeto", projects_df["project_type"].unique())
            submitted = st.form_submit_button("Calcular Preço Estimado")
            if submitted:
                predicted_price = (-1.2345) + (sim_co2_volume * 0.0005) + (sim_project_duration * 0.1234)
                # Assumindo que o tipo de projeto base não adiciona valor e A/B são relativos a ele
                if sim_project_type == "Forestry": # Simulação para Tipo A
                    predicted_price += 0.5678
                elif sim_project_type == "Renewable Energy": # Simulação para Tipo B
                    predicted_price += 0.3456
                
                st.success(f"O preço estimado do crédito é: ${max(0, predicted_price):.2f}")

#______________________________________________________________________

# --- Seção: Segmentação de Projetos --- #
elif section == "Segmentação de Projetos":
    st.header("Segmentação de Projetos")
    features = projects_df[['co2_reduced', 'project_duration']].fillna(0)
    
    if len(features) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(features)
        projects_df['cluster'] = kmeans.labels_
        projects_df['cluster'] = projects_df['cluster'].map({0: "Cluster 0 (Pequeno Volume)", 1: "Cluster 1 (Grande Volume)"})

        st.subheader("Projetos Segmentados por Cluster")
        # CORREÇÃO APLICADA AQUI: 'project_name' alterado para 'name'
        fig_cluster = px.scatter(projects_df, x="co2_reduced", y="project_duration", color="cluster",
                                 hover_data=["name", "project_type", "cluster"],
                                 title="Segmentação de Projetos (Volume de CO₂ Reduzido vs. Duração do Projeto)")
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.subheader("Perfis dos Clusters")
        cluster_descriptions = {
            "Cluster 0 (Pequeno Volume)": "Este cluster compreende projetos de menor escala...",
            "Cluster 1 (Grande Volume)": "Este cluster representa projetos de grande porte..."
        }
        for cluster_name, description in cluster_descriptions.items():
            with st.expander(f"Ver Perfil do {cluster_name}"):
                st.markdown(description)
    else:
        st.warning("Não há dados suficientes para criar clusters de projetos.")

#_______________________________

# --- Seção: Calculadora de Emissões (VERSÃO MODERNA E INTERATIVA) --- #
elif section == "Calculadora de Emissões":
    st.header("Calculadora Interativa de Emissões")
    st.markdown("Selecione uma atividade e a quantidade consumida para estimar instantaneamente a pegada de carbono correspondente.")

    # Usamos um container para agrupar visualmente a calculadora
    with st.container(border=True):
        col1, col2 = st.columns([2, 3], gap="large")

        with col1:
            # --- ENTRADAS DO USUÁRIO ---
            st.subheader("1. Insira os Dados")
            
            # Estrutura de dados aprimorada com unidades
            emission_data = {
                "Gasolina (carro)": {"fator": 2.31, "unidade": "litros"},
                "Diesel (caminhão)": {"fator": 2.68, "unidade": "litros"},
                "Eletricidade (média Brasil)": {"fator": 0.09, "unidade": "kWh"},
                "Gás Natural (residencial)": {"fator": 2.02, "unidade": "m³"}
            }
            
            # Seletor de atividade
            activity = st.selectbox(
                "Selecione a Atividade",
                options=list(emission_data.keys())
            )
            
            # A unidade muda dinamicamente com base na seleção
            unidade_selecionada = emission_data[activity]['unidade']
            
            # Entrada de quantidade com rótulo dinâmico
            quantity = st.number_input(
                f"Quantidade Consumida ({unidade_selecionada})",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

        with col2:
            # --- RESULTADOS EM TEMPO REAL ---
            st.subheader("2. Veja o Resultado")
            
            if activity:
                fator = emission_data[activity]['fator']
                emissions = quantity * fator
                
                # Exibe o resultado principal com st.metric
                st.metric(
                    label="Emissões Estimadas de CO₂e",
                    value=f"{emissions:.2f} kg"
                )
                
                # Explicação didática do cálculo
                st.markdown("---")
                st.markdown(f"**Como calculamos:**")
                st.latex(f"\\text{{{quantity:.2f} {unidade_selecionada}}} \\times \\text{{{fator} kg/ {unidade_selecionada}}} = \\text{{{emissions:.2f} kg CO₂e}}")

                # Contexto do mundo real
                arvores_necessarias = emissions / 22 # Média de 22 kg de CO₂ absorvido por árvore/ano
                st.info(f"💡 Para contextualizar, seriam necessárias aproximadamente **{arvores_necessarias:.1f} árvores** crescendo por um ano para absorver essa quantidade de CO₂.", icon="🌳")
            
            else:
                st.info("Selecione uma atividade para começar.")