import emoji
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def regrePolib(anual):
    st.header(anual)
    degree = st.sidebar.slider("Selecciona un grado para la Regresión Polinomial " + str(anual), 1, 20, 1, key = str(anual))
    dfstream = pd.read_csv("base.csv")
    X1 = dfstream.drop([ "x", "y", "cruce_osm", "Id", "dia_sem", "clave_mun", "mun", "ibaen_atro", "num_mes", "rango_hora", "cruce_setrans", "tipo_accidente", "rango_edad", "sexo","dileso", "tipo_usuario", ], axis = 1)
    X1["herido"].fillna("Ileso", inplace=True)
    df1 = X1[X1.anio==anual]
    df2 = df1[df1.anio==anual]
    df2 = df2[df2["herido"] != "Ileso"]
    y2 = df2["herido"]
    df2 = df2.drop(["anio", "dia"], axis = 1)
    df2["herido"] = LabelEncoder().fit_transform(df2["herido"])
    df2["mes"] = LabelEncoder().fit_transform(df2["mes"])
    y2 = LabelEncoder().fit_transform(y2)
    conteo_por_mes = df2[df2['herido'] == 1].groupby('mes').count()
    conteo_por_mes = conteo_por_mes.reset_index()
    ymes = conteo_por_mes["herido"]
    conteo = conteo_por_mes
    conteo_por_mes = conteo_por_mes.drop(["herido"], axis = 1)
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = polynomial_features.fit_transform(conteo_por_mes)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, ymes)
    Y_poly_pred = model_poly.predict(X_poly)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("")
    plt.scatter(conteo.mes, conteo.herido)
    plt.plot(conteo.mes, conteo.herido, color='magenta')
    plt.plot(conteo_por_mes, Y_poly_pred, color='red')
    plt.xlabel("Mes")
    plt.ylabel("Cantidad")
    plt.grid()
    st.pyplot() 

def regrePoli(anual,acu):
    
    degree = st.sidebar.slider("Selecciona un grado para la Regresión Polinomial " + str(anual), 1, 20, 1, key = str(anual))
    dfstream = pd.read_csv("base.csv")
    X1 = dfstream.drop([ "x", "y", "cruce_osm", "Id", "dia_sem", "clave_mun", "mun", "ibaen_atro", "num_mes", "rango_hora", "cruce_setrans", "tipo_accidente", "rango_edad", "sexo","dileso", "tipo_usuario", ], axis = 1)
    X1["herido"].fillna("Ileso", inplace=True)
    df1 = X1[X1.anio==anual]
    df2 = df1[df1.anio==anual]
    df2 = df2[df2["herido"] != "Ileso"]
    y2 = df2["herido"]
    df2 = df2.drop(["anio", "dia"], axis = 1)
    df2["herido"] = LabelEncoder().fit_transform(df2["herido"])
    df2["mes"] = LabelEncoder().fit_transform(df2["mes"])
    y2 = LabelEncoder().fit_transform(y2)
    conteo_por_mes = df2[df2['herido'] == 1].groupby('mes').count()
    conteo_por_mes = conteo_por_mes.reset_index()
    ymes = conteo_por_mes["herido"]
    
    

    conteo = conteo_por_mes
    conteo_por_mes = conteo_por_mes.drop(["herido"], axis = 1)

    acu = acu*11 
    index = 0
    lista = []
    for i in range(1,100):
        lista.append(i)

            
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = polynomial_features.fit_transform(conteo_por_mes)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, ymes)
    Y_poly_pred = model_poly.predict(X_poly)

    #plt.scatter(conteo.mes, conteo.herido)
    #plt.plot(conteo.mes, conteo.herido, color='magenta')
    #plt.plot(conteo_por_mes, Y_poly_pred, color='red')
    
    
    return [lista, conteo.herido, Y_poly_pred, acu]

def pruebaTotal():
    st.header(emoji.emojize("Regresion polinomial 📊"))
    st.subheader("2015 al 2021")
    st.write("A continuación se muestran 7 distintas gráficas que corresponden a regresiones polinomiales del 2015 al 2021, todas en un mismo gráfico.")
    st.write("Puede ajustar el grado de cada regresión en el respectivo slider del menú sidebar, por defualt se muestra una regresión de grado uno para cada año.")      
    
    arp1 =regrePoli(2015,0)
    arp2 =regrePoli(2016,1)
    arp3 =regrePoli(2017,2)
    arp4 =regrePoli(2018,3)
    arp5 =regrePoli(2019,4)
    arp6 =regrePoli(2020,5)
    arp7 =regrePoli(2021,6)
    #print("HERE")
    #print(arp1)
    #print(arp2)

    #print(len(arp1[0][0:12]))
    #print(len(arp1[1]))
    #print(len(arp1[3]))

    plt.scatter(arp1[0][0:12], arp1[1])
    plt.plot(arp1[0][0:12], arp1[1], color='magenta')
    plt.plot(arp1[0][0:12], arp1[2], color='red')
    
    plt.scatter(arp2[0][13:25], arp2[1])
    plt.plot(arp2[0][13:25], arp2[1], color='magenta')
    plt.plot(arp2[0][13:25], arp2[2], color='red')

    plt.scatter(arp3[0][26:38], arp3[1])
    plt.plot(arp3[0][26:38], arp3[1], color='magenta')
    plt.plot(arp3[0][26:38], arp3[2], color='red')

    plt.scatter(arp4[0][39:51], arp4[1])
    plt.plot(arp4[0][39:51], arp4[1], color='magenta')
    plt.plot(arp4[0][39:51], arp4[2], color='red')

    plt.scatter(arp5[0][52:64], arp5[1])
    plt.plot(arp5[0][52:64], arp5[1], color='magenta')
    plt.plot(arp5[0][52:64], arp5[2], color='red')

    plt.scatter(arp6[0][65:77], arp6[1])
    plt.plot(arp6[0][65:77], arp6[1], color='magenta')
    plt.plot(arp6[0][65:77], arp6[2], color='red')

    plt.scatter(arp7[0][78:90], arp7[1])
    plt.plot(arp7[0][78:90], arp7[1], color='magenta')
    plt.plot(arp7[0][78:90], arp7[2], color='red')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def pruebaTotalselect():
    st.header(emoji.emojize("Regresion polinomial 📊"))
    st.subheader(emoji.emojize("Interactiva 👆🏼"))
    st.write("A continuación se muestra de manera continua los graficos con regresiones polinomiales del 2015 al 2021, si desea unicamente visualizar determinados años, por favor ajuste el filtro de abajo")
    st.write("De la misma manera, puede ajustar el grado de las regresiones polinomiales seleccionadas en el respectivo slider del menú sidebar, por defualt se muestra una regresión de grado uno para cada año.")      
    dfstream = pd.read_csv("base.csv")
    years = dfstream["anio"].unique().tolist()
    selected_years = st.multiselect('Selecciona los años', years, default=years)

    num_plots = len(selected_years)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 6, 6))

    for i, year in enumerate(selected_years):
        arp = regrePoli(year, i)
        ax = axs[i] if num_plots > 1 else axs  # Utilizar el mismo subplot si solo hay un año seleccionado

        ax.scatter(arp[0][0:12], arp[1])
        ax.plot(arp[0][0:12], arp[1], color='magenta')
        ax.plot(arp[0][0:12], arp[2], color='red')
        ax.set_title(str(arp[3]))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)    

def inicio():
    st.title(emoji.emojize("Reto Analítica de Datos :chart_increasing:"))
    st.title("Bienvenido a la página de inicio")
    st.header("ITD Sexto Semestre")
    st.write(emoji.emojize("### Equipo :kiss_mark: : "))
    col17, col18 = st.columns(2)
    col17.caption(emoji.emojize('# Enrique Lemus 👻'))
    col17.caption('A01639698')
    col17.caption(emoji.emojize('# Javier Morales ⭐'))
    col17.caption('A01632535')
    col18.caption(emoji.emojize('# Jessica Vazquez 🧸'))
    col18.caption('A01639945')
    col18.caption(emoji.emojize('# Sebastian Escobedo 💩'))
    col18.caption('A01351597')

def datos():
    st.header(emoji.emojize("Datos Trafico 🚗 "))
    st.write("A continuación, se presentan las primeras filas del dataframe a manera de poder conocer las columnas que lo componen y los distintos datos que contienen")
    # data will be in cache
    @st.cache
    def get_data():
        return pd.read_csv('base.csv')
    df = pd.read_csv('base.csv') #get_data()
    st.dataframe(df.head())
    #Five Number Summary
    st.header(emoji.emojize("Five Number Summary 🖐🏼"))
    st.write("Debajo se presenta un *Five Number Analysis* de los datos numéricos del Dataframe.")
    st.write("Se compone de un recuento de todos los valores que hay en una determinada columna, el promedio de estos, la desviación estandar, el valor mínimo y el máximo así como los Cuartiles del 25%, 50% y 75%")
    five = df.describe()

    st.write(five)
    st.write("A continuación se presentan los Five Number Analysis de las columnas con valores numéricos que podrían ser más valiosas por ahora")

    col1, col2, col3 = st.columns(3)
    anio5 = df.anio.describe()
    col1.subheader(emoji.emojize("Año 📆"))
    col1.write(anio5)

    dia5 = df.dia.describe()
    col2.subheader(emoji.emojize("Día del Mes 📆"))
    col2.write(dia5)

    num_mes5 = df.num_mes.describe()
    col3.subheader(emoji.emojize("Mes del Año 📆"))
    col3.write(num_mes5)

    ## Box plot
    st.write("De la misma manera, se presentan los respectivos Boxplot (Cajas y bigotes) de los valores anteriormente mencionados")
    col4, col5, col6 = st.columns([1,1,1])

    col4.write("### Boxplot de año")
    fig = px.box(df, y="anio",height=755, width=200)
    col4.plotly_chart(fig)

    col5.write("### Boxplot de Dia del mes")
    fig = px.box(df, y="dia", height=700, width=200)
    col5.plotly_chart(fig)

    col6.write("### Boxplot de Mes del año")
    fig = px.box(df, y="num_mes", height=750, width=200)
    col6.plotly_chart(fig)

    # HistPlot DistPlot
    #(Skewness y Kurtosis)
    st.write(emoji.emojize("## Distribuciones 📝"))

    st.subheader(emoji.emojize("Skewness y Kurtosis de los Datos 🤔"))
    st.write("Aquí se presentan 2 tablas distintas, una para el valor de Skewness de cada columna numérica del dataframe y otra para los valores de Kurtosis")
    st.write("*Skewness* indica la tendencia u orientación de la distribución de la curva. Un valor de Skewness de o cercano a 0 corresponde a una distribución normal. Por otra parte, un valor menor a 0 resulta en un skewness positivo, una distribución cargada a la izquierda. Un valor de Skewness mayor a 1, indica un Skewness negativo, es decir una distribución cargada a la derecha.")
    st.write("*Kurtosis* representa qué tan 'puntiaguda' es la distribución. Un valor alto de Kurtosis indica una mayor concentración de valores en el promedio (punta) de la distribución. Un valor de 0 o muy cercano, supone una distribución normal, un calor mayor a 0 es una distribución muy puntiaguda, mientras que un valor menor a 0 es una distribución muy plana/alargada")

    skewnessdf = df.skew()
    kurtosisdf = df.kurt() 
    col20, col21, col22, col23 = st.columns(4)

    col21.write(skewnessdf.to_frame().rename(columns={0: "Skewness"}), unsafe_allow_html=True)
    col22.write(kurtosisdf.to_frame().rename(columns={0: "Kurtosis"}), unsafe_allow_html=True)

    #Distribución de años
    st.sidebar.header("Filtros para Graficos de Distribución")
    st.write("### ")
    st.write(emoji.emojize("### Distribución de los Años 📊"))
    st.write("##### A continuación se muestra una curva de densidad y un histograma de los accidentes del 2015 al 2021")
    st.write(" Si deseas cambiar los años mostrados en el histograma, por favor, ajusta los parámetros en el slider de *'Años'* en el menu sidebar")
    skewness_anio = df.anio.skew()
    kurtosis_anio = df.anio.kurt()
    col7, col8, col9= st.columns(3)

    #Histograma
    values = st.sidebar.slider("Años", int(df.anio.min()), int(df.anio.max()), (int(df.anio.min()), int(df.anio.max())), step=1)
    filtered_df = df[(df['anio'] >= values[0]) & (df['anio'] <= values[1])]
    hist = px.histogram(filtered_df, x="anio", nbins=7)
    hist.update_traces(marker_line_width=2, marker_line_color='black')
    hist.update_xaxes(title="Años")
    hist.update_yaxes(title="Número de Accidentes")
    fig1 = go.Figure(data=hist)
    fig1.update_layout(width=500, height=525)
    col9.plotly_chart(fig1)
    #Kde
    fig, ax = plt.subplots(figsize = (5,5))
    sns.kdeplot(data=filtered_df, x="anio", ax=ax)
    ax.set_xlabel("Año")
    ax.set_ylabel("Densidad")
    ax.grid(True)
    col7.plotly_chart(fig)

    #Skewness y Kurtosis de Años
    st.write("El valor de skewness para Año es: %.2f" % skewness_anio)
    st.write("El valor de kurtosis para Año es: %.2f" % kurtosis_anio)

    #Distribución de meses
    st.write("### ")
    st.write(emoji.emojize("### Distribución de los Meses 📊"))
    st.write("##### A continuación se muestra una curva de densidad y un histograma de los accidentes de los meses del 2015 al los del 2021")
    st.write(" Si deseas cambiar los años mostrados en el histograma, por favor, ajusta los parámetros en el slider de *'Meses del año'* en el menu sidebar")

    skewness_mes = df.num_mes.skew()
    kurtosis_mes = df.num_mes.kurt()
    col10, col11, col12= st.columns(3)

    #Histograma
    values2 = st.sidebar.slider("Meses del año", int(df.num_mes.min()), int(df.num_mes.max()), (int(df.num_mes.min()), int(df.num_mes.max())), step=1)
    filtered_df2 = df[(df['num_mes'] >= values2[0]) & (df['num_mes'] <= values2[1])]
    hist2 = px.histogram(filtered_df2, x="num_mes", nbins=12)
    hist2.update_traces(marker_line_width=2, marker_line_color='black')
    hist2.update_xaxes(title="Meses")
    hist2.update_yaxes(title="Número de Accidentes")
    fig3 = go.Figure(data=hist2)
    fig3.update_layout(width=500, height=525)
    col12.plotly_chart(fig3)

    #Kde
    fig2, ax2 = plt.subplots(figsize = (5,5))
    sns.kdeplot(data=filtered_df2, x="num_mes", ax=ax2)
    ax2.set_xlabel("Mes del año")
    ax2.set_ylabel("Densidad")
    ax2.grid(True)
    col10.plotly_chart(fig2)

    #Skewness y Kurtosis de Meses2
    st.write("El valor de skewness para los meses del año es: %.2f" % skewness_mes)
    st.write("El valor de kurtosis para los meses del año es: %.2f" % kurtosis_mes)

    #Distribución de Días
    st.write("### ")
    st.write(emoji.emojize("### Distribución de los Días 📊"))
    st.write("##### A continuación se muestra una curva de densidad y un histograma de los accidentes por día del 2015 al 2021")
    st.write(" Si deseas cambiar los años mostrados en el histograma, por favor, ajusta los parámetros en el slider de *'Días del mes'* en el menú sidebar")

    col13, col14, col15= st.columns(3)

    skewness_dia = df.dia.skew()
    kurtosis_dia = df.dia.kurt()

    #Histograma
    values3 = st.sidebar.slider("Días del mes", int(df.dia.min()), int(df.dia.max()), (int(df.dia.min()), int(df.dia.max())), step=1)
    filtered_df3 = df[(df['dia'] >= values3[0]) & (df['dia'] <= values3[1])]
    hist3 = px.histogram(filtered_df3, x="dia", nbins=31)
    hist3.update_traces(marker_line_width=2, marker_line_color='black')
    hist3.update_xaxes(title="Dias")
    hist3.update_yaxes(title="Número de Accidentes")
    fig5 = go.Figure(data=hist3)
    fig5.update_layout(width=500, height=525)
    col15.plotly_chart(fig5)

    #Kde
    fig4, ax3 = plt.subplots(figsize = (5,5))
    sns.kdeplot(data=filtered_df3, x="dia", ax=ax3)
    ax3.set_xlabel("dia")
    ax3.set_ylabel("Densidad")
    ax3.grid(True)
    col13.plotly_chart(fig4)


    st.write("El valor de skewness para los días es: %.2f" % skewness_dia)
    st.write("El valor de kurtosis para los días es: %.2f" % kurtosis_dia)

    #Tipo de Accidentes
    st.header(emoji.emojize("Tipos de Accidentes 🚑🚓"))
    anios_unicos = df['anio'].unique()
    selected_anios = st.multiselect('Selecciona los años', anios_unicos)
    col27,col28 = st.columns(2)
    tipos_excluidos = ["Otro", "Otro tipo de incidente", "Sin Datos","Contra Semoviente"]
    if len(selected_anios) == 0:
        df_filtrado = df
    else:
        df_filtrado = df[df['anio'].isin(selected_anios)]
    df_filtrado = df_filtrado[~df_filtrado["tipo_accidente"].isin(tipos_excluidos)]
    accidente_unico = df_filtrado.groupby("tipo_accidente").Id.count()
    accidente_unico = accidente_unico.rename("Cantidad de accidentes")
    col27.write(accidente_unico, unsafe_allow_html=True)
    col28.write("Aqui se muestra una tabla con todos los tipos de accidentes registrados y a un lado la suma total de accidentes de ese tipo en específico. Actualmente se muestra la suma total de tipo de accidente del 2015 al 2021, si desea ver uno o más años en específico ajuste los filtros arriba.")


    #Cantidad por año de Accidentes por medio de transporte
    st.header(emoji.emojize("Cantidad de accidentes por medio de transporte por año 🚌 🛵 🚲"))
    tipos_excluidos2 = ["Semoviente", "Sin Datos", "Triciclo"]
    transporte_unicos2 = df[~df['ibaen_atro'].isin(tipos_excluidos2)]['ibaen_atro'].unique()
    selected_transporte2 = st.multiselect('Selecciona el tipo de transporte', transporte_unicos2)
    col29,col30 = st.columns(2)
    if len(selected_transporte2) == 0:
        df_filtrado = df
    else:
        df_filtrado = df[df['ibaen_atro'].isin(selected_transporte2)]
    ibaen_unico = df_filtrado.groupby('anio')['Id'].count()
    ibaen_unico = ibaen_unico.rename("Cantidad de accidentes en el año")
    col29.write(ibaen_unico, unsafe_allow_html=True)
    col30.write("Aqui se muestra una tabla del 2015 al 2021 y a un lado la suma total de accidentes de tipo de medio de transporte en específico. Actualmente se muestra la suma total de accidentes sin importar en qué se iba del 2015 al 2021, si desea ver uno o más transportes en específico ajuste los filtros arriba.")

def prediccion():
    st.header(emoji.emojize("Predicción de si un usuario resulta ileso tras un accidente 💥"))
    st.sidebar.header('Elige los valores para predecir')
    st.write()

    def user_input():
        anio = st.sidebar.slider('Seleccione un Año', 2015, 2018, 2021)
        mes = st.sidebar.slider('Seleccione un mes', 1, 5, 12)
        dia = st.sidebar.slider('Seleccione un Día', 1, 15, 31)
        dia_sem = st.sidebar.selectbox('Seleccione un día de la semana', [1, 2, 3, 4, 5, 6, 7])
        mun = st.sidebar.slider('Seleccione un Municipio', 0, 3, 6)
        herido = st.sidebar.selectbox('Seleccione la condicion del usuario', [1, 2])
        ibaen_atro = st.sidebar.slider('Seleccione un medio de transporte', 1, 10, 15)
        rango_hora = st.sidebar.slider('Seleccione un rango de horas', 1, 12, 24)
        tipo_accidente = st.sidebar.slider('Seleccione un tipo de accidente', 1, 7, 10)
        rango_edad = st.sidebar.slider('Seleccionen un rango de edad', 1, 3, 6)
        tipo_usuario = st.sidebar.selectbox('Seleccione un tipo usuario', [0, 1, 2])
        sexo = st.sidebar.selectbox('Seleccione el Sexo del usuario', [0, 1, 2])
        
        data = {'anio': anio,
                'mes': mes,
                'dia': dia, 
                'dia_sem': dia_sem, 
                'mun': mun,
                'herido': herido,
                'ibaen_atro': ibaen_atro,
                'rango_hora': rango_hora, 
                'tipo_accidente': tipo_accidente,
                'rango_edad': rango_edad,
                'tipo_usuario': tipo_usuario,
                'sexo': sexo}
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input()
    st.write(" ")
    st.write("Por favor, con los sliders del sidebar, ajuste los parametros para cada variable con el objetivo de poder predecir si un usuario con esas caracteristicas sale ileso o no de un accidente.")
    st.write("En la parte de abajo de esta página se muestra un breve diccionario explicando qué es cada variable y cada valor a elegir de esta misma")
    st.write(" ")
    st.subheader(emoji.emojize('Parámetros para predicción 🧮'))
    st.write(df)

    # Load Trained Model
    filename = 'model.sav'
    model = pickle.load(open(filename, 'rb'))

    prediction = model.predict(df)
    col4, col5, col6 = st.columns(3)
    col5.subheader(emoji.emojize('🔮 Predicción 🔮'))    
    target_names = ["Ileso", "Lesionado"]
    col5.write('Resultado predicción: ' + target_names[prediction[0]])
    st.header(" ")
    st.subheader("*Diccionario de datos")
    col1, col2, col3 = st.columns(3)
    col1.write("### anio")
    col1.write("Año en el que pasa el accidente (2015 al 2021)")
    col2.write("### mes")
    col2.write("Mes en el que pasa el accidente (1 (Ene) al 12(Dic))")
    col3.write("### dia")
    col3.write("Día en el que pasa el accidente (1 al 31)")
    col1.write("### dia_sem")
    col1.write("Día de la semana en el que pasa el accidente (1 (Lun) al 7 (Dom))")
    col2.write("### mun")
    col2.write("Municipio en el que pasa el accidente (1 al 6)")
    col2.write("1 = El Salto")
    col2.write("2 = Guadalajara")
    col2.write("3 = San Pedro Tlaquepaque")
    col2.write("4 = Tlajomulco de Zuñiga")
    col2.write("5 = Tonalá")
    col2.write("6 = Zapopan")
    col3.write("### herido")
    col3.write("Condición de la persona en el accidente")
    col3.write("1 = Lesionado")
    col3.write("2 = Fallecido")
    col1.write("### ibaen_atro")
    col1.write("Medio de transporte en el que iba el usuario(1-15)")
    col1.write("1 = Ambulacia")
    col1.write("2 = Automovil")
    col1.write("3 = Bicicleta")
    col1.write("4 = Camión de Carga")
    col1.write("5 = Camioneta de Carga")
    col1.write("6 = Camioneta de Pasajeros")
    col1.write("7 = Ferrocarril")
    col1.write("8 = Foraneo")
    col1.write("9 = Macrobus")
    col1.write("10 = Motocicleta")
    col1.write("11 = Taxi")
    col1.write("12 = Traler")
    col1.write("13 = Transporte de Personal")
    col1.write("14 = Transporte Público")
    col1.write("15 = Tren Ligero")
    col2.write("### rango_hora")
    col2.write("Hora en la que pasó el accidente (1-24)")
    col2.write("1 = 0:00 a 0:59")
    col2.write("2 = 1:00 a 1:59")
    col2.write("3 = 10:00 a 10:59")
    col2.write("4 = 11:00 a 11:59")
    col2.write("5 = 12:00 a 12:59")
    col2.write("6 = 13:00 a 13:59")
    col2.write("7 = 14:00 a 14:59")
    col2.write("8 = 15:00 a 15:59")
    col2.write("9 = 16:00 a 16:59")
    col2.write("10 = 17:00 a 17:59")
    col2.write("11 = 18:00 a 18:59")
    col2.write("12 = 19:00 a 19:59")
    col2.write("13 = 2:00 a 2:59")
    col2.write("14 = 20:00 a 20:59")
    col2.write("15 = 21:00 a 21:59")
    col2.write("16 = 22:00 a 22:59")
    col2.write("17 = 23:00 a 23:59")
    col2.write("18 = 3:00 a 3:59")
    col2.write("19 = 4:00 a 4:59")
    col2.write("20 = 5:00 a 5:59")
    col2.write("21 = 6:00 a 6:59")
    col2.write("22 = 7:00 a 7:59")
    col2.write("23 = 8:00 a 8:59")
    col2.write("24 = 9:00 a 9:59")
    col3.write("### tipo_accidente")
    col3.write("Tipo de accidente sucedido (1-10)")
    col3.write("1 = Atropellamiento")
    col3.write("2 = Caida")
    col3.write("3 = Caida Al Exterior")
    col3.write("4 = Caida En El Interior")
    col3.write("5 = Colision")
    col3.write("6 = Contra Objeto Fijo")
    col3.write("7 = Contra Vehiculo Estacionado")
    col3.write("8 = Ferroviario")
    col3.write("9 = Otro Tipo de Accidente")
    col3.write("10 = Volcadura")
    col1.write("### rango_edad")
    col1.write("Rango de edad el usuario en el accidente (1-6)")
    col1.write("1 = 60")
    col1.write("2 = 0 - 17")
    col1.write("3 = 18 - 27")
    col1.write("4 = 28 - 37")
    col1.write("5 = 38 - 47")
    col1.write("6 = 48 - 59")
    col3.write("### tipo_usuario")
    col3.write("Tipo de usuario del accidente (0 - 2)")
    col3.write("0 = Conductor")
    col3.write("1 = Pasajero")
    col3.write("2 = Peaton")
    col3.write("### sexo")
    col3.write("Sexo del usuario en el accidente (0 - 2)")
    col3.write("0 = Hombre")
    col3.write("1 = Mujer")
    col3.write("2 = No Especificado")

def analisis():
   # data will be in cache
    @st.cache
    def get_data():
        return pd.read_csv('base.csv')
    df = pd.read_csv('base.csv') #get_data()

    dias_semana = {
        'lunes': 1,
        'martes': 2,
        'miércoles': 3,
        'jueves': 4,
        'viernes': 5,
        'sábado': 6,
        'domingo': 7
    }
    df['dia_sem_numerico'] = df['dia_sem'].map(dias_semana)
    frecuencia_dias = df['dia_sem_numerico'].value_counts()
    dia_mas_repetido = frecuencia_dias.idxmax()
    dia_menos_repetido = frecuencia_dias.idxmin()
    st.title(emoji.emojize("Análisis de días de la semana 📅"))
    col1, col2, col3 = st.columns(3)
    col1.write("### Dias de la semana")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("")
    col2.write("Lunes: 1")
    col2.write("Martes: 2")
    col2.write("Miércoles: 3")
    col2.write("Jueves: 4")
    col2.write("Viernes: 5")
    col2.write("Sábado: 6")
    col2.write("Domingo: 7")
    col2.write("")
    col2.write("")
    st.subheader(emoji.emojize("Días que más accidentes ocurren 💥"))
    st.write("El día de la semana en el que hubo más accidentes es:", dia_mas_repetido)
    st.write("El día de la semana en el que hubo menos accidentes es:", dia_menos_repetido)
    #Visualización del Mapa de accidentes
    st.header(emoji.emojize("Mapa de accidentes 🗺️"))
    st.sidebar.header("Filtros para Mapa de Accidentes")  
    df["x"] = df["x"].fillna(df["x"].mode()[0])
    df["y"] = df["y"].fillna(df["y"].mode()[0])
    df = df.rename(columns={'y': 'LAT', 'x': 'LON'})
    
    anios = df["anio"].unique()
    meses = df["mes"].unique()
    
    horas = df["rango_hora"].unique()
    horas = [hora for hora in horas if hora != "No especificado"]
    orden_horas = ["0:00 a 0:59", "1:00 a 1:59", "2:00 a 2:59", "3:00 a 3:59", "4:00 a 4:59", "5:00 a 5:59", "6:00 a 6:59", "7:00 a 7:59", "8:00 a 8:59", "9:00 a 9:59", "10:00 a 10:59", "11:00 a 11:59", "12:00 a 12:59", "13:00 a 13:59", "14:00 a 14:59", "15:00 a 15:59", "16:00 a 16:59", "17:00 a 17:59", "18:00 a 18:59", "19:00 a 19:59", "20:00 a 20:59", "21:00 a 21:59", "22:00 a 22:59", "23:00 a 23:59"]
    horas = sorted(horas, key=lambda x: orden_horas.index(x))
    
    dia_sem = df["dia_sem"].unique()
    orden_dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    dia_sem = sorted(dia_sem,key=lambda x: orden_dias.index(x))

    st.caption("### En el mapa podemos visualizar todos los accidentes registrados en parte de la ZMG")
    st.caption("### Si deseas ver los accidentes de uno o más años o meses en particular, por favor, ajusta los filtros en el menú sidebar")
    st.caption("Si no se selecciona un año(s) o mes(es) específico, lo que se presenta en el mapa son todos los accidentes registrados del 2015 al 2021")

    seleccion_anios = st.sidebar.multiselect("Selecciona uno o varios años", anios)
    seleccion_meses = st.sidebar.multiselect("Selecciona uno o varios meses", meses)
    seleccion_dia_sem = st.sidebar.multiselect("Seleccione uno o varios dias de la semana", dia_sem)
    seleccion_hora = st.sidebar.multiselect("Seleccione uno o varios rangos de hora", horas)
    filtered_df = df.copy()
    

    if seleccion_anios:
        filtered_df = filtered_df[filtered_df["anio"].isin(seleccion_anios)]
    if seleccion_meses:
        filtered_df = filtered_df[filtered_df["mes"].isin(seleccion_meses)]
    if seleccion_dia_sem:
        filtered_df = filtered_df[filtered_df["dia_sem"].isin(seleccion_dia_sem)]
    if seleccion_hora:
        filtered_df = filtered_df[filtered_df["rango_hora"].isin(seleccion_hora)]

    st.write(" ")

    if not filtered_df.empty:
        st.map(filtered_df[["LAT", "LON"]].dropna(how="any"))
    else:
        st.write("###  No sucedio nungún accidente durante el periodo de tiempo seleccionado en los filtros")

    st.write(" ")
    st.write(" ")

    df['dia_sem_numerico'] = df['dia_sem'].map(dias_semana)
    frecuencia_dias = df['dia_sem_numerico'].value_counts()
    dia_mas_repetido = frecuencia_dias.idxmax()


    st.subheader(emoji.emojize("Análisis de Outliers 🔍"))
    
    
    #Outliers de num_mes
    IQR = df.num_mes.quantile(0.75) - df.num_mes.quantile(0.25)
    lower = df.num_mes.quantile(0.25) - 1.5 * IQR
    upper = df.num_mes.quantile(0.75) + 1.5 * IQR
    outliers = []

    for x in df.num_mes:
        if x < lower or x > upper:
            outliers.append(x)

    if len(outliers) > 0:
        st.write("Valores atípicos (outliers) en la columna 'num_mes':")
        st.table(outliers)
    else:
        st.write("No se encontraron valores atípicos en la columna 'num_mes'.")
        

    #Outliers de clave_mun
    IQR = df.clave_mun.quantile(0.75) - df.clave_mun.quantile(0.25)
    lower = df.clave_mun.quantile(0.25) - 1.5 * IQR
    upper = df.clave_mun.quantile(0.75) + 1.5 * IQR
    outliers = []

    for x in df.clave_mun:
        if x < lower or x > upper:
            outliers.append(x)

    if len(outliers) > 0:
        st.write("Valores atípicos (outliers) en la columna 'clave_mun':")
        st.table(outliers)
    else:
        st.write("No se encontraron valores atípicos en la columna 'clave_mun'.")

    # Definir los rangos de hora
    rango_inicio = 8
    rango_fin = 17

    # Crear el diccionario de mapeo para los rangos de hora
    ran_hora = {}
    for hora in range(rango_inicio, rango_fin+1):
        rango_texto = f"{hora:02d}:00 a {hora:02d}:59"
        ran_hora[rango_texto] = hora

    # Crear una nueva columna 'rango_hora_numerico' con los valores convertidos
    df['rango_hora_numerico'] = df['rango_hora'].map(ran_hora)

    IQR = df.rango_hora_numerico.quantile(0.75) - df.rango_hora_numerico.quantile(0.25)
    lower = df.rango_hora_numerico.quantile(0.25) - 1.5 * IQR
    upper = df.rango_hora_numerico.quantile(0.75) + 1.5 * IQR

    outliers = []

    for x in df.rango_hora_numerico:
        if x < lower or x > upper:
            outliers.append(x)


    if len(outliers) > 0:
        st.write("Valores atípicos (outliers) en la columna 'rango_hora_numerico':")
        st.table(outliers)
    else:
        st.write("No se encontraron valores atípicos en la columna 'rango_hora_numerico'.")

    # Definir un diccionario de mapeo para los días de la semana
    dias_semana = {
        'lunes': 1,
        'martes': 2,
        'miércoles': 3,
        'jueves': 4,
        'viernes': 5,
        'sábado': 6,
        'domingo': 7
    }
    # Crear una nueva columna 'dia_sem_numerico' con los valores convertidos
    df['dia_sem_numerico'] = df['dia_sem'].map(dias_semana)

    IQR = df.dia_sem_numerico.quantile(0.75) - df.dia_sem_numerico.quantile(0.25)
    lower = df.dia_sem_numerico.quantile(0.25) - 1.5 * IQR
    upper = df.dia_sem_numerico.quantile(0.75) + 1.5 * IQR

    outliers = []

    for x in df.dia_sem_numerico:
        if x < lower or x > upper:
            outliers.append(x)


    if len(outliers) > 0: 
        st.write("Valores atípicos (outliers) en la columna 'dia_sem_numerico':")
        st.table(outliers)
        
    else:
        st.write("No se encontraron valores atípicos en la columna 'dia_sem_numerico'.")

    st.subheader(emoji.emojize("Análisis de Correlaciones 🔗"))

    df2 = df.apply(LabelEncoder().fit_transform)
    df_filled = df2.fillna(df2.mean())
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_filled.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

def clasificaciones():
    st.title("Análisis y Clasificaciones")
    st.header(emoji.emojize("☠️ Fallecidos y Lesionados 🚑"))
    @st.cache
    def get_data():
        return pd.read_csv('base.csv')
    df = pd.read_csv('base.csv')
    
    st.write("A continuación se muestra una tabla ordenada cronológicamente mes a mes de los Lesionados y Fallecidos. Puede elegir el o los años de los que quiera ver los Fallecidos y Lesionados en el menú del sidebar, si no se selecciona ningún año, lo que se muestra es la suma total del 2015-2021. ")

    col1, col2, col3, col4 = st.columns(4)
    st.sidebar.write("### Filtros conteo mensual de fallecidos y lesionados")  
    #Numero total de Lesionados por mes en orden cronológico
    anios_unicos = df['anio'].unique()
    selected_anios = st.sidebar.multiselect(emoji.emojize('Selecciona los años de Lesionados 🩸'), anios_unicos)
    if len(selected_anios) == 0:
        df_filtrado = df
    else:
        df_filtrado = df[df['anio'].isin(selected_anios)]
    orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    lesionados = df_filtrado[df_filtrado['herido'] == 'Lesionado'].groupby('mes')['herido'].count()
    lesionados = lesionados.reindex(orden_meses)
    lesionados = lesionados.rename("Lesionados")
    col2.write(lesionados, unsafe_allow_html=True)

    #Numero total de Fallecidos por mes en orden cronológico
    selected_anios = st.sidebar.multiselect(emoji.emojize('Selecciona los años de Fallecidos ☠️'), anios_unicos)
    if len(selected_anios) == 0:
        df_filtrado = df
    else:
        df_filtrado = df[df['anio'].isin(selected_anios)]
    
    orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    fallecidos = df_filtrado[df_filtrado['herido'] == 'Fallecido'].groupby('mes')['herido'].count()
    fallecidos = fallecidos.reindex(orden_meses)
    fallecidos = fallecidos.rename("Fallecidos")
    col3.write(fallecidos, unsafe_allow_html=True)
    
    #Hombres y Mujeres Multiaños
    st.header(emoji.emojize("Hombres y Mujeres involucrados en accidentes 🚹🚺"))
    st.write("El gráfico a continuación muestra una gráfica de barras en la que se muestra por género, la cantidad de accidentes que hubo mes con mes en un determinado año. Selecciona por favor uno o varios años entre el 2015 y el 2021 para mostrar la suma en la gráfica. Si ningún año es seleccioando, la gráfica muestra la suma de todos los registros del 2015 al 2021.")
    
    anios_deseados = st.multiselect("Ingrese los años deseados separados por comas:",anios_unicos)
    if anios_deseados:
        df_filtered = df[(df['anio'].isin(anios_deseados)) & (df['sexo'] != 'No especificado')]
        title = f"Cantidad de hombres y mujeres para los años: {', '.join(str(a) for a in anios_deseados)}"
    else:
        df_filtered = df[df['sexo'] != 'No especificado']
        title = "Suma total de hombres y mujeres mes a mes"

    
    fig, ax = plt.subplots(figsize=(15, 9))
    my_palette = {"Hombre": "#3FDE1F", "Mujer": "#FF1C86"}
    sns.histplot(
        data=df_filtered,
        x="mes",
        hue="sexo",
        multiple="stack",
        palette=my_palette,
        ax=ax
    )
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cantidad")
    ax.set_title(title)
    st.pyplot(fig)

def regPol():
    #Regresión Polinomial
    st.header(emoji.emojize("Regresión Polinomial 🚦 📉"))
    st.header(emoji.emojize("2015"))
    regrePoli(2015)
    st.header(emoji.emojize("2016"))
    regrePoli(2016)
    st.header(emoji.emojize("2017"))
    regrePoli(2017)
    st.header(emoji.emojize("2018"))
    regrePoli(2018)
    st.header(emoji.emojize("2019"))
    regrePoli(2019)
    st.header(emoji.emojize("2020 🦠"))
    regrePoli(2020)
    st.header(emoji.emojize("2021 💉"))
    regrePoli(2021)

    
st.sidebar.title("Menú")
selection = st.sidebar.selectbox("Selecciona una opción", ["Inicio", "Datos", "Análisis", "Clasificaciones","Predicción", "Regresión Polinomial", "Regresión Polinomial Interactiva"])
if selection == "Inicio": 
    inicio()
elif selection == "Datos":
    datos()
elif selection == "Análisis":
    analisis()
elif selection == "Clasificaciones":
    clasificaciones()
elif selection == "Predicción":
    prediccion()
elif selection == "Regresión Polinomial":
    pruebaTotal()
elif selection == "Regresión Polinomial Interactiva":
    pruebaTotalselect()
