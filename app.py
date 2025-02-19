import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
import google.generativeai as genai

# On ajoute la sidebar et on lui affecte nos pages
with st.sidebar:
        st.image("logo.png",width=100)
        selection = option_menu(
                    menu_title=None,
                    options = ["Accueil","Marché","Geopolitique","Nos données","Prediction de la tendance","Prediction du prix"]
        )

# connection a la bdd
conn = st.connection("postgresql", type="sql")

# Recuperationd des données sous forme de dataframe
#df_alpha = conn.query("SELECT * FROM alphavantage;", ttl="10m")
df_production = conn.query("SELECT * FROM api_production;", ttl="10m")
df_stock = conn.query("SELECT * FROM api_stock;", ttl="10m")
df_supply = conn.query("SELECT * FROM api_supply;", ttl="10m")
df_sentiment = conn.query("SELECT * FROM sentiment_analysis_energy;", ttl="10m")
df_indicators = conn.query("SELECT * FROM webscraping_oil_indicators;", ttl="10m")
df_yahoo = conn.query("SELECT * FROM yahoo;", ttl="10m")
df_yahoo = df_yahoo[df_yahoo["_close"]>0]

def preprod():
        global df_stock, df_production,df_supply,df_yahoo
        
        for df in [df_yahoo, df_production, df_stock, df_supply]:
                df["_date"] = pd.to_datetime(df["_date"])
                if df is not df_yahoo:
                        df["_value"] = pd.to_numeric(df["_value"])

      
        for df in [df_production, df_stock, df_supply]:
                df.set_index("_date", inplace=True)
                df = df.resample("D").ffill() 
                df.reset_index(inplace=True)

     
        df_X = df_yahoo.copy()
        df_production = df_production.rename(columns={"_value": "_value_production"})  
        df_stock = df_stock.rename(columns={"_value": "_value_stock"}) 
        df_supply = df_supply.rename(columns={"_value": "_value_supply"})   

        for df in [df_production, df_stock, df_supply]:
                df = df.select_dtypes(include=[np.number, "datetime64"])
                df_X=df_X.merge(df, on="_date",how="left")
                
        df_X["_value_production"]=df_X["_value_production"].ffill()
        df_X["_value_stock"]=df_X["_value_stock"].ffill()
        df_X["_value_supply"]=df_X["_value_supply"].ffill()
        df_X.dropna(inplace=True)
        df_X.reset_index(inplace=True,drop=True)
        return df_X

def postprod():
        global df_indicators, df_sentiment

        st.subheader("Les indicateurs techniques :")
        st.write(" ")
        df_indicators=df_indicators.rename(columns={"_date":"date","_name":"name","_value":"value","_action":"action"})
        df_indicators=df_indicators.drop(columns=["date"])
        df_indicators=df_indicators.style.apply(color_indicator, axis=1).to_html(index=False)
        st.markdown(df_indicators, unsafe_allow_html=True)
        

        st.subheader("Le sentiment des médias est :")
        st.write(" ")
        df_sentiment=df_sentiment.drop(columns=["_date"])
        df_sentiment=df_sentiment.T.reset_index()
        df_sentiment.columns = ["Source", "Sentiment"]
        df_sentiment=df_sentiment.style.apply(color_sentiment, axis=1).to_html(index=False)
        st.markdown(df_sentiment, unsafe_allow_html=True)
        
def color_sentiment(sentiment):
        if "positif" in sentiment["Sentiment"] :
                return ["background-color: darkgreen"] * len(sentiment) 
        elif "négatif" in sentiment["Sentiment"] :
                return ["background-color: darkred"] * len(sentiment) 
        else: 
                return ["background-color:hsl(199, 89.50%, 7.50%)"] * len(sentiment)

def color_indicator(indicator):
        if indicator["action"] == "Buy":
                return ["background-color: darkgreen"] * len(indicator) 
        elif indicator["action"] == "Sell":
                return ["background-color: darkred"] * len(indicator)  
        else:
                return ["background-color:hsl(199, 89.50%, 7.50%)"] * len(indicator)



#page acceuil
if selection == "Accueil":
       
        st.title("Bienvenue sur OilRush")
        st.write("Application de conseils sur le Crude oil WTI")
        st.write("Nous utilisons la puissance de l'ia couplée à une collecte de données minitieuse pour vous fournir les meilleurs insights possibles.")
        st.image("wallpaper.jpg")
        
      
#page marché
if selection == "Marché":
        st.title("Aperçu du marché :")
        
        fig = px.line(df_yahoo, x="_date", y="_close",line_shape="spline", render_mode="svg",title="Crude oil WTI prix de cloture quotidien")
        fig.update_layout(xaxis_title="Date", yaxis_title="Prix de clôture (USD)")
        st.plotly_chart(fig)

#page geopolitique
if selection == "Geopolitique":

        if st.checkbox("Accéder à une IA générative"):

                st.write(" ")
                # Création du modèle de Chatbot
                genai.configure(api_key="AIzaSyC26o1TL0urrDJ6XJSmP99vxkvPyu5Sm3g")
                model = genai.GenerativeModel('gemini-1.5-flash')

                # Création du prompt système
                system_prompt = """
                Tu es un spécialiste du Crude Oil WTI. Tu connais les éléments géopolitiques qui peuvent influencer son prix. Tu sais quelles sont les principales actualités quotidiennes.
                """

                # Initialisation de l'historique avec le prompt système
                chat = model.start_chat(history=[{'role': 'user', 'parts': [system_prompt]}])

                # Interface Streamlit
                st.title("Actualité géopolitique")
                st.write("Ce Chatbot va vous aider à prendre la meilleure décision possible.")

                # Zone de saisie pour l'utilisateur
                user_input = st.text_input(" ")

                # Si un message est envoyé
                if user_input:
                        # Envoyer le message à l'API et obtenir la réponse
                        response = chat.send_message(user_input)
                        st.markdown(response.text)

               


#page indicateurs
if selection == "Nos données":
        st.title("Nos données")
        st.write("Par soucis de transparence nous vous permettons de visualiser toutes les données que nous récoltons sur le Crude oil WTI")
        st.write("Ces données alimentent nos modèles de machine learning")

        dict_df={"df_production":df_production,"df_stock":df_stock,"df_supply":df_supply,"df_yahoo":df_yahoo}
        liste_df={"","df_production","df_stock","df_supply","df_yahoo"}
        select=st.selectbox("Choisissez un dataset",liste_df)

        for key,val in dict_df.items():
                if select == key:
                        st.dataframe(val)

#page prediction tendance
if selection == "Prediction de la tendance":
        st.title("Prediction de la tendance")
        st.write("Nous tenterons ici de prédire la tendance du prix du baril wti quotidien")

        df=preprod()
        df['tendance'] = (df["_close"].diff() > 0).astype(int)
        df.dropna(inplace=True)

        X=df.drop(columns=["tendance","_date"])
        y=df['tendance']
        scaler=MinMaxScaler()
        X=scaler.fit_transform(X)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

        xgb = XGBClassifier()
        xgb.fit(X_train, y_train)

        y_pred=xgb.predict(X_test)
        scoretrain=xgb.score(X_train,y_train)
        precision = accuracy_score(y_test, y_pred)
        
        st.subheader("Prédiction de la tendance du pétrole WTI avec XGB :")
        st.write(" ")
        #st.write(f"Score train du  modele: **{scoretrain:.2%}**")
        #st.write(f"Précision du modèle : **{precision:.2%}**")

        latest_data = X[-1].reshape(1, -1)
        next_day_trend = xgb.predict(latest_data)[0]

        if next_day_trend == 1:
                st.success("📈 Le modèle prévoit une **hausse** du prix.")
        else:
                st.error("📉 Le modèle prévoit une **baisse** du prix.")


        postprod()

#page prediction du prix
if selection=="Prediction du prix":
        st.title("Prediction du prix")
        st.write("Nous tenterons ici de prédire le prix du baril wti quotidien")
        df=preprod()
        X=df.drop(columns=["_close","_date"])
        y=df['_close']

        scaler=MinMaxScaler()
        X=scaler.fit_transform(X)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

        modelLin=KNeighborsRegressor()
        modelLin.fit(X_train,y_train)

        y_pred=modelLin.predict(X_test)
        scoretrain=modelLin.score(X_train,y_train)
        precision = modelLin.score(X_test, y_pred)
        
        st.subheader("Prédiction du prix du pétrole WTI avec KNN :")
        st.write(" ")
        #st.write(f"Score train du  modele: **{scoretrain:.2%}**")
        #st.write(f"Précision du modèle : **{precision:.2%}**")

        latest_data = X[-1].reshape(1, -1)
        next_day_trend = modelLin.predict(latest_data)[0]
        next_day_trend=round(next_day_trend,2)

        st.info(f"Le modele prévoit un prix de : **{next_day_trend}$**.")
        

        postprod()