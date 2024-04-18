import base64
import streamlit as st
import pandas as pd
import os
import yfinance as yf
import yesg
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import google.generativeai as genai
import nltk
from datetime import datetime


# os.chdir('E:\projects\data_quest')

st.set_page_config(layout="wide")

icon_image = 'icon.jpg'

genai.configure( api_key='AIzaSyCzgzDiCEy_FCJK0HXcBRY7jeYQQAy4mHI' )

companies_df = pd.read_excel( 'forturne_df.xlsx', sheet_name='Data' )

beta = pd.read_excel("forturne_df.xlsx", sheet_name='Parameters')

gemini_model = genai.GenerativeModel('gemini-pro')

nltk.download('vader_lexicon')

### Function for model score
def FinancialScore(comp,Para):
    """FinancialScore('DELL',beta)"""
    try:    
        df=pd.DataFrame()
        print(comp)
        companyData = yf.Ticker(comp)
        
        #Balance sheet
        df10=Para[Para['Segment']=='Balance sheet'][['Variable', 'Imputation']]
        df1=companyData.quarterly_balance_sheet.reset_index()
        if df1.empty:
            #blank1.append(comp)
            print("Balance sheet data unavailable.")
            #setting as imputed value
            df1=df10[['Variable']]#pd.DataFrame()#Para[Para['Segment']=='Balance sheet'][['Variable', 'Imputation']]
        df1.rename(columns={'index':'Variable'}, inplace=True)
        df12=df10.merge(df1, how='left', on='Variable')
        df=pd.concat([df,df12])
        df.reset_index(drop=True, inplace=True)
        #print("Balance")
        #print(df)
        
        #cash flow
        df30=Para[Para['Segment']=='Cash flow'][['Variable', 'Imputation']]
        df3=companyData.quarterly_cash_flow.reset_index()
        if df3.empty:
            #blank3.append(comp)
            print("Cash flow data unavailable.")
            #setting as imputed value
            df3=df30[['Variable']]#pd.DataFrame()#Para[Para['Segment']=='Cash flow'][['Variable', 'Imputation']]
        df3.rename(columns={'index':'Variable'}, inplace=True)
        df32=df30.merge(df3, how='left', on='Variable')
        df=pd.concat([df,df32])
        df.reset_index(drop=True, inplace=True)
        #print("cash")
        #print(df)
        
        #financials
        df40=Para[Para['Segment']=='Financials'][['Variable', 'Imputation']]
        df4=companyData.quarterly_financials.reset_index()
        if df4.empty:
            #blank4.append(comp)
            print("Financial data unavailable.")
            #setting as imputed value
            df4=df40[['Variable']]#pd.DataFrame()#Para[Para['Segment']=='Financials'][['Variable', 'Imputation']]
        df4.rename(columns={'index':'Variable'}, inplace=True)
        df42=df40.merge(df4, how='left', on='Variable')
        df=pd.concat([df,df42])
        df.reset_index(drop=True, inplace=True)
        #print("fins")
        #print(df)
        
        # If Imputated, then column name set as earliest for further fallback process
        df.rename(columns={'Imputation':datetime(1971,1,1)}, inplace=True)
        #print("imputation rename")
        #print(df)
        
        # fetching focus cols
        #df=df[df['Variable'].isin(list(Para['Variable']))].reset_index(drop=True)
        #print(df)
        
        # extracting months
        #months=[ts.month for ts in df.columns[1:]]
        # saving sorted months for further fallback process
        cols_sorted=sorted(df.columns[1:])
        cols_sorted.sort()
        # replacing column names from timestamp to months
        #df.columns=['Variable']+months
            
        i=0
        # Missing imputation : Set a fallback value as next column
        for i in range(1,len(cols_sorted)):
            #print(cols_sorted[i])
            df[cols_sorted[i]] = df[cols_sorted[i]].fillna(df[cols_sorted[i-1]])
            #print(df)
        
        latest_col=cols_sorted[i]
        #considering only column with latest value imputation
        data_comp0 = df[['Variable',latest_col]]
        data_comp=data_comp0.merge(Para, how='left', on='Variable')
        #print(data_comp)
        
        # checking if still any of the variable has missing value
        MissingFlag = list(data_comp.isna()[latest_col])
        if any(x is True for x in MissingFlag):
            #FinalScoreList.append("Data Missing")
            print('Missing imputation for : '+ comp+' ===============================')
            print(data_comp0)
            Dict = {'Company': comp, 'Result': 'Data unavailable for some variable'} 
            print(Dict)
        
        ModelScore=256.85072+sum(data_comp[latest_col]*data_comp['Parameter'])
        FinalScore=85+(ModelScore/24)
        #print("Model score: "+str(ModelScore))
        #print("Final score: "+str(FinalScore))
        #FinalScoreList.append(FinalScore)
        Dict = {'Company': comp, 'Result': FinalScore} 
        print(Dict)
        
        FinalScore = round( FinalScore, 3 )
    
    except:
        Dict = {'Company': comp, 'Result': 'Data unavailable'} 
        print(Dict)
        FinalScore = None
    
    return(FinalScore)


# function for ESG score
def ESGscore(comp):
    """ESGscore('CARR')"""
    try:
        comp_esg = yesg.get_historic_esg(comp).tail(1)['Total-Score'][0]
        
        return( comp_esg )
    except:
        return( None )

# Function to perform sentiment analysis on a given text
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to get news sentiment for a given ticker symbol
def get_news_sentiment(ticker):
    
    company = yf.Ticker(ticker)
        
    news_data = company.news
    
    sentiments = [analyze_sentiment(news['title']) for news in news_data]
    
    sentiment_counts = pd.Series(sentiments).value_counts(normalize=True) * 100
    
    sentiment_df = pd.DataFrame( sentiment_counts )
        
    return sentiment_df.T


def main():
    
    col0_0, col0_1 = st.columns([10,1])
    
    with col0_0:
        st.title(":orange[First ]:green['AI']:orange[d]")
        
    with col0_1:
        st.image(icon_image, width=100)   
    
    
    col01, col02 = st.columns([5, 3])
    
    with col01:
        company_name = st.selectbox( 'Company Name', list( companies_df.Company ) )
        
    company_index = companies_df[companies_df['Company'] == company_name].index[0]
    
    ticker_name = companies_df.loc[companies_df['Company'] == company_name, 'Ticker'].values[0]
    
    Top5competitors = companies_df[ (companies_df['Industry'] == companies_df.loc[company_index].Industry) ].head()
    
    # & (companies_df['Company'] != companies_df.loc[company_index].Company)
    
    Top5competitors.reset_index( inplace = True, drop = True )
            
    print( company_name )
        
    with col02:
        st.write('')
        st.write('')
        search_button = st.button("Submit")
    
    sentiment_df = get_news_sentiment( ticker_name )
    
    sentiment_df.index = [ ticker_name ]
    
    sentiment_data = pd.DataFrame()
    
    esg_data = pd.DataFrame()
    
    financial_score_data = pd.DataFrame()
    

    for i in range( Top5competitors.shape[0] ):
        
        company = Top5competitors.Company[i]
        
        ticker = Top5competitors.Ticker[i]
        
        curr_company_sentiment = get_news_sentiment(company)
        
        curr_company_sentiment.index = [ company ]
                
        sentiment_data = pd.concat([sentiment_data, curr_company_sentiment])
        
        curr_esg_df = pd.DataFrame( { 'Company': company, 'ESG_Score': ESGscore( ticker ) }, index=[0] )
        
        esg_data = pd.concat( [ curr_esg_df, esg_data ] )
        
        curr_finscore_df = pd.DataFrame( { 'Company': company, 'Financial_Score': FinancialScore( ticker, beta ) }, index=[0] )
        
        financial_score_data = pd.concat( [ curr_finscore_df, financial_score_data ] )
        
        
    
    # Resetting index after concatenation
    # sentiment_data.reset_index( inplace = True )
    
    # Create DataFrame for sentiment data
    sentiment_plot_df = pd.DataFrame(sentiment_data).fillna(0)
    
    # Define colors
    colors = ['limegreen', ' coral', 'white']
    
    fig = px.bar(sentiment_plot_df, x=sentiment_plot_df.index, y=['Positive', 'Negative', 'Neutral'], 
             color_discrete_sequence=colors, barmode='stack',  width = 400,
             title='Market Sentiment for ' + company_name)
    
    fig.update_xaxes(title_text='Company')
    
    fig.update_yaxes(title_text='Sentiment Score')
    
        
    col3, col4, col5, col6 = st.columns(4)
    
    col3.metric( "Financial Score", "", str( FinancialScore( ticker_name, beta ) ) )
    
    with col4:
        st.metric( "ESG Score", "", str( ESGscore( ticker_name ) ) )
        
    with col5:
        sentiment_df_0 = sentiment_df.copy()
        
        sentiment_df_0.insert(0, 'Sentiment', ['Score'])
        
        st.dataframe( sentiment_df_0, hide_index=True )
    
    col7, col8 = st.columns( [ 4, 2 ] )
    
    
    with col7:
        #st.plotly_chart(fig )
        
        st.subheader(f':green[Credit Report - {company_name}]', divider='orange')
        
        response = gemini_model.generate_content(f"You're an advanced AI system specializing in generating detailed financial reports to assist relationship managers in making credit disbursal decisions for {company_name}. Your task is to provide a comprehensive report that will aid a relationship manager in deciding whether to disburse credit to {company_name} based on several factors. The report should include bullet points outlining reasons, current numbers, and market sentiment to justify why or why not the loan should be granted.Please generate a report with the following points:- Reason for considering credit disbursement to {company_name}- Current financial numbers of {company_name}- Market sentiment regarding {company_name}'s creditworthiness- Justification on whether the credit should be disbursed to {company_name} based on the above factorsFor instance, you would mention the reason for considering credit disbursement to {company_name} as their strong cash reserves and consistent revenue growth. You would proceed to provide {company_name}'s current financial numbers such as revenue, profit, and debt levels. Discuss the market sentiment, citing analysts' opinions on {company_name}'s market performance and credit risk. End the report with a conclusive recommendation on whether the credit should be disbursed to {company_name} or not based on the information provided. Include recent events and news also.")
        
        st.markdown( response.text )
        
    with col8:
        
        st.subheader(f':red[ Comparison with Industry Peers]', divider='orange')
        
        st.markdown( '<span style="color: salmon; font-weight: bold;">Financial Score</span>', unsafe_allow_html=True )
        
        st.dataframe( financial_score_data, hide_index=True )
        
        st.markdown( '<span style="color: salmon; font-weight: bold;">ESG Score</span>', unsafe_allow_html=True )
        
        st.dataframe( esg_data, hide_index=True )
        
        st.markdown('<span style="color: salmon; font-weight: bold;">Market Sentiment</span>', unsafe_allow_html=True)
        
        st.write( sentiment_plot_df )
        
    
    

if __name__ == "__main__":
    main()