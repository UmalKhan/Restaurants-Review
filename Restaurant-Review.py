import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import anthropic
import pandas as pd
import streamlit as st
import json
import re
from typing import List
import matplotlib.pyplot as plt
from datetime import datetime
from groq import Groq
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger_eng')
your_api_key = ''
os.environ['GROQ_API_KEY'] = your_api_key
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
options = Options()
options.add_argument("--headless")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_common_terms(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tagged_tokens = pos_tag(filtered_tokens)
    adjectives = [word for word, tag in tagged_tokens if tag in ('JJ', 'JJR', 'JJS')]
    lemmatized_adjectives = [lemmatizer.lemmatize(word) for word in adjectives]
    freq_dist = FreqDist(lemmatized_adjectives)
    return freq_dist.most_common(10)

def calculate_average_sentiment(sentiments):
    sentiment_values = {'positive': 1, 'neutral': 0, 'negative': -1}
    sentiment_scores = []

    for sentiment in sentiments:
        if isinstance(sentiment, str):
            cleaned_sentiment = sentiment.strip().lower()
            if cleaned_sentiment in sentiment_values:
                sentiment_scores.append(sentiment_values[cleaned_sentiment])
            else:
                sentiment_scores.append(0)
        else:
            sentiment_scores.append(0)

    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

def calculate_average_rating(ratings):
    if isinstance(ratings, pd.Series):
        ratings = ratings.tolist()

    ratings = [rating for rating in ratings if isinstance(rating, (int, float))]

    return sum(ratings) / len(ratings) if ratings else 0


def load_data(filename: str):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_comments(review_text: str):
    parts = review_text.split('|', 5)
    while len(parts) < 5:
        parts.append("")
    food_sent = parts[0]
    food_theme = parts[1]
    staff_sent = parts[2]
    staff_theme = parts[3]
    rating = parts[4]
    return food_sent, food_theme, staff_sent, staff_theme, rating

def load_reviews(filename: str):
    df = pd.read_csv(filename)
    
    def clean_date(date_str):
        if isinstance(date_str, str):
            if "Dined" in date_str:
                if "day" in date_str:
                    days_ago = int(date_str.split(' ')[1])
                    return datetime.now() - pd.Timedelta(days=days_ago)
                else:
                    date_str = date_str.replace('Dined on ', '')
                    return datetime.strptime(date_str, "%B %d, %Y")
        return None

    df['Date'] = df['Date'].apply(clean_date)
    
    df = df.dropna(subset=['Date'])
    df.reset_index(drop=True, inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

def load_reviews_from_csv(file_path):
    df = pd.read_csv(file_path)
    reviews = df.to_dict(orient='records')
    return reviews

def review_scrapping(base_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    driver.get(base_url)
    driver.implicitly_wait(10)
    pagination = driver.find_element(By.CSS_SELECTOR, 'footer._1BEc9Aeng-Q-')
    page_numbers = pagination.find_elements(By.CSS_SELECTOR, 'a.ojKcSDzr190-')
    total_pages = 0
    for page in page_numbers:
        page_number = page.text.strip()
        if page_number.isdigit():
            total_pages = max(total_pages, int(page_number))
    driver.quit()
    # total_pages = 2
            
    restaurant_name = soup.find(class_ = 'E-vwXONV9nc-').text.strip()
    reviews = []
    for page in range(1, total_pages):
        page = str(page)
        url = base_url + f"?page={page}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        for review in soup.find_all('li', class_='afkKaa-4T28-'):
            content = review.find('span', class_='l9bbXUdC9v0- ZatlKKd1hyc- ukvN6yaH1Ds-').text.strip()
            rating = review.find('span', class_='-y00OllFiMo-').text.strip()
            date = review.find('p', class_ = 'iLkEeQbexGs-').text.strip()

            reviews.append({
                'Review Content': content,
                'Rating': rating,
                'Date': date,
            })

    df = pd.DataFrame(reviews)
    df.to_csv(f"{restaurant_name}_reviews.csv", index=False)
    return reviews, restaurant_name

def analyze_review(message,sys_message="You are a review analysis tool. Extract and separate comments about food quality and staff/service. Exclude any personal information, unnecessary formatting, intros like 'Here’s the output...' and newlines like '/n' or '//. Provide the output in a single line with only the following structure: [Food Sentiment] | [Food Themes] | [Service Sentiment] | [Service Themes] | [The rating number]. Respond only with the output and nothing else.",
        model="llama3-8b-8192"):
    message = str(message)
    messages = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(model=model, messages=messages)

    return response.choices[0].message.content

def process_reviews(reviews, res_name):
    categorized_data = []

    # try:
    #     with open(f"{res_name}_categorized_reviews.json", 'r', encoding='utf-8') as f:
    #         categorized_data = json.load(f)
    # except FileNotFoundError:
    #     pass

    for i, review in enumerate(reviews):
        result = analyze_review(review)
        categorized_data.append(result)

        with open(f"{res_name}_categorized_reviews.json", 'w', encoding='utf-8') as f:
            json.dump(categorized_data, f, indent=4)

    return categorized_data

st.markdown(
                """
                <style>
                [data-testid="stAppViewContainer"] {
                    background-color: #A83D4C;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

#               https://www.opentable.com/r/rh-rooftop-restaurant-oak-brook

user_res = st.text_input("Enter the Restaurant URL:")
loading_placeholder = st.empty()
with loading_placeholder:
    process_button = st.button("Fetch and Process Reviews")
if user_res or process_button:        
    loading_placeholder.markdown(
        """
        <p style="text-align: center; font-size: 18px; color: #ffffff; font-weight: bold;">
                Please wait, scrapping reviews...
        </p>
        <div class="loading-container">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
        <style>
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
        .dot {
            width: 15px;
            height: 15px;
            margin: 0 5px;
            background-color: #ffffff;
            border-radius: 50%;
            animation: bounce 1.5s infinite;
        }
        .dot:nth-child(2) {
            animation-delay: 0.3s;
        }
        .dot:nth-child(3) {
            animation-delay: 0.6s;
        }
        @keyframes bounce {
            0%, 80%, 100% {
                transform: scale(0);
            }
            40% {
                transform: scale(1);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    reviews, res_name = review_scrapping(user_res)     
    og_name = 'ilili'
    loading_placeholder.success("Reviews processed!")
    loading_placeholder.markdown(
        """
        <p style="text-align: center; font-size: 18px; color: #ffffff; font-weight: bold;">
                Please wait, processing reviews...
        </p>
        <div class="loading-container">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
        <style>
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }
        .dot {
            width: 15px;
            height: 15px;
            margin: 0 5px;
            background-color: #ffffff;
            border-radius: 50%;
            animation: bounce 1.5s infinite;
        }
        .dot:nth-child(2) {
            animation-delay: 0.3s;
        }
        .dot:nth-child(3) {
            animation-delay: 0.6s;
        }
        @keyframes bounce {
            0%, 80%, 100% {
                transform: scale(0);
            }
            40% {
                transform: scale(1);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    og_reviews = load_reviews_from_csv("opentable_reviews.csv")
    # reviews = load_reviews_from_csv("RH Rooftop Restaurant Oak Brook_reviews.csv")
    # res_name = "RH Rooftop Restaurant Oak Brook"
    try_reviews = reviews[:10]
    og_review = process_reviews(og_reviews[:10], og_name)
    categorized_reviews = process_reviews(try_reviews, res_name)
    comp_reviews = load_data(f"{res_name}_categorized_reviews.json")
    og_reviews = load_data(f"{og_name}_categorized_reviews.json")
    loading_placeholder.success("Reviews categorized")
    
    df = load_reviews("opentable_reviews.csv")
    comp_df = load_reviews(f"{res_name}_reviews.csv")
    
    #assert len(df) == len(comp_df)
    
    st.markdown(
        f"""
        </style>
        <div style="display: flex; justify-content: space-between; gap: 30px">
            <div style="flex: 1; display: flex; justify-content: center; align-items: center; margin-bottom: 20px; background-color: #E6CCC5 ; border: 8px solid #000000; border-radius: 15px; width: 300px;">
                <h2 style="color: #000000; text-align: center">ilili Reviews</h2>
            </div>
            </style>
            <div style="flex: 1; display: flex; justify-content: center; align-items: center; margin-bottom: 20px; background-color: #E6CCC5 ; border: 8px solid #000000; border-radius: 15px; width: 300px;">
                <h2 style="color: #000000; text-align: center">{res_name} Reviews</h2>
            </div>
        
        </div>
        """,
        unsafe_allow_html=True
    )
    
    for index, (og_review, review) in enumerate(zip(og_reviews, comp_reviews)):
        og_food_sent = og_food_comments = og_staff_sent = og_service_comments = og_rating = food_sent = food_comments = staff_sent = service_comments = rating = ' ' 
        og_food_sent, og_food_comments, og_staff_sent, og_service_comments, og_rating = extract_comments(og_review)
        food_sent, food_comments, staff_sent, service_comments, rating = extract_comments(review)
        df.loc[index, ['Food_Sentiment', 'Food_Comments', 'Staff_Sentiment', 'Service_Comments', 'Rating']] = [og_food_sent, og_food_comments, og_staff_sent, og_service_comments, og_rating]
        comp_df.loc[index, ['Food_Sentiment', 'Food_Comments', 'Staff_Sentiment', 'Service_Comments', 'Rating']] = [food_sent, food_comments, staff_sent, service_comments, rating]
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; gap: 30px;">
                <div style="flex: 1;">
                    <div style='border: 1px solid #ccc; padding: 10px; margin: 10px 0; background-color: #E6CCC5; border-radius: 15px; width: 300px;'>
                        <b style='color: #000000;'>Overall Rating:</b> <span style='color: #A83D4C;'>{og_rating}</span><br>
                        <b style='color: #000000;'>Food Quality:</b> <span style='color: #2E4053;'>{og_food_sent}: {og_food_comments}</span><br>
                        <b style='color: #000000;'>Staff/Service:</b> <span style='color: #345635;'>{og_staff_sent}: {og_service_comments}</span><br>
                    </div>
                </div>
                <div style="flex: 1;">
                    <div style='border: 1px solid #ccc; padding: 10px; margin: 10px 0; background-color: #E6CCC5; border-radius: 15px; width: 300px;'>
                        <b style='color: #000000;'>Overall Rating:</b> <span style='color: #A83D4C;'>{rating}</span><br>
                        <b style='color: #000000;'>Food Quality:</b> <span style='color: #2E4053;'>{food_sent}: {food_comments}</span><br>
                        <b style='color: #000000;'>Staff/Service:</b> <span style='color: #345635;'>{staff_sent}: {service_comments}</span><br>
                    </div>
                </div>
            
            </div>
            """,
            unsafe_allow_html=True
        )

    og_food_text = ' '.join(df['Food_Comments'].dropna())
    comp_food_text = ' '.join(comp_df['Food_Comments'].dropna())
    og_service_text = ' '.join(df['Service_Comments'].dropna())
    comp_service_text = ' '.join(comp_df['Service_Comments'].dropna())

    og_food_themes = extract_common_terms(og_food_text)
    comp_food_themes = extract_common_terms(comp_food_text)
    og_service_themes = extract_common_terms(og_service_text)
    comp_service_themes = extract_common_terms(comp_service_text)
    
    og_sentiments = df[['Food_Sentiment', 'Staff_Sentiment']]
    comp_sentiments = comp_df[['Food_Sentiment', 'Staff_Sentiment']]

    og_ratings = df['Rating']
    comp_ratings = comp_df['Rating']
    
    og_food_sentiment_avg = calculate_average_sentiment(og_sentiments['Food_Sentiment']) * 1000
    og_Staff_Sentiment_avg = calculate_average_sentiment(og_sentiments['Staff_Sentiment']) * 1000
    
    if (og_food_sentiment_avg > 0.5):
        og_food_sentiment = "Positive"
    elif (og_food_sentiment_avg > 0):
        og_food_sentiment = "Neutral"
    else:
        og_food_sentiment = "Negative"

    if (og_Staff_Sentiment_avg > 0.5):
        og_staff_sentiment = "Positive"
    elif (og_Staff_Sentiment_avg > 0):
        og_staff_sentiment = "Neutral"
    else:
        og_staff_sentiment = "Negative"

    # comp_food_sentiment_avg = calculate_average_sentiment(comp_sentiments['Food_Sentiment']) * 1000
    # comp_Staff_Sentiment_avg = calculate_average_sentiment(comp_sentiments['Staff_Sentiment']) * 1000
    
    comp_food_sentiment_avg = calculate_average_sentiment(comp_sentiments['Food_Sentiment'])
    comp_Staff_Sentiment_avg = calculate_average_sentiment(comp_sentiments['Staff_Sentiment'])
    
    if (comp_food_sentiment_avg > 0.5):
        comp_food_sentiment = "Positive"
    elif (comp_food_sentiment_avg > 0):
        comp_food_sentiment = "Neutral"
    else:
        comp_food_sentiment = "Negative"
        
    if (comp_Staff_Sentiment_avg > 0.5):
        comp_staff_sentiment = "Positive"
    elif (comp_Staff_Sentiment_avg > 0):
        comp_staff_sentiment = "Neutral"
    else:
        comp_staff_sentiment = "Negative"

    og_avg_rating = calculate_average_rating(og_ratings)
    comp_avg_rating = calculate_average_rating(comp_ratings)


    df = load_reviews("opentable_reviews.csv")
    comp_df = load_reviews(f"{res_name}_reviews.csv")

    fig, ax = plt.subplots(2, figsize=(15, 8))

    ax[0].plot(df['Date'].iloc[::200], df['Rating'].iloc[::200], label="ilili Ratings", color='black')
    # ax[1].plot(comp_df['Date'][::200], comp_df['Rating'][::200], label=f"{res_name} Ratings", color='black')
    ax[1].plot(comp_df['Date'], comp_df['Rating'], label=f"{res_name} Ratings", color='black')

    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Rating")
    ax[0].set_title("ilili Rating Trend Over Time")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Rating")
    ax[1].set_title(f"{res_name} Rating Trend Over Time")

    col1, col2 = st.columns(2)

    with col1:
        st.header("ilili")
        st.subheader("Sentiment Averages")
        st.write(f"Food Sentiment Avg: {og_food_sentiment}")
        st.write(f"Service Sentiment Avg: {og_staff_sentiment}")
        st.subheader("Average Rating")
        st.write(f"Avg Rating: {og_avg_rating:.2f}")
        st.subheader("Top 5 Food Themes")
        for theme, count in og_food_themes[:5]:
            st.write(f"  - {theme.capitalize()} (mentioned {count} times)")
        st.subheader("Top 5 Service Themes")
        for theme, count in og_service_themes[:5]:
            st.write(f"  - {theme.capitalize()} (mentioned {count} times)")

    with col2:
        st.header(f"{res_name}")

        st.subheader("Sentiment Averages")
        st.write(f"Food Sentiment Avg: {comp_food_sentiment}")
        st.write(f"Service Sentiment Avg: {comp_staff_sentiment}")

        st.subheader("Average Rating")
        st.write(f"Avg Rating: {comp_avg_rating:.2f}")

        st.subheader("Top 5 Food Themes")
        for theme, count in comp_food_themes[:5]:
            st.write(f"  - {theme.capitalize()} (mentioned {count} times)")

        st.subheader("Top 5 Service Themes")
        for theme, count in comp_service_themes[:5]:
            st.write(f"  - {theme.capitalize()} (mentioned {count} times)")

    st.markdown("---")
    st.subheader("Summary Comparison")

    st.pyplot(fig)
