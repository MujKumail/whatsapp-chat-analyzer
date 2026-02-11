import streamlit as st
import preprocessor, helper
import seaborn as sns
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="WhatsApp Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DARK THEME CUSTOM CSS
st.markdown("""
<style>

div[data-testid="metric-container"] {
    background-color: #161B22;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #2A2F3A;
}

.fade-in {
    animation: fadeIn 1.2s ease-in;
}

@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)


# ANIMATED METRIC FUNCTION
def animated_metric(label, value):
    placeholder = st.empty()
    step = max(1, value // 50)

    for i in range(0, value + 1, step):
        placeholder.metric(label, i)
        time.sleep(0.01)

    placeholder.metric(label, value)


# SIDEBAR
st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is None:
    st.markdown("<h1 style='text-align: center;'>📊 WhatsApp Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #9CA3AF;'>Upload a WhatsApp chat file to begin analysis</h4>", unsafe_allow_html=True)


# MAIN LOGIC
if uploaded_file is not None:

    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)


    # DATE FILTER
    min_date = df['only_date'].min()
    max_date = df['only_date'].max()

    start_date, end_date = st.sidebar.date_input(
        "Choose Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Filter DataFrame
    df = df[(df['only_date'] >= start_date) &
            (df['only_date'] <= end_date)]

    # USER SELECT
    user_list = df['user'].unique().tolist()

    if 'group_notification' in user_list:
        user_list.remove('group_notification')

    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox(
        "Show analysis wrt",
        user_list
    )

    if st.sidebar.button("Show Analysis"):

        # Loading Animation
        with st.spinner("Analyzing chat data..."):
            time.sleep(1.5)


        # TOP STATISTICS (ANIMATED)
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            animated_metric("Total Messages", num_messages)

        with col2:
            animated_metric("Total Words", words)

        with col3:
            animated_metric("Media Shared", num_media_messages)

        with col4:
            animated_metric("Links Shared", num_links)

        st.markdown('</div>', unsafe_allow_html=True)


        # MONTHLY TIMELINE
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'])
        plt.xticks(rotation=90)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)


        # DAILY TIMELINE
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'])
        plt.xticks(rotation=90)
        st.pyplot(fig)


        # ACTIVITY MAP
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            st.subheader("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)


        # HEATMAP
        st.title("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(user_heatmap, cmap="coolwarm")
        st.pyplot(fig)


        # MOST BUSY USERS
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values)
                plt.xticks(rotation=90)
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)


        # WORDCLOUD
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        ax.axis("off")
        st.pyplot(fig)


        # MOST COMMON WORDS
        st.title("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.barh(most_common_df['word'], most_common_df['count'])
            st.pyplot(fig)

        with col2:
            st.dataframe(most_common_df)

        # EMOJI ANALYSIS
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            fig, ax = plt.subplots()
            ax.pie(
                emoji_df[1].head(),
                labels=emoji_df[0].head(),
                autopct="%0.2f"
            )
            st.pyplot(fig)


        # SENTIMENT ANALYSIS
        st.title("Sentiment Analysis")

        sentiment_df = helper.sentiment_analysis(selected_user, df)
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()

        st.subheader("Overall Sentiment Distribution")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%0.2f"
            )
            st.pyplot(fig)

        with col2:
            st.dataframe(
                sentiment_counts.reset_index().rename(
                    columns={'index': 'Sentiment', 'sentiment_label': 'Count'}
                )
            )

        # Sentiment Trend
        st.subheader("Sentiment Trend Over Time")
        trend = sentiment_df.groupby('only_date')['sentiment_score'].mean().reset_index()

        fig, ax = plt.subplots()
        ax.plot(trend['only_date'], trend['sentiment_score'])
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Extreme Messages
        st.subheader("Most Positive Messages")
        st.dataframe(
            sentiment_df.sort_values(by='sentiment_score', ascending=False)[
                ['user', 'message', 'sentiment_score']
            ].head()
        )

        st.subheader("Most Negative Messages")
        st.dataframe(
            sentiment_df.sort_values(by='sentiment_score')[
                ['user', 'message', 'sentiment_score']
            ].head()
        )





