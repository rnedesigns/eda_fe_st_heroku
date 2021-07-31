import streamlit as st
import pandas as pd
from PIL import Image

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.express as px
import plotly.graph_objects as go

app_banner = st.beta_container()
header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
graphs = st.beta_container()
model_trained = st.beta_container()
plotly_table = st.beta_container()

# Styling with the Streamlit:
# Customising CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Defining global var for background color of the app:
background_color = '#F5F5F5'

# To get CSV dataset
@st.cache
def get_data(datafile):
    ori_df = pd.read_csv(datafile)
    return ori_df

with app_banner:
    logo_col, brief_col = st.beta_columns(2)

    img = Image.open('assets/imgs/logotemp.jpg')
    # logo_col = st.image(img)
    # st.write(logo_col)
    st.image(img)

with header:
    st.title("Welcome to my first streamlit app")
    st.write('This is the dashboard for EDA i did on movies dataset from internet for predicting budget of the movie based on features such as genre, however select other input field from the list provided to predict budget as other of these fields adds to predict budget of the movie. Ex-runtime of the movie')

with dataset:
    st.header('Movies Dataset')
    st.text('Sources found from online link')
    movies_df = get_data('datasets/movies.csv')
    # st.write(movies_df.iloc[:5, :5])
    st.write(movies_df.head(10))
    # st.dataframe(movies_df)

    # To enhance the max values row-wise:
    """
    But only for numerical columns and one-hot/label encoding is quite 
    important in Data Science projects
    """
    # st.dataframe(movies_df.style.highlight_max(axis=0))

    st.write('Dataset in the form of Table (Streamlit Table)')
    st.table(movies_df.iloc[:55, 1:6])

    st.subheader('Movies name graph')
    movies = pd.DataFrame(movies_df['title'].value_counts()).head(20)
    st.bar_chart(movies)

    st.subheader('Types of Genres in this dataset are: ')
    genres_list = movies_df['genres'].unique()
    st.write(list(genres_list))
    st.write('Number of Genres:', len(movies_df['genres'].unique()))
    st.write('Number of movies in each genre:', movies_df['genres'].value_counts())
    st.write('Total number of movies in this dataset:', movies_df['genres'].value_counts().sum())

with features:
    st.header('List of features i retained from dataset')
    st.subheader('All features and their definitions are:')
    st.markdown('* **title**: This is the name of the movie')
    st.markdown('* **genres**: Type/category of the movie')
    st.markdown('* **budget**: Investment made for the movie')
    st.markdown('* **revenue**: How much returns or money made from the movie')

with graphs:
    st.header('visualization for my findings and insights')
    st.text('My model will take input from the user and provides the prediction')

with model_trained:
    st.header('Time to train the model')
    st.text('Here you can choose the hyperparameter of the model and check the performance change:')

    sel_col, dis_col = st.beta_columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimator =  sel_col.selectbox('How many trees should be there?', options=[100,200,300, 'No limit'], index=0)

    input_feature = sel_col.text_input('Which features should be used as input features?', 'Movie_ID')

    sel_col.write('List of input features from the dataset to select:')
    sel_col.write(movies_df.columns)

    if n_estimator == 'No limit':
        rfr = RandomForestRegressor(max_depth=max_depth)
    else:
        rfr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)

    input_x = movies_df[[input_feature]]
    output_y = movies_df[['budget']]

    rfr.fit(input_x, output_y)
    prediction = rfr.predict(input_x)

    dis_col.subheader('Mean absolute error of the model is:')
    dis_col.write(mean_absolute_error(output_y, prediction))

    dis_col.subheader('Mean squared error of the model is:')
    dis_col.write(mean_squared_error(output_y, prediction, squared=False))

    dis_col.subheader('R squared score of the model is:')  
    dis_col.write(r2_score(output_y, prediction))

with plotly_table:
    st.title('Movies dataset in table:')
    fig = go.Figure(data=[go.Table(
        # columnorder=[1,2],
        columnwidth=[1,1,1,1,1.3,1,1,1,1],
        header=dict(values=list(movies_df[['genres','Movie_ID','title','budget','original_language','popularity','release_date','revenue','runtime']].columns), 
                    fill_color='#FD8E72', #lightskyblue
                    align='center',
                    line_color='darkslategray'), 
        cells=dict(values=[movies_df.genres, movies_df.Movie_ID, movies_df.title, movies_df.budget, movies_df.original_language, movies_df.popularity, movies_df.release_date, movies_df.revenue, movies_df.runtime], 
                    fill_color='#E5ECF6', 
                    align='left',
                    line_color='darkslategray'))
    ])
    fig.update_layout(paper_bgcolor=background_color, height=700, margin=dict(l=5,r=5,b=5,t=5,pad=4))
    st.write(fig)
