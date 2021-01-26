# ----------------------------------------------------------------------------------------------------------------------

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import gspread
import base64
from oauth2client.service_account import ServiceAccountCredentials
from pandas.io.json import json_normalize

# Creates a cache and allows data to be stored to ensure faster running
@st.cache(persist=True)

# ----------------------------------------------------------------------------------------------------------------------

# Layout functions
def _max_width_():
    """
    Streamlit is fitted to the users screen resolution
    """
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
_max_width_()

# Title
st.title('FILM CLUB STATISTICS (EST. 2020)')
st.text('')
st.text('')
st.markdown("""
A statistical exploration of Film Club, a growing record of over 600 films.
Numbers are pulled automatically from a google sheet.

James Rilett / Leo Loman / Tom Naccarato
""")

# ----------------------------------------------------------------------------------------------------------------------

# Google API and DF Build
scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('./cool-eye-301417-5b2b82c69fbb.json', scope)
gc = gspread.authorize(credentials)
spreadsheet_key = '1_6mn3Dq77vlhhcgwoijQvuX0c-tghOPZUQT2PE-HxuM'
book = gc.open_by_key(spreadsheet_key)
worksheet = book.worksheet("Sheet1")
table = worksheet.get_all_values()
films = pd.DataFrame(table[1:], columns=table[0])
films = films[['Film', 'Year', 'Directors', 'Genre', 'Seen J', 'Review J', 'Seen L', 'Review L', 'Seen N', 'Review N']]
films = films.replace(r'^\s*$', np.nan, regex=True)

# ----------------------------------------------------------------------------------------------------------------------

# Formatting
films['Review J'] = films['Review J'].str.slice(0,2)
films['Review L'] = films['Review L'].str.slice(0,2)
films['Review N'] = films['Review N'].str.slice(0,2)
films['Seen J'].fillna(value = False, inplace = True)
films['Seen L'].fillna(value = False, inplace = True)
films['Seen N'].fillna(value = False, inplace = True)
films.replace({'Yes': True, 'YES': True}, inplace = True)

# ----------------------------------------------------------------------------------------------------------------------

# Scores DataFrames
def scoresdfbuilder(i, df = films):
    """ This function receives an initial and
    generates a scores dataframe which utilises
    the respective Review (initial) column. """
    ifilms = df[['Film', 'Year', 'Directors', 'Genre', f'Seen {i}', f'Review {i}']]
    ifilms = ifilms[ifilms[f'Review {i}'].notna()]
    ifilms[f'Review {i}'] = ifilms[f'Review {i}'].str.replace('BL', '11')
    ifilms[f'Review {i}'] = ifilms[f'Review {i}'].astype(str).astype(int)
    ifilms.Year = ifilms.Year.astype(str).astype(int)
    ifilms['Decade'] = (ifilms.Year//10)*10
    imean = ifilms[f'Review {i}'].mean()
    imean = round(imean,2)

    return ifilms, imean

jfilms, jmean = scoresdfbuilder('J')
lfilms, lmean = scoresdfbuilder('L')
nfilms, nmean = scoresdfbuilder('N')

# ----------------------------------------------------------------------------------------------------------------------

# Gif
st.text('')
st.text('')
file_ = open('filmgif1.gif', 'rb')
contents = file_.read()
data_url = base64.b64encode(contents).decode('utf-8')
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------------------------------------------------

# Visualisation
## Mean Scores
st.markdown('***')
st.title('Mean Scores')
st.text('')
st.text('')

meansdata = {'Name': ['James', 'Leo', 'Naccers'], 'Mean Score': [jmean, lmean, nmean]}
meansdf = pd.DataFrame(meansdata)

st.write('James\' average film rating is {}'.format(meansdf.iloc[0,1]))
progress_bar = st.progress(0)
status_text = st.empty()
progress_bar.progress(int(meansdf.iloc[0,1])*10)

st.write('Leo\'s average film rating is {}'.format(meansdf.iloc[1,1]))
progress_bar = st.progress(0)
status_text = st.empty()
progress_bar.progress(int(meansdf.iloc[1,1])*10)

st.write('Naccers\' average film rating is {}'.format(meansdf.iloc[2,1]))
progress_bar = st.progress(0)
status_text = st.empty()
progress_bar.progress(int(meansdf.iloc[2,1])*10)

# ----------------------------------------------------------------------------------------------------------------------

## Scores Analysis
st.markdown('***')
st.title('Scores Analysis')
st.text('')
st.text('')
st.text('')


sdist = pd.DataFrame([jfilms['Review J'].value_counts(),
                      lfilms['Review L'].value_counts(),
                      nfilms['Review N'].value_counts() ])
sdist = sdist.transpose()
sdist = sdist.fillna(0)

sns.set(rc={'figure.figsize':(20,12)})
sns.set_style('white')
ax = sdist.plot.bar()
ax.set_title('Individual Score Distribution', fontsize = 35, fontweight = 'bold', y = 1.05)
ax.set_ylabel('Frequency', fontsize = 30, y = 0.5, labelpad = 20)
ax.set_xlabel('Rating', fontsize = 30, y = 1.05, labelpad = 20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(labels = ['James', 'Leo', 'Naccers'], prop={'size': 25})
st.pyplot(plt.gcf())

st.text('')
st.text('')
st.text('')

tdist = pd.DataFrame(sdist.sum(axis = 1))

sns.set(rc={'figure.figsize':(20,12)})
sns.set_style('white')
ax = tdist.plot.bar()
ax.set_title('Joint Score Distribution', fontsize = 35, fontweight = 'bold', y = 1.05)
ax.set_ylabel('Frequency', fontsize = 30, y = 0.5, labelpad = 20)
ax.set_xlabel('Rating', fontsize = 30, y = 1.05, labelpad = 20)
ax.tick_params(axis='both', which='major', labelsize=20)
for p in ax.patches:
    x = p.get_x() + p.get_width() / 2
    y = p.get_y() + p.get_height()
    ax.annotate(p.get_height(),(x-0.1, y+0.08), size = 15)
plt.legend([],[], frameon=False)
st.pyplot(plt.gcf())

# ----------------------------------------------------------------------------------------------------------------------

## Scores Standard Deviations from the Mean
st.markdown('***')
st.title('Standard Deviations')
st.text('')
st.text('')
st.markdown("""
This dataframe highlights each members standard deviation from the mean for their respective scores.
""")
st.text('')
st.text('')

stddf = pd.DataFrame(sdist.std())
stddf.columns = ['Standard Dev.']
st.dataframe(stddf)

# ----------------------------------------------------------------------------------------------------------------------

## Genre Breakdown
st.markdown('***')
st.title('Top Genres')
st.text('')
st.text('')
### Top 10 Genres

def topgenredfbuilder(i):
    """ This function receives an initial and
    generates a Top 10 genres dataframe which
    utilises the respective Review (initial) column. """
    ifilms, imean = scoresdfbuilder(f'{i}')
    igenre = pd.DataFrame(ifilms.Genre.value_counts().head(10))
    igenre.reset_index(level=0, inplace=True)
    igenre.columns = ['Genre', 'Count']

    return igenre

jgenre = topgenredfbuilder('J')
lgenre = topgenredfbuilder('L')
ngenre = topgenredfbuilder('N')
st.text('')
st.text('')

jcounts = jgenre.Count
jlabels = jgenre.Genre
my_explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
fig, ax = plt.subplots(figsize = (10,10))
ax.pie(jcounts, labels = jlabels,autopct='%1.1f%%', explode = my_explode)
ax.set_title('James\'s Top 10 Watched Genres ', fontsize = 15, fontweight = 'bold')
st.pyplot(fig)

st.text('')
st.text('')
st.text('')

lcounts = lgenre.Count
llabels = lgenre.Genre
my_explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
fig, ax = plt.subplots(figsize = (10,10))
ax.pie(lcounts, labels = llabels,autopct='%1.1f%%', explode = my_explode)
ax.set_title('Leo\'s Top 10 Watched Genres ', fontsize = 15, fontweight = 'bold')
st.pyplot(fig)

st.text('')
st.text('')
st.text('')

ncounts = ngenre.Count
nlabels = ngenre.Genre
my_explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
fig, ax = plt.subplots(figsize = (10,10))
ax.pie(ncounts, labels = llabels,autopct='%1.1f%%', explode = my_explode)
ax.set_title('Naccers\' Top 10 Watched Genres ', fontsize = 15, fontweight = 'bold')
st.pyplot(fig)

# ----------------------------------------------------------------------------------------------------------------------

### Genre Scores Breakdown
st.markdown('***')
st.title('Genre Scores Breakdown')
st.text('')
st.text('')
st.markdown("""
The following visualisations focus on the Top 10 watched genres for each member and present an average
score for each one respectively.
""")
st.text('')
st.text('')

def genredfbuilder(i):
    """ This function generates genre orientated dataframes for
    each film club member."""
    ifilms, imean = scoresdfbuilder(f'{i}')
    igenre = topgenredfbuilder(f'{i}')
    igdf = ifilms[['Genre', f'Review {i}']]
    igdf = igdf.groupby('Genre', as_index = False)[f'Review {i}'].mean().round(2)
    igdf1 = igdf.loc[igdf['Genre'].isin(list(igenre.Genre))]
    igdf1.set_index('Genre', inplace = True)
    igdf1 = igdf1.sort_values(f'Review {i}')

    return igdf1

jgdf1 = genredfbuilder('J')
lgdf1 = genredfbuilder('L')
ngdf1 = genredfbuilder('N')

st.text('')
st.text('')
st.text('')

sns.set(rc={'figure.figsize':(40,20)})
sns.set_style('white')
sns.set_palette('Set2')
ax = jgdf1.plot.barh()
ax.set_title('James\' Average Genre Scores  ', fontsize = 60, fontweight = 'bold', y = 1.05)
ax.set_ylabel('', fontsize = 15, y = 0.5)
ax.set_xlabel('Average Score', fontsize = 40, y = 1.05, labelpad = 30)
ax.set_xlim(5,8)
ax.tick_params(axis = 'both', labelsize = 50)
plt.legend([],[], frameon=False)
st.pyplot(plt.gcf())

st.text('')
st.text('')
st.text('')

sns.set(rc={'figure.figsize':(40,20)})
sns.set_style('white')
sns.set_palette('coolwarm')
ax = lgdf1.plot.barh()
ax.set_title('Leo\'s Average Genre Scores  ', fontsize = 60, fontweight = 'bold', y = 1.05)
ax.set_ylabel('', fontsize = 15, y = 0.5)
ax.set_xlabel('Average Score', fontsize = 40, y = 1.05, labelpad = 30)
ax.set_xlim(5,8)
ax.tick_params(axis = 'both', labelsize = 50)
plt.legend([],[], frameon=False)
st.pyplot(plt.gcf())

st.text('')
st.text('')
st.text('')

sns.set(rc={'figure.figsize':(40,20)})
sns.set_style('white')
sns.set_palette('hls')
ax = ngdf1.plot.barh()
ax.set_title('Naccers\' Average Genre Scores  ', fontsize = 60, fontweight = 'bold', y = 1.05)
ax.set_ylabel('', fontsize = 15, y = 0.5)
ax.set_xlabel('Average Score', fontsize = 40, y = 1.05, labelpad = 30)
ax.set_xlim(5,10)
ax.tick_params(axis = 'both', labelsize = 50)
plt.legend([],[], frameon=False)
st.pyplot(plt.gcf())
st.text('')
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------

## Decade Breakdown
st.markdown('***')
st.title('Decade Breakdown')
st.text('')
st.text('')
st.text('')

def decadedfbuilder(i):
    """ This function generates decade based dataframes for
    each film club member."""
    ifilms, imean = scoresdfbuilder(f'{i}')
    idecdf = ifilms[['Decade', f'Review {i}']]
    idecdf.Decade = idecdf.Decade.apply(str)
    idecdf = idecdf.groupby('Decade', as_index = False)[f'Review {i}'].mean().round(2)
    idecdf.set_index('Decade', inplace = True)

    return idecdf

jdecdf = decadedfbuilder('J')
ldecdf = decadedfbuilder('L')
ndecdf = decadedfbuilder('N')

sns.set(rc={'figure.figsize':(70,30)})
sns.set_style('white')
sns.set_palette('Set2')
ax = jdecdf.plot.bar()
ax.set_title('James\' Decade Scoring Breakdown', fontsize = 90, fontweight = 'bold', y = 1.05)
ax.set_ylabel('Average Score', fontsize = 80, labelpad = 30)
ax.set_xlabel('Decade', fontsize = 80, y = 5, labelpad = 40)
ax.tick_params(axis = 'both', labelsize = 50)
plt.legend([],[], frameon=False)
st.pyplot(plt.gcf())

st.text('')
st.text('')
st.text('')
st.text('')
st.text('')

sns.set(rc={'figure.figsize':(70,30)})
sns.set_style('white')
sns.set_palette('coolwarm')
ax = ldecdf.plot.bar()
ax.set_title('Leo\'s Decade Scoring Breakdown', fontsize = 90, fontweight = 'bold', y = 1.05)
ax.set_ylabel('Average Score', fontsize = 80, labelpad = 30)
ax.set_xlabel('Decade', fontsize = 80, y = 5, labelpad = 40)
ax.tick_params(axis = 'both', labelsize = 50)
plt.legend([],[], frameon=False)
st.pyplot(plt.gcf())

st.text('')
st.text('')
st.text('')
st.text('')
st.text('')

sns.set(rc={'figure.figsize':(70,30)})
sns.set_style('white')
sns.set_palette('hls')
ax = ndecdf.plot.bar()
ax.set_title('Naccers\' Decade Scoring Breakdown', fontsize = 90, fontweight = 'bold', y = 1.05)
ax.set_ylabel('Average Score', fontsize = 80, labelpad = 30)
ax.set_xlabel('Decade', fontsize = 80, y = 5, labelpad = 40)
ax.tick_params(axis = 'both', labelsize = 50)
plt.legend([],[], frameon=False)
st.pyplot(plt.gcf())

# ----------------------------------------------------------------------------------------------------------------------

## Common Films
st.markdown('***')
st.title('Common Films')
st.text('')
st.text('')
st.markdown("""
 Here we will consider films that all three members have scored and use standard deviations to determine different
 groups.
""")
st.text('')
st.text('')

### Most Disagreed
st.markdown('***')
st.subheader('Most Disagreed')
st.text('')
st.text('')

commondf = pd.merge(jfilms, lfilms, on = 'Film' , how = 'inner')
commondf1 = pd.merge(commondf, nfilms, on = 'Film', how = 'inner')
commondf1.drop(['Year_x', 'Directors_x', 'Genre_x', 'Seen J',
                'Decade_x', 'Year_y', 'Directors_y', 'Genre_y', 'Seen L',
               'Decade_y', 'Seen N'], axis = 1, inplace = True)
cols = ['Film', 'Year', 'Decade', 'Directors', 'Genre', 'Review J', 'Review L', 'Review N']
commondf1 = commondf1[cols]

col1, col2 = st.beta_columns(2)
with col1:
    stddf = pd.DataFrame(commondf1[['Review J', 'Review L', 'Review N']].std(axis = 1))
    stddf['Films'] = list(commondf1.Film)
    stddf.columns = ['STD', 'Film']
    stddf1 = stddf.sort_values('STD', ascending = False).head(10)
    st.dataframe(stddf1)
with col2:
    file_ = open('filmgif5.gif', 'rb')
    contents = file_.read()
    data_url = base64.b64encode(contents).decode('utf-8')
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )
st.text('')
st.text('')
st.text('')
st.text('')



# ----------------------------------------------------------------------------------------------------------------------

### Most Agreed
st.markdown('***')
st.subheader('Most Agreed')


col1, col2 = st.beta_columns(2)
with col1:
    stddf2 = stddf.sort_values('STD', ascending = True).head(10)
    st.dataframe(stddf2)
with col2:
    file_ = open('filmgif4.gif', 'rb')
    contents = file_.read()
    data_url = base64.b64encode(contents).decode('utf-8')
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

st.text('')
st.text('')
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------

### Top 5 Films
st.markdown('***')
st.subheader('Film Clubs Top 5 Films')
st.text('')
st.text('')
st.markdown("""
The following only considers films that have been scored by all members.
""")
st.text('')
st.text('')

col1, col2 = st.beta_columns(2)
with col1:
    cols2 = ['Film', 'Review J', 'Review L', 'Review N']
    commondf2 = commondf1[cols2]
    commondf2 = pd.DataFrame(commondf2.sum(axis = 1))
    commondf2['Film'] = list(commondf1.Film)
    commondf2.columns = ['Combined Score', 'Film']
    commondf3 = commondf2.sort_values('Combined Score', ascending = False).head()
    st.dataframe(commondf3)
with col2:
    file_ = open('filmgif3.gif', 'rb')
    contents = file_.read()
    data_url = base64.b64encode(contents).decode('utf-8')
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

st.text('')
st.text('')
st.text('')
st.text('')

# ----------------------------------------------------------------------------------------------------------------------

### Worst 5 Films
st.markdown('***')
st.subheader('Film Clubs Worst 5 Films')
st.text('')
st.text('')

col1, col2 = st.beta_columns(2)
with col1:
    commondf4 = commondf2.sort_values('Combined Score', ascending = True).head()
    st.dataframe(commondf4)
with col2:
    file_ = open('filmgif2.gif', 'rb')
    contents = file_.read()
    data_url = base64.b64encode(contents).decode('utf-8')
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

st.text('')
st.text('')
st.text('')
st.text('')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
st.markdown('***')
# Extras
