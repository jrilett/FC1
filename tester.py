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
