import requests
import streamlit as st

# Setting page layout
st.set_page_config(page_title='Demo App',
                   page_icon='📷',
                   layout='wide',
                   initial_sidebar_state='expanded')

type = 'caption'
decode_method = 'beam'
question = None

with st.sidebar:
    st.header('Configuration:')

    # Upload a image
    image = st.file_uploader(label='Upload a Image...', type=('jpg', 'png'))

    # Display radio for type of prediction
    type = st.radio(label='Choose a type for prediction',
                    options=('caption', 'vqa'),
                    horizontal=True)
    
    if type == 'caption':
        decode_method = st.radio(label='Choose a decode method for caption',
                                 options=('beam', 'nucleus'),
                                 horizontal=True)
        
    elif type == 'vqa':
        question = st.text_input(label='Enter your question for the image.',
                                 placeholder='Eg. What is there in this image?')

    # Display button for submitting the image
    submit_button = st.button('Submit', use_container_width=True)

# Column for main part
col1, col2 = st.columns(2)

with col1:
    st.header('Image')
    if image is not None:  
        contents = image.getvalue()
        st.image(contents)

with col2:
    st.header('Result')

    if question:
        st.write(f'Question: {question}')
    
    if submit_button and image is not None:
        files = {'file': image.getvalue()}
    
        with st.spinner('Loading, this will be quick...'):
            params = {'type': type,
                      'decode_method': decode_method,
                      'question': question}
        
            res = requests.post(f'http://localhost:8080/predict', 
                                files=files, 
                                params=params)
            result = res.json()
            st.write(f'Prediction: {result["prediction"]}')