import streamlit as st
from PIL import Image
import io
from test import process
from streamlit_image_comparison import image_comparison
from streamlit_tags import st_tags, st_tags_sidebar




def main():
    st.set_page_config(page_title="Poster Genertor", layout="centered")

    st.title(':sparkles: Poster Genertor :sparkles:')
    st.caption(':email: s3cui@ucsd.edu', unsafe_allow_html=True)
    st.caption(':yum: https://github.com/ik4sumi',unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        #image = Image.open(uploaded_file)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        prompt1 = st.text_input("Enter prompt 1 (describe the product):")
        #prompt2 = st.text_input("Enter prompt 2 (describe the poster):",placeholder="poster, phone, c4d,oc renderer, bright and cheerful, high saturation color, natural light, UI illustration, surrealism, rich in detail")
        keywords = st_tags(
                            label='Enter prompt 2 (describe the poster)',
                            text='Press enter to add more',
                            value=['poster', 'phone', 'c4d', 'oc renderer', 'bright and cheerful', 'high saturation color', 'natural light', 'UI illustration', 'surrealism', 'rich in detail'],
                            suggestions=[],
                            maxtags=-1)
        prompt2 = ','.join(keywords)
        on = st.toggle("Use refiner", value=True, key="refiner")

        if st.button('Process'):
            result_image = process(uploaded_file, prompt1, prompt2, refiner=on)

            #st.image(result_image, caption='Processed Image.', use_column_width=True)
            uploaded_image=Image.open(uploaded_file)
            image_comparison(
                img1=result_image,
                img2=uploaded_image,
                label1="Processed",
                label2="Original",
                starting_position=100
            )

if __name__ == "__main__":
    main()