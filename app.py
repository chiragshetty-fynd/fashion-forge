import json
from PIL import Image
import streamlit as st
from skimage import io
from session import Session
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")


if "image" not in st.session_state:
    st.session_state.image = None

print("Creating Session!")
if "session" not in st.session_state:
    print("Creating new session!")
    st.session_state.session = Session()
    st.session_state.kwargs = {"properties": {}}


with st.sidebar:
    st.session_state.kwargs["gender"] = st.selectbox(
        "Gender", ("Male", "Female"), placeholder="Male"
    )

    if st.session_state.kwargs["gender"] == "Male":
        st.session_state.kwargs["clothing_type"] = st.selectbox(
            "Clothing Type",
            (
                "T-Shirt",
                "Shirts",
                "Suits",
                "Blazers",
                "Coats",
                "Jackets",
                "Kurta",
                "Trousers",
                "Jeans",
                "Shorts",
                "Pants",
                "Tank-Tops",
                "Sweaters",
            ),
        )
    else:
        st.session_state.kwargs["clothing_type"] = st.selectbox(
            "Clothing Type",
            (
                "Dress",
                "Saree",
                "T-Shirt",
                "Shirts",
                "Suits",
                "Blazers",
                "Coats",
                "Jackets",
                "Kurti",
                "Trousers",
                "Jeans",
                "Shorts",
                "Pants",
                "Tank-Tops",
            ),
        )

    st.session_state.kwargs["properties"]["fabric"] = st.selectbox(
        "Fabric", ("cotton", "silk", "linen", "polyster", "denim", "woolen")
    )
    st.session_state.kwargs["properties"]["color"] = st.selectbox(
        "Color", ("black", "green", "blue", "yellow", "white", "red")
    )
    st.session_state.kwargs["properties"]["pattern"] = st.selectbox(
        "Pattern", ("plain", "stripes", "checks", "print")
    )

    generate_button = st.button("Generate", type="primary")

print("Generating!")
with st.spinner("Generating..."):
    if generate_button:
        st.session_state.image = st.session_state.session.generate_first(
            **st.session_state.kwargs
        )

print("Displaying!")
drawing_mode = st.selectbox("Drawing tool:", ("polygon", "rect", "circle"))
# st.image(image, caption='Result')
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color="#eee",
    background_image=Image.open(st.session_state.image)
    if st.session_state.image is not None
    else None,
    update_streamlit=True,
    width=Session.IMG_WIDTH,
    height=Session.IMG_HEIGHT,
    drawing_mode=drawing_mode,
    point_display_radius=0,
    key="canvas",
)

print("Prcsessing Mask")
if st.session_state.image is not None:
    print("Image is not None")
    # Do something interesting with the image data and paths
    if canvas_result is not None and canvas_result.json_data is not None:
        if len(canvas_result.json_data["objects"]) > 0:
            st.session_state.kwargs["canvas_results"] = canvas_result.json_data[
                "objects"
            ]


st.session_state.kwargs["prompt"] = st.text_input("Custom prompt", value="")
modify_generated = st.button("Revise generation", type="primary")
print(f"{st.session_state.kwargs.keys()=}")
print(f"{modify_generated=} {('canvas_results' in st.session_state.kwargs)=}")
if modify_generated and ("canvas_results" in st.session_state.kwargs):
    with st.spinner("Modifying..."):
        st.session_state.image = st.session_state.session.inpaint(
            **st.session_state.kwargs
        )
        # st.session_state.image = st.session_state.session.generate(
        #     **st.session_state.kwargs
        # )


# if __name__ == "__main__":
#     main()
