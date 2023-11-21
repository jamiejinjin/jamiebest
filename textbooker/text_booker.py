import base64
from pathlib import Path
import os
import requests
from time import sleep
from typing import List, Dict, Optional

import regex as re
import cv2
from important.llmsdk.openai import ChatOpenAI
import io
from PIL import Image
from joblib import Parallel, delayed
from dotenv import load_dotenv
from googleapiclient.discovery import build
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from uuid import uuid4
from requests.exceptions import ConnectionError


image_root = Path("images")


def create_hex() -> str:
    return uuid4().hex[:8]


def download_image(link: str, image_root) -> Optional[Path]:
    """
    Down load image from a link
    
    If download success, return the local path of the image
    
    If download fail, return None
    """
    image_root = Path(image_root)
    image_path = image_root / link.split("/")[-1]
    
    # in case the image is already downloaded
    if image_path.exists():
        return image_path
    try:
        res = requests.get(link)
    except ConnectionError:
        return None
    if res.status_code == 200:
        with open(image_path, "wb") as f:
            f.write(res.content)
        return image_path
    else:
        return None
    
    
def google_search(search_term, api_key, cse_id, **kwargs) -> List[str]:
    """
    You have to manage your custom search engine at:
    https://cse.google.com/cse/all

    Say you project name is <project name>
    https://console.cloud.google.com/apis/api/customsearch.googleapis.com/metrics?project=<project name>
    
    """
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, searchType='image', **kwargs).execute()
    return list(result['link'] for result in res['items'])


def merge_conversation(messages: List[Dict[str, str]]) -> str:
    """
    Merge the conversation into a string
    """
    conversation: str = ""
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
    return conversation


def build_search_phrase(chat: ChatOpenAI, question: str) -> str:
    SYSTEM_ROLE = """
    For the input question, if we want to find the related picture from google image to help answer that question.
    Like map, structure diagram, profile photo, art, etc. Like the kind you can find on a text book
    What search phrase should we use?
    Please answer only the search phrase without any other words or punctuations.
    """
    return chat(question, system=SYSTEM_ROLE)


def pack_message_with_image(content: str, image: Image):
    """
    Pack a message with an image
    """
    filename = str(image_root/ f"{create_hex()}.jpg")
    image.convert("RGB").save(filename)
    with open(filename, "rb") as image_file:
        base_64_image = base64.b64encode(
            image_file.read()).decode('utf-8')
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": content,
            },
            {
                "image_url": {
                        "url": f"data:image/jpeg;base64,{base_64_image}"
                    },
                "type": "image_url"
            },
        ]
    }


def pick_image_prompt(chat: ChatOpenAI, question: str, image: Image) -> str:
    """
    Pick the image from the 9 images
    """
    SYSTEM_ROLE = f"""
    What the best image to to help answer this question?
    Like map, structure diagram, profile photo, art, etc. Like the kind you can find on a text book.
    Please answer only the int number of the image without any other words or punctuations. like 1, 2, 3, etc.
    """
    message = pack_message_with_image(question, image)
    messages = [message,]
    reply = chat(messages, system=SYSTEM_ROLE)
    if reply:
        result = re.search(r"\d+", reply)
    try:
        return int(result[0])
    except ValueError as e:
        return 0


def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image and return it
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def combine_images(images, ):
    """
    Combine 9 or less images into a single 512 x 512 image
    """
    images = list(images)
    titles = list(f"image {i}" for i in range(len(images)))
    
    if len(images) == 0:
        return None
    elif len(images) == 1:
        return images[0].resize((512, 512))
    else:
        if len(images) > 9:
            images = images[:9]
            titles = titles[:9]
            
        fig, axs = plt.subplots(3, 3, figsize=(9, 9), )
        for ax, image, title in zip(axs.flatten(), images, titles):
            image_arr = np.array(Image.open(image))
            ax.imshow(image_arr)
            ax.set_title(title)
            ax.axis("off")

        pil_result = Image.fromarray(np.uint8(fig2img(fig)))
        plt.close(fig)
        return pil_result


def get_keys() -> List[str]:
    """
    Get the keys from the .env file
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    return GOOGLE_API_KEY, GOOGLE_CSE_ID


load_dotenv()


if "search_phrases" not in st.session_state:
    st.session_state["search_phrases"] = []


GOOGLE_API_KEY, GOOGLE_CSE_ID = get_keys()


with st.sidebar:
    model_name = st.selectbox(
        "Model",
        ["gpt-4-vision-preview",]
        )
    
    if "chat" not in st.session_state:
        from dotenv import load_dotenv
        load_dotenv("../.env")
        st.session_state["chat"] = ChatOpenAI(model_name=model_name)
        
    else:
        chat = st.session_state["chat"]
        chat.model_name = model_name

st.title("Text Booker")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask some question"):
    st.session_state.messages.append(dict(content=prompt, role="user"))
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages]
        
        # image search and  management
        # merged = merge_conversation(messages)
        search_phrase = build_search_phrase(chat, prompt)
        st.session_state["search_phrases"].append(search_phrase)
        with st.sidebar:
            st.warning(f"search phrase: '{search_phrase}'")
            img_links = google_search(
                search_phrase, GOOGLE_API_KEY, GOOGLE_CSE_ID, num=10)
        
        local_images = Parallel(n_jobs=10, backend="threading")(
            delayed(download_image)(link, image_root) for link in img_links
        )
        local_images = list(i for i in local_images if i is not None)
        with st.sidebar:
            for image in local_images:
                with Image.open(image) as img:
                    st.image(img)
        
        combined_image = combine_images(local_images)
        if combined_image is not None:
            with st.spinner("Pick the best image to help answer the question"):
                image_idx = pick_image_prompt(chat, prompt, combined_image)
            illustrate = Image.open(local_images[image_idx]).convert("RGB")
            message = pack_message_with_image(prompt, illustrate)
            messages = messages[:-1] + [message,]
        else:
            st.warning("No related images found")
        
        full_response = ""
        
        messages.insert(0, dict(
            role="system",
            content="Please answer the question, if it helps you can answer with the details in the picture"
            ))
        
        message_placeholder = st.empty()
        # with st.spinner("Answering the question"):
        for token in chat.stream(messages):
            full_response += token
            print(token, end="")
            message_placeholder.markdown(full_response + "▌")
            
        if full_response[-1] == "▌":
            full_response = full_response[:-1]
            message_placeholder.markdown(full_response)
        # st.image(illustrate)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
