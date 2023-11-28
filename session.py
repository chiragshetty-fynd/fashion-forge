import os
import cv2
import json
import numpy as np
import os.path as osp
from skimage import io
from openai import OpenAI
from skimage.draw import polygon2mask
from diffusers.utils import load_image
from inpaint import pipeline, generator


KEY_FILE = ".key"
ASST_ID = "asst_zOpsQCniEVnYWWK75xXJfWFT"

with open(KEY_FILE, "r") as f:
    config = json.load(f)

print("Loading Keys")
OPEN_AI_API_KEY = config["key"]
OPEN_AI_ORG_ID = config["organizationId"]


def get_assistant(client, asst_id=ASST_ID):
    assistant = client.beta.assistants.retrieve(asst_id)
    return assistant

def start_new_chat(client):
    empty_thread = client.beta.threads.create()
    return empty_thread

def add_message(client, thread, content):
    thread_message = client.beta.threads.messages.create(
        thread_id = thread.id,
        role="user",
        content=content,
    )
    return thread_message

def inverse(mask):
    mask[mask == 255] = 128
    mask[mask == 0] = 255
    mask[mask == 128] = 0
    return mask


class Session:
    IMG_WIDTH = 1024
    IMG_HEIGHT = 1024
    TEMP_DIR = "./tmp"
    # PROMPT_HEADER = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: You're a tool that lets users create custom designs for their clothing"
    # PROMPT_HEADER = "You're a tool that lets users create custom designs for their clothing"
    PROMPT_HEADER = "Generate a ultra-realistic clothing item without any person, with a plain background. Focus should be on the clothing described further. "
    os.makedirs(osp.join(TEMP_DIR, "images"), exist_ok=True)
    os.makedirs(osp.join(TEMP_DIR, "masks"), exist_ok=True)

    def __init__(
        self,
        model="dall-e-3",
        size="1024x1024",
        quality="standard",
        org_id=OPEN_AI_ORG_ID,
        api_key=OPEN_AI_API_KEY,
    ):
        print("Initializing session!")
        self.size = size
        self.model = model
        self.prompts = list()
        self.quality = quality
        self.generator = generator
        self.inpaint_pipeline = pipeline
        self.client = OpenAI(organization=org_id, api_key=api_key)
        self.prompt_enhancer = get_assistant(self.client)

    @property
    def img_path(self):
        return osp.join(Session.TEMP_DIR, "images", f"{len(self.prompts)}.png")

    @property
    def mask_path(self):
        return osp.join(Session.TEMP_DIR, "masks", f"{len(self.prompts)}.png")

    @property
    def sd_mask_path(self):
        return osp.join(Session.TEMP_DIR, "masks", f"{len(self.prompts)}.jpg")

    @staticmethod
    def properties_2_prompt(clothing_type, gender, properties):
        prompt = f"A {clothing_type} for a {gender}"
        for key, value in properties.items():
            prompt += f", with {value} {key}"
        return prompt

    def mask_gen(self, results):
        polygons = [
            np.asarray(
                [
                    list(map(int, res[::-1][:-1]))
                    for res in result["path"]
                    if len(res) == 3
                ],
                dtype=np.int32,
            )
            for result in results
        ]
        masks = [
            polygon2mask((Session.IMG_HEIGHT, Session.IMG_WIDTH), polygon) * 255
            for polygon in polygons
        ]

        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask += mask

        final_mask = final_mask.astype(np.uint8)
        rgba = io.imread(self.img_path)
        rgba[:, :, -1] = inverse(final_mask)
        io.imsave(self.mask_path, rgba)
        io.imsave(
            self.mask_path.replace(".png", ".jpg"), inverse(final_mask.astype(np.uint8))
        )

    def generate_first(self, **kwargs):

        ## Start new thread with assistant on first prompt
        self.thread = start_new_chat(self.client)

        properties = kwargs.get("properties", dict())
        gender = kwargs.get("gender", "male")
        clothing_type = kwargs.get("clothing_type", "t-shirts")
        self.prompts += [
            Session.PROMPT_HEADER
            + Session.properties_2_prompt(clothing_type, gender, properties)
        ]
        prompt = self.prompts[-1]
        enhanced_prompt = add_message(self.client, self.thread, prompt)
        response = self.client.images.generate(
            model=self.model,
            prompt=enhanced_prompt,
            size=f"{Session.IMG_HEIGHT}x{Session.IMG_WIDTH}",
            quality=self.quality,
            n=1,
        )
        io.imsave(
            self.img_path,
            cv2.cvtColor(io.imread(response.data[0].url), cv2.COLOR_RGB2RGBA),
        )
        return self.img_path

    def generate(self, **kwargs):
        print("Modifying generated image!!")
        self.mask_gen(kwargs["canvas_results"])
        mask_path = self.mask_path
        prev_img_path = self.img_path
        self.prompts += [kwargs.get("prompt")]
        prompt = ", ".join(self.prompts).replace(Session.PROMPT_HEADER, "")
        enhanced_prompt = add_message(self.client, self.thread, prompt)
        print(f'Prompts = {", ".join(self.prompts)}')
        response = self.client.images.edit(
            # model=self.model,
            image=open(prev_img_path, "rb"),
            mask=open(mask_path, "rb"),
            prompt=enhanced_prompt,
            n=1,
        )
        io.imsave(
            self.img_path,
            cv2.cvtColor(io.imread(response.data[0].url), cv2.COLOR_RGB2RGBA),
        )
        return self.img_path

    def inpaint(self, **kwargs):
        print("Modifying generated image!!")
        self.mask_gen(kwargs["canvas_results"])
        mask = load_image(self.sd_mask_path).resize(
            (Session.IMG_WIDTH, Session.IMG_HEIGHT)
        )
        prev_img = load_image(self.img_path).resize(
            (Session.IMG_WIDTH, Session.IMG_HEIGHT)
        )
        self.prompts += [kwargs.get("prompt")]
        prompt = ", ".join(self.prompts).replace(Session.PROMPT_HEADER, "")
        enhanced_prompt = add_message(self.client, self.thread, prompt)
        result = self.inpaint_pipeline(
            prompt=enhanced_prompt,
            image=prev_img,
            mask_image=mask,
            guidance_scale=8.0,
            num_inference_steps=20,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=self.generator,
        ).images[0]
        result.save(self.img_path)
        return self.img_path
