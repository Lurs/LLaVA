from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
from PIL import Image


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_id = "/home/larsvi/text-generation-webui/models/llava-hf_llava-1.5-7b-hf/"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
image_url = "/home/larsvi/text-generation-webui/docs/Tjara-character.png"
#image = Image.open(requests.get(image_url, stream=True).raw)
image = Image.open(image_url)


prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nPlease describe this image\nASSISTANT:",
]

inputs = processor(prompts, images=[image, image], padding=True, return_tensors="pt").to("cuda")
for k,v in inputs.items():
  print(k,v.shape)

output = model.generate(**inputs, max_new_tokens=200)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
for text in generated_text:
  print(text.split("ASSISTANT:")[-1])



