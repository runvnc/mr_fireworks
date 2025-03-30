from lib.providers.services import service
import os
import base64
from io import BytesIO
from openai import AsyncOpenAI
import json
import traceback

client = AsyncOpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key = os.environ.get("FIREWORKS_API_KEY", "NOKEY"),
)

def concat_text_lists(message):
    """Concatenate text lists into a single string"""
    # if the message['content'] is a list
    # then we need to concatenate the list into a single string
    out_str = ""
    if isinstance(message['content'], str):
        return message
    else:
        for item in message['content']:
            if isinstance(item, str):
                out_str += item + "\n"
            else:
                if 'text' in item:
                    out_str += item['text'] + "\n"
    message.update({'content': out_str})
    return message

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, 
                     temperature=0.0, max_tokens=5000, num_gpu_layers=0):
    try:
        print("Fireworks stream_chat (OpenAI compatible mode)")
        
        model_name = os.environ.get("AH_OVERRIDE_LLM_MODEL", "accounts/fireworks/models/deepseek-r1-basic")
        reasoning = False
        if "r1" in model_name:
            reasoning = True

        #last_role = messages[-1]['role']
        #second_last_role = messages[-2]['role']
        #if last_role == second_last_role:
        #    messages = messages[:-1]

        #messages = [concat_text_lists(m) for m in messages]

        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens
        )

        print("Opened stream with model:", model_name)

        async def content_stream(original_stream):
            done_reasoning = False
            if reasoning:
                yield '[{"reasoning": "'
            async for chunk in original_stream:
                delta = chunk.choices[0].delta
                if os.environ.get('AH_DEBUG') == 'True':
                    try:
                        print('\033[93m' + str(delta.content) + '\033[0m', end='')
                        #print('\033[92m' + str(chunk.choices[0].delta.content) + '\033[0m', end='')
                    except Exception as e:
                        pass
                if hasattr(delta, 'reasoning_content'):
                    # we actually need to escape the reasoning_content but not convert it to full json
                    # i.e., it's a string, we don't want to add quotes around it
                    # but we need to escape it like a json string
                    json_str = json.dumps(delta.reasoning_content)
                    without_quotes = json_str[1:-1]
                    yield without_quotes
                    print('\033[92m' + str(delta.reasoning_content) + '\033[0m', end='')
                elif hasattr(delta, 'content'):
                    if delta.content is not None and  "</think>" in delta.content:
                        yield '"}] <<CUT_HERE>>' 
                        #yield '"}, '
                        print('END REASONING!')
                        done_reasoning = True
                        continue
                    elif delta.content is None:
                        continue
                    if reasoning and not done_reasoning:
                        json_str = json.dumps(delta.content)
                        without_quotes = json_str[1:-1]
                        yield without_quotes    
                    else:
                        yield delta.content or ""

        return content_stream(stream)

    except Exception as e:
        trace = traceback.format_exc()
        print('Fireworks.ai (OpenAI mode) error:', e)
        print(trace)
        #raise

@service()
async def format_image_message(pil_image, context=None):
    """Format image for FireWorks using OpenAI's image format"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

@service()
async def get_image_dimensions(context=None):
    """Return max supported image dimensions for FireWorks"""
    return 4096, 4096, 16777216  # Max width, height, pixels
