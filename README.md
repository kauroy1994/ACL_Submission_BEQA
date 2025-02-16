# BEQA Documentation

## A. **Installation**
Use pip install and obtain the following dependencies:

```
!pip install datasets pydantic beautifulsoup4
!pip install sentence-transformers
!pip install scikit-learn
!pip install nltk
!pip install rouge_score
!pip install datasets
!pip install fuzzywuzzy
```

## B. **Running Instructions**

- Refer the `.ipynb` file for a demonstration of how to run and extend the code for custom datasets and retrievers

## C. **Helpful References** 

### Using Other APIs for instead of GROQ in BEQA

The **BEQA transform** involves using a Large Language Model to produce a structured JSON output (e.g. an answer with specific fields) from a given input. Groq’s JSON mode ensures the output is valid JSON or returns an error if the model fails to produce proper JSON ([GroqCloud](https://console.groq.com/docs/text-chat#:~:text=Error%20Code%3A)). Following the same pattern, we can implement BEQA with other LLM APIs by:

1. **Prompt Construction:** Clearly instruct the model (via system or user message) to output a JSON with the desired structure (e.g. specific keys and types).  
2. **API Request:** Call the model’s API (OpenAI, Ollama, Cohere, Mistral, etc.), using any available parameter to enforce JSON formatting (many APIs offer a `response_format={"type":"json_object"}` option ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,that%20parse%20into%20valid%20JSON)) ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=15%2016%20print%28res.message.content))).  
3. **Parse and Validate:** Load the returned string into a Python JSON object (e.g. with `json.loads`). Optionally verify that required keys are present and types are correct.  
4. **Retry on Failure:** If the response isn’t valid JSON (or is missing fields), implement a retry mechanism – e.g. resend the request (possibly with a reinforced instruction or slight modification) up to a few times. Also handle API errors (rate limits, timeouts) by catching exceptions and retrying after a delay.

Below are **analogous Python implementations** for multiple LLM services, each following the above structure. Each code snippet assumes you have the respective API’s latest Python SDK installed and an API key (if required). Comments in the code highlight each step.

#### Using OpenAI API (ChatGPT/GPT-4)

OpenAI’s latest models support a *JSON mode* that guarantees valid JSON output when enabled ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,that%20parse%20into%20valid%20JSON)). We still include an explicit instruction in the prompt to define the expected JSON structure ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,is%20constrained%20to%20only%20generate)), and then use the `response_format={"type":"json_object"}` parameter. The code below shows how to use the ChatCompletion API to get a JSON answer, parse it, and retry on JSON errors:

1. **Construct Prompt:** We use a system message to tell the assistant to output only JSON with specific keys (e.g. `"answer"` and `"confidence"`).  
2. **API Call:** We call `openai.ChatCompletion.create` with a model that supports JSON mode (e.g. `"gpt-4-1106-preview"` or a later version) and include `response_format={"type": "json_object"}` ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,that%20parse%20into%20valid%20JSON)).  
3. **Parse JSON:** We attempt to parse the returned text to a Python dict using `json.loads`.  
4. **Retry Logic:** If parsing fails (or required fields are missing), we retry the API call up to 3 times, with a brief delay between attempts.

```python
import openai, json, time

openai.api_key = "YOUR_OPENAI_API_KEY"

# 1. Prompt with instructions for JSON output
messages = [
    {"role": "system", "content": "You are a helpful assistant that **only** outputs a JSON object with keys 'answer' and 'confidence'."},
    {"role": "user", "content": "<<<USER_INPUT_HERE>>>"}  # e.g. a question or text for the BEQA transform
]

max_retries = 3
result_data = None
for attempt in range(max_retries):
    try:
        # 2. OpenAI ChatCompletion API call with JSON mode enabled
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",  # GPT-4 model supporting JSON mode ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,that%20parse%20into%20valid%20JSON))
            messages=messages,
            response_format={"type": "json_object"}  # request valid JSON output
        )
        output_text = response['choices'][0]['message']['content']
        
        # 3. Parse the JSON string to a Python dict
        result_data = json.loads(output_text)
        
        # Optional: validate expected keys
        if 'answer' not in result_data or 'confidence' not in result_data:
            raise KeyError("Missing expected keys in JSON")
        # If we reach here, JSON is parsed and valid
        break  
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Attempt {attempt+1}: JSON parsing failed - {e}")
        if attempt < max_retries - 1:
            # 4. Retry after reinforcing prompt or waiting briefly
            time.sleep(1)  # small delay before retry
            # Optionally, reinforce instruction in the system prompt if needed:
            messages[0]["content"] += " Remember: output must be a valid JSON."
            continue
        else:
            raise  # re-raise error if final attempt fails

# Use result_data (parsed JSON) as needed
print(result_data)
```

**How it works:** The system prompt forces JSON formatting, and the `response_format={"type":"json_object"}` parameter constrains GPT-4 to only produce valid JSON ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,that%20parse%20into%20valid%20JSON)). OpenAI’s documentation notes that without this parameter, even well-crafted prompts can occasionally yield invalid JSON ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,is%20constrained%20to%20only%20generate)), so using the JSON mode greatly reduces errors. If the model still returns malformed JSON or misses keys, the code catches the error and retries. 

#### Using Ollama (Local LLM via Ollama API)

[Ollama](https://ollama.ai) is a tool to run open-source LLMs locally, exposing a RESTful API. We can use the **Ollama Python library** for convenience ([GitHub - ollama/ollama-python: Ollama Python library](https://github.com/ollama/ollama-python#:~:text=from%20ollama%20import%20chat%20from,ollama%20import%20ChatResponse)). Ollama doesn’t have a built-in JSON-enforcement mode, so we rely purely on prompt instructions and post-processing.

1. **Construct Prompt:** We create a system-style instruction (though Ollama uses a similar message format) telling the model to output only JSON with the required structure.  
2. **API Call:** Using the `ollama.chat` function (or a direct HTTP request to the local server), we send the prompt to a chosen model. In this example we assume a model is already downloaded (e.g. `"llama2-uncensored"` or any available model). We disable streaming to get the full response at once.  
3. **Parse JSON:** Attempt to parse the response text with `json.loads`.  
4. **Retry Logic:** If parsing fails, we can retry the generation. With local models, another strategy is to adjust the prompt (e.g. make the instruction more explicit) and try again.

```python
import json, time
from ollama import chat  # Ollama Python SDK

# 1. Define prompt messages for Ollama
messages = [
    {"role": "system", "content": "You are a JSON formatter. You will output **only** a JSON object with keys 'answer' and 'confidence'."},
    {"role": "user", "content": "<<<USER_INPUT_HERE>>>"}
]

max_retries = 3
result_data = None
for attempt in range(max_retries):
    # 2. Call Ollama API (local) with the messages
    response = chat(model="llama2-uncensored", messages=messages, stream=False)
    # Ollama's response is a ChatResponse object; get the assistant content:
    output_text = response['message']['content']  # or response.message.content
    
    try:
        # 3. Parse JSON output
        result_data = json.loads(output_text)
        # validate keys if needed
        if 'answer' not in result_data or 'confidence' not in result_data:
            raise KeyError("Missing keys")
        break  # parsing successful
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Ollama attempt {attempt+1} failed to produce valid JSON: {e}")
        if attempt < max_retries - 1:
            # 4. Retry after a short delay or by reiterating instruction
            time.sleep(0.5)
            # Strengthen the instruction in prompt (e.g. remind the model):
            messages[-1]["content"] += "\n(Remember: output only JSON.)"
            continue
        else:
            raise

print(result_data)
```

**Explanation:** We use Ollama’s chat endpoint (by default hosted at `localhost:11434`) via the Python SDK. The usage is similar to OpenAI’s: we provide a list of message dictionaries and get a response object ([GitHub - ollama/ollama-python: Ollama Python library](https://github.com/ollama/ollama-python#:~:text=from%20ollama%20import%20chat%20from,ollama%20import%20ChatResponse)). We explicitly instruct the model to act as a JSON formatter. Since there's no guaranteed JSON mode, the prompt is crucial. The code parses the output and retries if it’s not valid JSON. Each retry can reinforce the instruction (as shown by appending a reminder to the user prompt) before calling the model again.

*Note:* Running this code requires an Ollama instance with the specified model. The parameter `stream=False` (or using the `/api/generate` endpoint) ensures we get a complete JSON in one response (streaming would otherwise yield partial chunks) ([Using the Ollama API to run LLMs and generate responses locally - DEV Community](https://dev.to/jayantaadhikary/using-the-ollama-api-to-run-llms-and-generate-responses-locally-18b7#:~:text=1,is%20used%20to%20generate%20a)). 

#### Using Cohere API

Cohere’s API offers structured output features similar to OpenAI’s. In **Cohere’s Chat API (v2)**, you can set `response_format={"type": "json_object"}` to guarantee the model’s reply is valid JSON ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=15%2016%20print%28res.message.content)). Cohere **requires** that you also explicitly prompt the model to produce JSON, otherwise the generation can hang or fail ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=)). We’ll use Cohere’s Python SDK to demonstrate this:

1. **Construct Prompt:** For Cohere, we include the instruction directly in the user message (or you can use a system message if supported) – e.g. *“Generate a JSON with fields X and Y for the following input…”*. This ensures the model knows to output JSON ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=)).  
2. **API Call:** We use `cohere.ClientV2.chat` with the `model` of choice (e.g. a Command model) and include `response_format={"type": "json_object"}` in the request. This forces a JSON output from Cohere’s model ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=15%2016%20print%28res.message.content)).  
3. **Parse JSON:** Read the model’s response text and convert it using `json.loads`.  
4. **Retry Logic:** If parsing or validation fails, retry the API call. (In practice, Cohere’s JSON mode should rarely return invalid JSON if prompted correctly.)

```python
import cohere, json, time

co = cohere.ClientV2(api_key="YOUR_COHERE_API_KEY")

# 1. Prompt instructing JSON output (Cohere may treat the first message as user if no system role given)
messages = [
    {"role": "user", "content": "Generate a JSON object with fields 'answer' and 'confidence'. Base it on the following input:\n<<<USER_INPUT_HERE>>>"}
]

max_retries = 3
result_data = None
for attempt in range(max_retries):
    # 2. Call Cohere Chat API with JSON response_format
    response = co.chat(
        model="command-r-plus-08-2024",  # e.g. use latest Command model ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=4%205%20res%20%3D%20co,))
        messages=messages,
        response_format={"type": "json_object"}
    )
    # Cohere's response object has the generated message(s)
    output_text = response.message.content[0].text  # get text of the first (and only) response message ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=15%2016%20print%28res.message.content))

    try:
        # 3. Parse JSON output
        result_data = json.loads(output_text)
        # Check for expected keys
        if 'answer' not in result_data or 'confidence' not in result_data:
            raise KeyError("JSON missing keys")
        break
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Cohere attempt {attempt+1} JSON parse failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(1)
            # If needed, reiterate instruction in the prompt
            messages[0]["content"] += "\n(Remember to respond only with a JSON.)"
            continue
        else:
            raise

print(result_data)
```

**Details:** We used a `user` role message to tell the model exactly what JSON to output (you could prepend a system role message if the SDK supports it). The `response_format={"type": "json_object"}` flag activates **JSON mode** on Cohere’s API, so the returned content is guaranteed to be valid JSON syntax ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=15%2016%20print%28res.message.content)). (The model will literally not return anything non-JSON in this mode.) As Cohere’s docs note, it’s important the prompt explicitly says *“Generate a JSON…”* to avoid the model getting confused ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=)). The code then parses the JSON and verifies the keys. If something is wrong, it logs an error and retries with a reinforced prompt. 

#### Using Mistral API

[Mistral AI](https://mistral.ai) provides an API for their models (e.g. Mistral 7B and larger). They similarly support a JSON mode: by setting `response_format={"type": "json_object"}`, the Mistral API will attempt to return only valid JSON ([JSON mode | Mistral AI Large Language Models](https://docs.mistral.ai/capabilities/structured-output/json_mode/#:~:text=Users%20have%20the%20option%20to,of%20our%20models%20through%20API)). We’ll use Mistral’s Python client (`mistralai` library) to replicate the BEQA transform steps:

1. **Construct Prompt:** Provide a system message describing the JSON format, and a user message with the actual input. For example, system: *“You are a data assistant that replies with a JSON containing ...”*.  
2. **API Call:** Use `client.chat.complete` with the desired model and include the `response_format={"type":"json_object"}` parameter ([JSON mode | Mistral AI Large Language Models](https://docs.mistral.ai/capabilities/structured-output/json_mode/#:~:text=chat_response%20%3D%20client,%7D)). This triggers Mistral’s JSON output mode.  
3. **Parse JSON:** Load the returned string into a Python dict. If Mistral’s JSON mode fails to produce valid JSON, their API might return an error (HTTP 400) that you can catch ([GroqCloud](https://console.groq.com/docs/text-chat#:~:text=Error%20Code%3A)) – in our code, we’ll handle it similarly to a parse failure.  
4. **Retry Logic:** On failure, wait briefly and retry the call. You might also change the prompt or use a different model (Mistral’s docs suggest some models are better at JSON output than others ([GroqCloud](https://console.groq.com/docs/text-chat#:~:text=Recommendations%20for%20best%20beta%20results%3A))).

```python
import json, time
from mistralai import Mistral

client = Mistral(api_key="YOUR_MISTRAL_API_KEY")

messages = [
    {"role": "system", "content": "You are an assistant that outputs **only** JSON. The JSON should have keys 'answer' and 'confidence'."},
    {"role": "user", "content": "<<<USER_INPUT_HERE>>>"} 
]

max_retries = 3
result_data = None
for attempt in range(max_retries):
    try:
        # 2. Call Mistral chat completion with JSON mode
        response = client.chat.complete(
            model="mistral-large-latest",  # latest Mistral model in the cloud ([JSON mode | Mistral AI Large Language Models](https://docs.mistral.ai/capabilities/structured-output/json_mode/#:~:text=client%20%3D%20Mistral,chat_response%20%3D%20client.chat.complete))
            messages=messages,
            response_format={"type": "json_object"}
        )
        output_text = response.choices[0].message.content  # get the assistant's JSON string
        # 3. Parse JSON
        result_data = json.loads(output_text)
        if 'answer' not in result_data or 'confidence' not in result_data:
            raise KeyError("Missing keys in JSON")
        break  # success
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Mistral attempt {attempt+1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(1)
            # Optionally refine prompt or use a different model variant if available
            continue
        else:
            raise

print(result_data)
```

**Notes:** The `mistralai` client usage is straightforward: you initialize the client with your API key, then call `chat.complete` with messages and parameters ([JSON mode | Mistral AI Large Language Models](https://docs.mistral.ai/capabilities/structured-output/json_mode/#:~:text=%7B%20,model%2C%20messages%20%3D%20messages)) ([JSON mode | Mistral AI Large Language Models](https://docs.mistral.ai/capabilities/structured-output/json_mode/#:~:text=)). Mistral’s JSON mode works like OpenAI’s and Cohere’s, requiring an instructive prompt and the `response_format` flag. The example above assumes the cloud API – if instead you were running a Mistral model locally with their `mistral-inference` library, you’d omit `response_format` and just rely on prompt instructions, similar to the Ollama case.

##### Using Anthropic’s Claude API (Claude 2)

Anthropic’s Claude models do not (as of now) have a specific JSON-enforcement parameter, but they are quite good at following formatting instructions. The approach is to **strongly prompt** Claude to respond with JSON, and then validate and retry if needed. Anthropic’s Python SDK allows a chat-style interface for Claude (Claude 2, Claude Instant, etc.) ([GitHub - anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python#:~:text=messages%3D%5B%20%7B%20,latest%22%2C%20%29%20print%28message.content)). Here’s how we can implement the BEQA transform:

1. **Construct Prompt:** We’ll use a system message to describe the JSON output format, and pass the user query as the user message. (Anthropic’s API treats the first message as system if you label it so.) For example: *system:* “You are an assistant that replies in JSON only, with keys X and Y.” *user:* “<question or text>”.  
2. **API Call:** Use `anthropic.Anthropic` client to create a completion. We call `client.messages.create` with the list of messages and specify the Claude model (e.g. `"claude-2"` for the latest version) ([GitHub - anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python#:~:text=messages%3D%5B%20%7B%20,latest%22%2C%20%29%20print%28message.content)).  
3. **Parse JSON:** Attempt to parse the `message.content` returned by Claude into a dict. Claude usually follows the instruction, but if the output isn’t valid JSON, we’ll catch it.  
4. **Retry Logic:** If parsing fails, we can modify the prompt (e.g. reiterate the JSON-only requirement or simplify the request) and call again. We loop this a few times or until we get a valid JSON.

```python
import anthropic, json, time

client = anthropic.Anthropic(api_key="YOUR_ANTHROPIC_API_KEY")

messages = [
    {"role": "system", "content": "You are a helpful assistant that **only outputs JSON**. The JSON should contain an 'answer' field and a 'confidence' field (0-100)."},
    {"role": "user", "content": "<<<USER_INPUT_HERE>>>"} 
]

max_retries = 3
result_data = None
for attempt in range(max_retries):
    response = client.messages.create(
        model="claude-2", 
        messages=messages,
        max_tokens_to_sample=300  # adjust as needed
    )
    output_text = response.content  # Claude's answer as text
    try:
        result_data = json.loads(output_text)
        if 'answer' not in result_data or 'confidence' not in result_data:
            raise KeyError("Missing keys")
        break  # got valid JSON
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Claude attempt {attempt+1}: invalid JSON output - {e}")
        if attempt < max_retries - 1:
            time.sleep(1)
            # Strengthen the prompt for the next attempt
            messages[0]["content"] += " You MUST reply only in JSON format."
            continue
        else:
            raise

print(result_data)
```

**Explanation:** We use Anthropic’s chat API by providing a list of messages with roles ([GitHub - anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python#:~:text=messages%3D%5B%20%7B%20,latest%22%2C%20%29%20print%28message.content)). Claude 2 will read the system instruction and attempt to follow it. In many cases, simply instructing Claude like *“Output in JSON format with keys X, Y…”* will yield a correct JSON (Anthropic even documents examples of prompting for structured output ([Enforcing JSON Outputs in Commercial LLMs](https://datachain.ai/blog/enforcing-json-outputs-in-commercial-llms#:~:text=PROMPT%20%3D%20,))). However, to be safe, we parse the output and if it fails, we give Claude another chance, this time appending an even more forceful reminder in the system prompt. This usually corrects any deviations. Since Claude’s API doesn’t have an automatic JSON validator, this manual retry loop helps achieve reliability similar to the other APIs.

#### Additional Notes and Other APIs

The pattern above can be applied to **other LLM APIs** as well:

- **Azure OpenAI Service:** Follows the same schema as OpenAI’s API. (Azure OpenAI also supports `response_format={"type":"json_object"}` on the latest models, just like the OpenAI examples.)  
- **Hugging Face Inference API or Transformers Pipeline:** If using a local model via 🤗 Transformers, you would simply prompt the model to produce JSON and then parse the output. There’s no built-in JSON guarantee, so you may need to validate and retry as shown for Ollama. Some libraries (like [Jsonformer](https://github.com/1rgs/jsonformer)) can constrain generation to JSON, but those are outside standard APIs.  
- **Google PaLM API (Vertex AI):** Google’s PaLM 2 and upcoming models (Gemini) have their own methods for structured output (e.g., function calling or output schema). The general approach remains: provide a schema in the prompt or parameters, get the response, and parse it. (Google’s APIs may wrap the result in a JSON or Proto message automatically, depending on the client.)

Each API has its nuances, but the **core steps are the same**: *prompt clearly for JSON -> call the model -> parse result -> retry on error*. By following this pattern, you can implement the BEQA transform across different platforms while ensuring the output is a valid JSON object consistent with your schema.

**References:**

- OpenAI – Enabling JSON mode for GPT model ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,that%20parse%20into%20valid%20JSON)) ([python - OpenAI API: How do I enable JSON mode using the gpt-4-vision-preview model? - Stack Overflow](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model#:~:text=,is%20constrained%20to%20only%20generate))】  
- Groq API – JSON mode guarantees valid JSON or returns an error on failur ([GroqCloud](https://console.groq.com/docs/text-chat#:~:text=Error%20Code%3A))】  
- Cohere API – JSON output mode with `response_format={"type":"json_object"} ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=15%2016%20print%28res.message.content))】 (requires prompt to explicitly ask for JSO ([How do Structured Outputs Work? — Cohere](https://docs.cohere.com/v2/docs/structured-outputs#:~:text=))】)  
- Mistral AI – JSON mode available on all models via API with `response_format={"type":"json_object"} ([JSON mode | Mistral AI Large Language Models](https://docs.mistral.ai/capabilities/structured-output/json_mode/#:~:text=Users%20have%20the%20option%20to,of%20our%20models%20through%20API))】 (example usage of Mistral’s Python client for JSON outpu ([JSON mode | Mistral AI Large Language Models](https://docs.mistral.ai/capabilities/structured-output/json_mode/#:~:text=chat_response%20%3D%20client,%7D))】)  
- Ollama – Python library usage for local LLM calls (messages and content retrieval ([GitHub - ollama/ollama-python: Ollama Python library](https://github.com/ollama/ollama-python#:~:text=from%20ollama%20import%20chat%20from,ollama%20import%20ChatResponse))】  
- Anthropic Claude – Python SDK supports chat with role-based message ([GitHub - anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python#:~:text=messages%3D%5B%20%7B%20,latest%22%2C%20%29%20print%28message.content))】; prompting Claude for JSON output exampl ([Enforcing JSON Outputs in Commercial LLMs](https://datachain.ai/blog/enforcing-json-outputs-in-commercial-llms#:~:text=PROMPT%20%3D%20,))】.
