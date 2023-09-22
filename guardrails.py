import json
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import boto3
from pydantic import BaseModel, ValidationError
from typing import Optional

# Load AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

# Initialize the AWS client
client = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    endpoint_url="https://bedrock.us-east-1.amazonaws.com",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

MODEL = "anthropic.claude-v1"
accept = "application/json"
contentType = "application/json"


class PII(BaseModel):
    """Pydantic model for PII information."""
    explanation: Optional[str]
    name: Optional[str]
    number: Optional[str]
    email: Optional[str]
    address: Optional[str]
    social_security_number: Optional[str]
    credit_card_number: Optional[str]


async def query(prompt: str) -> str:
    """Perform a query against the model."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens_to_sample": 1000,
                "temperature": 0.0,
                "stop_sequences": ["Human:"],
            }
        )

        response = await loop.run_in_executor(
            pool, 
            lambda: client.invoke_model(
                body=body, 
                modelId=MODEL, 
                accept=accept, 
                contentType=contentType
            )
        )
    response_body = json.loads(response.get("body").read())
    return response_body.get("completion")


async def pii_detection(sentence: str) -> bool:
    """Detect if the sentence contains PII."""
    prompt = """
            Human: You can only respond with the word "True" or "False", where your answer indicates whether the text in the user's message contains PII.
            Do not explain your answer, and do not use punctuation.
            Your task is to identify whether the text extracted from your company files
            contains sensitive PII information that should not be shared with the broader company. Here are some things to look out for:
            - An email address that identifies a specific person in either the local-part or the domain
            - The postal address of a private residence (must include at least a street name)
            - The postal address of a public place (must include either a street name or business name)
            - Notes about hiring decisions with mentioned names of candidates. The user will send a document for you to analyze.
            - A phone number that identifies a specific person
            - A name that identifies a specific person \n\n
            """
    question = f"Does the following sentence contain PII? {sentence}"
    response = await query(prompt + question + "\nAI: ")
    response = response.strip()
    if response.startswith("True"):
        return True
    
    return False


async def pii_extraction(sentence: str) -> dict:
    """Extract PII information from a sentence."""
    prompt = """
    Human: Your job is to extract PII information from text. We will load the output as a JSON object so don't respond with anything but the JSON object. Your response should be in the following format:
    AI:
    ```
    {
        "name": "The name of the person",
        "number": "The phone number of the person",
        "email": "The email of the person",
        "address": "The address of the person",
        "social_security_number": "The social security number of the person",
        "credit_card_number": "The credit card number of the person",
        "explanation": "Explanation of your answer",
    }
    ```
    Human:
    """
    question = f"Extract the PII from the following sentence: {sentence}"

    retries = 0
    backoff_time = 1

    while retries < 3:
        try:
            response = await query(prompt + question + "\nAI: ")
            parsed_response = PII.parse_raw(response)
            return parsed_response.dict()
        
        except ValidationError as e:
            print(f"Validation failed: {e}")
            retries += 1
            await asyncio.sleep(backoff_time)
            # Linear backoff of 1, 2, 3 seconds
            backoff_time += 1
    return {"error": "Max retries reached"}
            

async def main():
    """Main function to loop user input and check for PII."""
    while True:
        sentence = input("Please enter a sentence to check for PII (type 'exit' to quit):\n")
        if sentence.lower() == 'exit':
            break

        contains_pii = await pii_detection(sentence)

        pii_color = "\033[92m" if contains_pii else "\033[91m"
        
        print(f"\033[94mSENTENCE:\033[0m '{sentence}'")
        print(f"\033[94mContains PII:\033[0m {pii_color}{contains_pii}\033[0m")

        if contains_pii:
            pii_info = await pii_extraction(sentence)
            print("\n\033[94mEXTRACTED PII:\033[0m")
            for key, value in pii_info.items():
                print(f"\033[93m{key.capitalize()}:\033[0m {value}")

        print("------------------------\n")


if __name__ == "__main__":
    asyncio.run(main())

