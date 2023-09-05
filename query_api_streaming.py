from text_generation import Client
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="input file",required=True)
args = parser.parse_args()


client = Client("http://127.0.0.1:8080",timeout=100)
with open(args.file, "r") as f:
    prompt = f.read()


for response in client.generate_stream(prompt, max_new_tokens=256):
    if not response.token.special:
        token = response.token.text
        print(token, end="", flush=True)