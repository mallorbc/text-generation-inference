from text_generation import Client
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="input file",required=True)
args = parser.parse_args()


client = Client("http://127.0.0.1:8080",timeout=180)
with open(args.file, "r") as f:
    prompt = f.read()
print(client.generate(prompt, max_new_tokens=1024).generated_text)

