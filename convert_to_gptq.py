from transformers import GPTQConfig,AutoModelForCausalLM,AutoTokenizer, AutoConfig
import argparse
from optimum.gptq import GPTQQuantizer, load_quantized_model
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bits", help="How many bits to use",default=4)
    parser.add_argument("-m","--model",help="model name",required=True)
    parser.add_argument("-o","--output",help="output file",required=True)
    parser.add_argument("--max_seq_len",help="max sequence length",default=None,type=int)


    args = parser.parse_args()

    if args.max_seq_len is None:
        config = AutoConfig.from_pretrained(args.model)
        config = config.to_dict()
        if "max_position_embeddings" in config:
            args.max_seq_len = config["max_position_embeddings"]

    #make sure the output dir doesn't exist
    assert not os.path.exists(args.output), "Output directory already exists"

    os.makedirs(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(args.model,device_map="auto")
    quantizer = GPTQQuantizer(bits=args.bits, dataset=["wikitext2","c4","c4-new","ptb","ptb-new"], model_seqlen = args.max_seq_len, tokenizer=tokenizer)

    quantized_model = quantizer.quantize_model(model, tokenizer)

    quantized_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

