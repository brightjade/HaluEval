import os
import random
import openai
import time
import json
import argparse
import tiktoken

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm, trange


# openai.api_key = 'sk-'

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
                    args.model_name_or_path,
                    cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer


def load_model(args, output_dir=None):
    if output_dir:
        print(f"Loading model from {output_dir}...")
    else:
        print(f"Loading model from {args.model_name_or_path}...")

    model = AutoModelForCausalLM.from_pretrained(
                output_dir if output_dir else args.model_name_or_path,
                torch_dtype=args.dtype,
                device_map="auto",
                use_cache=False,
                resume_download=True,
                cache_dir=args.cache_dir if args.cache_dir else None)

    model.config.pretraining_tp = 1
    return model


def get_qa_response(tokenizer, model, model_type, question, answer, instruction):
    message = [
        {"role": "system", "content": "You are a hallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Question#: " + question +
                                    "\n#Answer#: " + answer +
                                    "\n#Your Judgement#: "} 
    ]
    prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"

    # Proprietary model
    if tokenizer is None or model is None:
        while True:
            try:
                if model_type == "gpt-3.5-turbo":
                    res = openai.ChatCompletion.create(
                        model_type="gpt-3.5-turbo",
                        messages=message,
                        temperature=0.0,
                    )
                    response = res['choices'][0]['message']['content']
                else:
                    res = openai.Completion.create(
                        engine=model_type,
                        prompt=prompt,
                        temperature=0.0
                    )
                    response = res["choices"][0]['text'].strip()
                break
            except openai.error.RateLimitError:
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(60)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)
    else:   # Open-source model
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            gen_tokens = model.generate(input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    return response


def get_dialogue_response(tokenizer, model, model_type, dialog, response, instruction):
    message = [
        {"role": "system", "content": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Dialogue History#: " + dialog +
                                    "\n#Response#: " + response +
                                    "\n#Your Judgement#: "}
    ]
    prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    
    if tokenizer is None or model is None:
        while True:
            try:
                if model_type == "gpt-3.5-turbo":
                    res = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=message,
                        temperature=0.0,
                    )
                    response = res['choices'][0]['message']['content']
                else:
                    res = openai.Completion.create(
                        model=model_type,
                        prompt=prompt,
                        temperature=0.0
                    )
                    response = res["choices"][0]['text'].strip()
                break
            except openai.error.RateLimitError:
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(60)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)

    else:   # Open-source model
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            gen_tokens = model.generate(input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    return response


def num_tokens_from_message(message, model_type="davinci"):
    encoding = tiktoken.encoding_for_model(model_type)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model_type="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model_type) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split()[:-1])
    prompt = prompt1 + prompt2
    return prompt


def get_summarization_response(tokenizer, model, model_type, document, summary, instruction):
    message = [
        {"role": "system", "content": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Document#: " + document +
                                    "\n#Summary#: " + summary +
                                    "\n#Your Judgement#: "}
    ]
    prompt1 = instruction + "\n\n#Document#: " + document
    prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    if model == "davinci":
        prompt = truncate_message(prompt1, prompt2)
    else:
        prompt = prompt1 + prompt2
    
    # Proprietary model
    if tokenizer is None or model is None:
        while True:
            try:
                if model == "gpt-3.5-turbo":
                    res = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=message,
                        temperature=0.0,
                    )
                    response = res['choices'][0]['message']['content']
                else:
                    res = openai.Completion.create(
                        model=model,
                        prompt=prompt,
                        temperature=0.0
                    )
                    response = res["choices"][0]['text'].strip()
                break
            except openai.error.RateLimitError:
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(60)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)
    else:  # Open-source model
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            gen_tokens = model.generate(input_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    return response


def evaluation_qa_dataset(tokenizer, model, model_type, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        failed = 0
        for i in (pbar := trange(len(data), position=0, leave=True)):
            knowledge = data[i]["knowledge"]
            question = data[i]["question"]
            hallucinated_answer = data[i]["hallucinated_answer"]
            right_answer = data[i]["right_answer"]

            if random.random() > 0.5:
                answer = hallucinated_answer
                ground_truth = "Yes"
            else:
                answer = right_answer
                ground_truth = "No"

            ans = get_qa_response(tokenizer, model, model_type, question, answer, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                # incorrect += 1
                failed += 1
                pbar.set_description_str(f"sample {i} fails...({ans})")
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
                incorrect += 1

            assert(gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            pbar.set_description_str(f"sample {i} success...({ans})")
            dump_jsonl(gen, output_path, append=True)

        print(f'{correct} correct samples, {incorrect} incorrect samples, {failed} failed samples, ' + \
              f'Acc: {correct/len(data)}, Acc w/o fail: {correct/(len(data)-failed)}')
        log_result(output_path,
                   f'{correct} correct samples, {incorrect} incorrect samples, {failed} failed samples, ' + \
                   f'Acc: {correct/len(data)}, Acc w/o fail: {correct/(len(data)-failed)}')


def evaluation_dialogue_dataset(tokenizer, model, model_type, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        failed = 0
        for i in (pbar := trange(len(data), position=0, leave=True)):
            knowledge = data[i]["knowledge"]
            dialog = data[i]["dialogue_history"]
            hallucinated_response = data[i]["hallucinated_response"]
            right_response = data[i]["right_response"]

            if random.random() > 0.5:
                response = hallucinated_response
                ground_truth = "Yes"
            else:
                response = right_response
                ground_truth = "No"

            ans = get_dialogue_response(tokenizer, model, model_type, dialog, response, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                # incorrect += 1
                failed += 1
                pbar.set_description_str(f"sample {i} fails...({ans})")
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            pbar.set_description_str(f"sample {i} success...({ans})")
            dump_jsonl(gen, output_path, append=True)

        print(f'{correct} correct samples, {incorrect} incorrect samples, {failed} failed samples, ' + \
              f'Acc: {correct/len(data)}, Acc w/o fail: {correct/(len(data)-failed)}')
        log_result(output_path,
                   f'{correct} correct samples, {incorrect} incorrect samples, {failed} failed samples, ' + \
                   f'Acc: {correct/len(data)}, Acc w/o fail: {correct/(len(data)-failed)}')

def evaluation_summarization_dataset(tokenizer, model, model_type, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        failed = 0
        for i in (pbar := trange(len(data), position=0, leave=True)):

            document = data[i]["document"]
            hallucinated_summary = data[i]["hallucinated_summary"]
            right_summary = data[i]["right_summary"]

            if random.random() > 0.5:
                summary = hallucinated_summary
                ground_truth = "Yes"
            else:
                summary = right_summary
                ground_truth = "No"

            ans = get_summarization_response(tokenizer, model, model_type, document, summary, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                # incorrect += 1
                failed += 1
                pbar.set_description_str(f"sample {i} fails...({ans})")
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            pbar.set_description_str(f"sample {i} success...({ans})")
            dump_jsonl(gen, output_path, append=True)

        print(f'{correct} correct samples, {incorrect} incorrect samples, {failed} failed samples, ' + \
              f'Acc: {correct/len(data)}, Acc w/o fail: {correct/(len(data)-failed)}')
        log_result(output_path,
                   f'{correct} correct samples, {incorrect} incorrect samples, {failed} failed samples, ' + \
                   f'Acc: {correct/len(data)}, Acc w/o fail: {correct/(len(data)-failed)}')


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


def log_result(output_path, result):
    output_path = output_path.replace(".json", ".txt")
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(result + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--task", type=str, default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--model_type", type=str, default="llama2-chat", help="model name")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="return empty string for proprietary model")
    parser.add_argument("--cache_dir", type=str, default="../../../.cache/", help="directory where open-source models are cached")
    parser.add_argument("--output_dir", type=str, default="", help="directory where fine-tuned models are saved. return empty string if not applicable")
    parser.add_argument("--save_dir", type=str, default=".results/", help="directory where evaluation results are saved")
    args = parser.parse_args()

    # Data type for model inference
    args.dtype = torch.bfloat16

    # Fix seed for reproducibility
    random.seed(42)

    instruction_file = "{}/{}_evaluation_instruction.txt".format(args.task, args.task)
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()

    # Need a new save directory for vanilla models
    if args.output_dir == "":
        args.save_dir = f"../../neg-inst/.checkpoints/{args.model_type}"
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = args.output_dir

    model_type = args.model_type
    output_path = os.path.join(args.save_dir, "{}/{}_halu_results.json".format(args.task, args.task))
    if not os.path.isfile(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, 'x')
    data = "../data/{}_data.json".format(args.task)

    # Load open-source model
    if args.model_name_or_path:
        tokenizer = load_tokenizer(args)
        model = load_model(args, args.output_dir)
        model.eval()
    else:
        tokenizer = None
        model = None

    if args.task == "qa":
        evaluation_qa_dataset(tokenizer, model, model_type, data, instruction, output_path)
    elif args.task == "dialogue":
        evaluation_dialogue_dataset(tokenizer, model, model_type, data, instruction, output_path)
    elif args.task == "summarization":
        evaluation_summarization_dataset(tokenizer, model, model_type, data, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")
