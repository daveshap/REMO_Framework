import os
import re
import json
from glob import glob
from time import time, sleep
import openai


#####     simple helper functions


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


#####     OpenAI functions


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdfasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            #text = re.sub('[\r\n]+', '\n', text)
            #text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def get_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


#####     REMO functions


def read_chat_logs(folder):
    files = glob(os.path.join(folder, "*.txt"))
    logs = []
    for file in files:
        with open(file, 'r') as f:
            logs.append((file, f.read()))
    return logs


def read_existing_summaries(folder):
    files = glob(os.path.join(folder, "*.json"))
    existing_files = set()
    for file in files:
        summary = load_json(file)
        existing_files.update(summary['files'])
    return existing_files


def filter_new_logs(logs, existing_files):
    #return [log for log in logs if log[0] not in existing_files]
    return [log for log in logs if os.path.basename(log[0]) not in existing_files]    


def chunk_logs(logs, chunk_size=5):
    return [logs[i:i + chunk_size] for i in range(0, len(logs), chunk_size)]


def summarize_chunks(chunks):
    summaries = list()
    for chunk in chunks:
        text = "\n".join(["{}: {}".format(log[0].split('_')[-1].replace('.txt','').upper(), log[1]) for log in chunk])
        print('\n\n##################      INPUT\n\n', text)
        prompt = open_file('prompt_01_temporal_summary.txt').replace('<<INPUT>>', text)
        summary = gpt3_completion(prompt)
        print('\n\n##################      SUMMARY\n\n', summary)
        summaries.append((chunk, summary))
    return summaries


def create_summary_metadata(chunk, summary, embedding):
    start_time = float(chunk[0][0].split("_")[2])
    end_time = float(chunk[-1][0].split("_")[2])
    #files = [log[0] for log in chunk]
    files = [os.path.basename(log[0]) for log in chunk]

    metadata = {
        "files": files,
        "time_start": start_time,
        "time_end": end_time,
        "summary": summary,
        "vector": embedding
    }
    return metadata


def save_metadata(folder, metadata):
    timestamp = time()
    filename = os.path.join(folder, f"summary_{timestamp}.json")
    save_json(filename, metadata)


def main():
    print('\n\n##################      STARTING: reading existing layer 1 logs and layer 2 summaries... skipping processed logs...')
    logs = read_chat_logs("layer1_logs")
    existing_files = read_existing_summaries("layer2_temporal")
    new_logs = filter_new_logs(logs, existing_files)

    if not new_logs:
        print("\n\n##################      No new logs to process! Exiting...")
        return

    print('\n\n##################      Chunking and summarizing new logs...')
    chunks = chunk_logs(new_logs)
    summaries = summarize_chunks(chunks)

    print('\n\n##################      Saving summaries... almost done')
    if not os.path.exists("layer2_temporal"):
        os.makedirs("layer2_temporal")

    for chunk, summary in summaries:
        print('\n\n###### Chunk\n\n', chunk)
        print('\n\n###### Summary\n\n', summary)
        embedding = get_embedding(summary)
        print('\n\n###### Saving metadata....')
        metadata = create_summary_metadata(chunk, summary, embedding)
        save_metadata("layer2_temporal", metadata)

    print("\n\n##################      Processed and saved summaries to layer2_temporal folder. All done!")


if __name__ == "__main__":
    openai.api_key = open_file('key_openai.txt')
    main()