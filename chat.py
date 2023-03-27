import re
import os
import json
import openai
import tkinter as tk
from time import time, sleep
from threading import Thread
from tkinter import ttk, scrolledtext
from pprint import pprint as pp


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


def get_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def chatbot(messages, model="gpt-4"):
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            text = response['choices'][0]['message']['content']
            #pp(response)
            # save debug
            filename = 'debug/debug_%s_api_response.json' % time()
            save_json(filename, response)
            filename = 'debug/debug_%s_convo.json' % time()
            save_json(filename, messages)
            
            # check if conversation is getting too long
            if response['usage']['total_tokens'] >= 7800:
                a = messages.pop(1)
            
            # save chat log
            filename = 'chat_%s_raven.txt' % time()
            if not os.path.exists('layer1_logs'):
                os.makedirs('layer1_logs')
            save_file('layer1_logs/%s' % filename, text)
            return text
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            if 'maximum context length' in str(oops):
                a = messages.pop(1)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                print(f"Exiting due to excessive errors in API: {oops}")
                exit(1)
            print(f'Retrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)


#####     main loop


if __name__ == '__main__':
    openai.api_key = open_file('key_openai.txt')
    default_system = open_file('default_system.txt')
    conversation = list()
    conversation.append({'role': 'system', 'content': default_system})
    counter = 0
    while True:
        # get user input, save to file
        a = input('\n\nUSER: ')
        conversation.append({'role': 'user', 'content': a})
        filename = 'chat_%s_user.txt' % time()
        if not os.path.exists('layer1_logs'):
            os.makedirs('layer1_logs')
        save_file('layer1_logs/%s' % filename, a)
        
        # update SYSTEM based upon recalled info
        # TODO - flatten and embed current convo
        # TODO - semantic search for relevant topics in layer3_semantic
        # TODO - update system message
        
        # generate a response
        response = chatbot(conversation)
        conversation.append({'role': 'assistant', 'content': response})
        print('\n\nRAVEN: %s' % response)