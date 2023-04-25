import os
import yaml
import shutil
import openai
import numpy as np
from time import time, sleep
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import tensorflow_hub as hub


embedding_model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
)


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as outfile:
        outfile.write(content)


def save_yaml(filepath, data):
    with open(filepath, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)


def load_yaml(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def create_message(
    message: str, speaker: str, timestamp: float
) -> Dict[str, Any]:
    # Create message dictionary
    return {"content": message, "speaker": speaker, "timestamp": timestamp}


def save_message(root_folder, message: Dict[str, Any]):
    timestamp, speaker = message["timestamp"], message["speaker"]
    filename = f"chat_{timestamp}_{speaker}.yaml"
    filepath = os.path.join(root_folder, "L1_raw_logs", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_yaml(filepath, message)


def search_tree(root_folder, query):
    # TODO add a "forks" parameter to allow for branching relevance
    # TODO add a "fuzziness" parameter
    #  that can generate a random vector to modify the search query
    query_embedding = embedding_model([query]).numpy()
    level = 6
    taxonomy = []

    while level > 2:
        level_dir = os.path.join(root_folder, f"L{level}_summaries")
        if os.path.exists(level_dir) and os.listdir(level_dir):
            break
        level -= 1

    while level > 2:
        level_files = [
            os.path.join(level_dir, f)
            for f in os.listdir(level_dir)
            if f.endswith(".yaml")
        ]
        max_similarity = -1
        closest_file = None

        for file in level_files:
            data = load_yaml(file)
            similarity = (
                cosine_similarity(
                    query_embedding,
                    np.array(data["vector"]).reshape(1, -1),
                )
                [0][0]
            )

            if similarity > max_similarity:
                max_similarity = similarity
                closest_file = file

        closest_data = load_yaml(closest_file)
        taxonomy.append(closest_data["content"])

        if level == 2:
            break

        level -= 1
        level_dir = os.path.join(root_folder, f"L{level}_summaries")
        child_files = closest_data["files"]
        level_files = [os.path.join(level_dir, f) for f in child_files]

    return taxonomy


def rebuild_tree(root_folder: str, max_cluster_size: int = 10):
    # Delete all folders except L1_raw_logs, L2_message_pairs and .git
    for folder_name in os.listdir(root_folder):
        if folder_name not in {"L1_raw_logs", "L2_message_pairs", ".git"}:
            folder_path = os.path.join(root_folder, folder_name)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

    # Create L2 directory if it does not exist
    l2_message_pairs_dir = os.path.join(root_folder, "L2_message_pairs")
    if not os.path.exists(l2_message_pairs_dir):
        os.makedirs(l2_message_pairs_dir)

    # Process any missing messages in L1 to generate message pairs for L2
    process_missing_messages(root_folder)

    # Cluster L2 message pairs using cosine similarity, up to 10 per cluster
    clusters = cluster_elements(
        root_folder, "L2_message_pairs", max_cluster_size
    )

    # Create summaries and save them in the next rank (L3_summaries)
    create_summaries(
        root_folder, clusters, f"L3_summaries", "L2_message_pairs"
    )

    # If top rank (e.g. L3_summaries) has > max_cluster_size files,
    # repeat process, creating new taxonomical ranks
    current_rank = 3
    while True:
        # calculate clusters at new rank
        clusters = cluster_elements(
            root_folder, f"L{current_rank}_summaries", max_cluster_size
        )

        # summarize those clusters
        create_summaries(
            root_folder,
            clusters,
            f"L{current_rank + 1}_summaries",
            f"L{current_rank}_summaries",
        )
        current_rank += 1

        # if clusters less than max cluster size, we are done :)
        if len(clusters) <= max_cluster_size:
            break


def process_missing_messages(root_folder: str):
    raw_logs_dir = os.path.join(root_folder, "L1_raw_logs")
    message_pairs_dir = os.path.join(root_folder, "L2_message_pairs")

    # Get list of processed message filenames
    processed_messages = set(os.listdir(message_pairs_dir))

    # Sort raw log files by timestamp
    raw_log_files = os.listdir(raw_logs_dir)

    for i in range(len(raw_log_files) - 1):
        file1_path = os.path.join(raw_logs_dir, raw_log_files[i])
        file2_path = os.path.join(raw_logs_dir, raw_log_files[i + 1])

        # Check if message pair is already processed
        message_pair_filename = f"pair_{raw_log_files[i + 1]}"
        if message_pair_filename in processed_messages:
            continue

        # Load raw log data
        file1_data = load_yaml(file1_path)
        file2_data = load_yaml(file2_path)

        context = file1_data["content"]
        response = file2_data["content"]
        speaker = file2_data["speaker"]
        timestamp = file2_data["timestamp"]

        combined_text = context + " --- " + response
        embedding = embedding_model([combined_text]).numpy().tolist()

        message_pair_data = {
            "content": combined_text,
            "speaker": speaker,
            "timestamp": timestamp,
            "vector": embedding,
        }

        # Save message pair in L2_message_pairs folder
        message_pair_path = os.path.join(
            message_pairs_dir, message_pair_filename
        )
        save_yaml(message_pair_path, message_pair_data)


def create_summaries(
    root_folder: str,
    clusters: List[List[str]],
    target_folder: str,
    source_folder: str,
):
    source_folder_path = os.path.join(root_folder, source_folder)
    target_folder_path = os.path.join(root_folder, target_folder)
    os.makedirs(target_folder_path, exist_ok=True)

    for i, cluster in enumerate(clusters):
        # Combine content of cluster elements
        combined_content = ""
        files = []
        for file in cluster:
            filepath = os.path.join(source_folder_path, file)
            data = load_yaml(filepath)
            combined_content += data["content"] + " \n---\n "
            files.append(filepath)

        # Generate summary with LLM
        summary = quick_summarize(combined_content)

        # Create embedding for summary
        summary_embedding = embedding_model([summary]).numpy().tolist()

        # Save summary in target folder
        summary_data = {
            "content": summary,
            "vector": summary_embedding,
            "files": files,
            "timestamp": time(),
        }

        timestamp = time()
        summary_filename = f"summary_{i}_{timestamp}.yaml"
        summary_filepath = os.path.join(target_folder_path, summary_filename)
        save_yaml(summary_filepath, summary_data)


def cluster_elements(
    root_folder: str, target_folder: str, max_cluster_size: int = 10
) -> List[List[str]]:
    folder_path = os.path.join(root_folder, target_folder)
    yaml_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".yaml")
    ]

    # Load vectors
    vectors = []
    for file in yaml_files:
        filepath = os.path.join(folder_path, file)
        data = load_yaml(filepath)
        vectors.append(data["vector"][0])

    # Calculate number of clusters
    num_clusters = int(np.ceil(len(yaml_files) / max_cluster_size))

    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    # Group files by cluster
    clusters = [[] for _ in range(num_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(yaml_files[i])

    return clusters


def maintain_tree(root_folder: str):
    l2_message_pairs_dir = os.path.join(root_folder, "L2_message_pairs")

    # Create L2 directory if it does not exist
    if not os.path.exists(l2_message_pairs_dir):
        os.makedirs(l2_message_pairs_dir)

    # Get list of files in L2 before processing missing messages
    l2_files_before = set(os.listdir(l2_message_pairs_dir))

    # Process missing messages to generate new message pairs in L2
    process_missing_messages(root_folder)

    # Get list of files in L2 after processing missing messages
    l2_files_after = set(os.listdir(l2_message_pairs_dir))

    # Calculate the difference between the two lists
    # to obtain the new message pairs
    new_message_pairs = l2_files_after - l2_files_before
    # new_message_pairs = [
    #     os.path.join("L2_message_pairs", f)
    #     for f in l2_files_after - l2_files_before
    # ]

    # Iterate through new files in L2
    # and check cosine similarity to files in L3
    integrate_new_elements(
        root_folder, "L3_summaries", new_message_pairs, 0.75
    )


def integrate_new_elements(
    root_folder: str,
    target_folder: str,
    new_elements: List[str],
    threshold: float,
):
    target_dir = os.path.join(root_folder, target_folder)

    # Create target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for new_element in new_elements:
        new_element_path = os.path.join(
            root_folder, "L2_message_pairs", new_element
        )
        new_element_data = load_yaml(new_element_path)
        new_element_vector = (
            np.array(new_element_data["vector"])
            .reshape(1, -1)
        )

        max_similarity = -1
        closest_file = None

        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            file_data = load_yaml(file_path)
            file_vector = np.array(file_data["vector"]).reshape(1, -1)

            similarity = (
                cosine_similarity(new_element_vector, file_vector)
                [0][0]
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                closest_file = file

        if max_similarity > threshold:
            # Update the corresponding summary
            # and record the name of the modified file
            closest_file_path = os.path.join(target_dir, closest_file)
            closest_file_data = load_yaml(closest_file_path)
            closest_file_data["files"].append(new_element)

            combined_content = (
                closest_file_data["content"]
                + " --- "
                + new_element_data["content"]
            )
            updated_summary = quick_summarize(combined_content)
            updated_summary_embedding = (
                embedding_model([updated_summary]).numpy().tolist()
            )

            closest_file_data["content"] = updated_summary
            closest_file_data["vector"] = updated_summary_embedding
            closest_file_data["timestamp"] = time()

            save_yaml(closest_file_path, closest_file_data)
        else:
            # Create a new summary for the new_element
            combined_content = new_element_data["content"]
            new_summary = quick_summarize(combined_content)
            new_summary_embedding = (
                embedding_model([new_summary]).numpy().tolist()
            )

            new_summary_data = {
                "content": new_summary,
                "vector": new_summary_embedding,
                "files": [new_element],
                "timestamp": time(),
            }

            new_summary_filename = (
                f"summary_{len(os.listdir(target_dir))}.yaml"
            )
            new_summary_filepath = os.path.join(
                target_dir, new_summary_filename
            )
            save_yaml(new_summary_filepath, new_summary_data)


def quick_summarize(text):
    max_chunk_size = 10000

    if len(text) <= max_chunk_size:
        prompt = (
            (
                "Write a detailed summary of the following:"
                "\n\n%s"
                "\n\nDETAILED SUMMARY:"
            )
            % text
        )
        response = gpt3_completion(prompt)
        return response
    else:
        # Split the text into evenly sized chunks
        num_chunks = int(np.ceil(len(text) / max_chunk_size))
        chunk_size = int(np.ceil(len(text) / num_chunks))
        text_chunks = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]

        # Summarize each chunk
        summaries = []
        for chunk in text_chunks:
            prompt = (
                (
                    "Write a detailed summary of the following:"
                    "\n\n%s"
                    "\n\nDETAILED SUMMARY:"
                )
                % chunk
            )
            response = gpt3_completion(prompt)
            summaries.append(response)

        # Concatenate the summaries and return the result
        final_summary = " ".join(summaries)
        return final_summary


def gpt3_completion(
    prompt,
    engine="text-davinci-003",
    temp=0.0,
    top_p=1.0,
    tokens=1000,
    freq_pen=0.0,
    pres_pen=0.0,
    stop=["asdfasdfasdf"],
):
    openai.api_key = open_file("key_openai.txt")
    
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding="ASCII", errors="ignore").decode()
    
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
                stop=stop,
            )
            text = response["choices"][0]["text"].strip()
            # text = re.sub('[\r\n]+', '\n', text)
            # text = re.sub('[\t ]+', ' ', text)
            
            filename = "%s_gpt3.txt" % time()
            if not os.path.exists("gpt3_logs"):
                os.makedirs("gpt3_logs")
                
            save_file(
                "gpt3_logs/%s" % filename,
                prompt + "\n\n==========\n\n" + text,
            )
            
            return text
        
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print("Error communicating with OpenAI:", oops)
            sleep(1)
